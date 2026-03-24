"""
Bilibili 业务层 workflow。

这些流程跨越 DB、模板渲染和远端 API，不应继续留在 uploader adapter 文件中。
"""
import json
import sqlite3
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from vat.services.playlist_service import PlaylistService
from vat.models import TaskStep
from vat.uploaders.template import render_upload_metadata
from vat.utils.logger import setup_logger

if TYPE_CHECKING:
    from vat.uploaders.bilibili import BilibiliUploader


logger = setup_logger("bilibili_workflows")


def season_sync(db, uploader: "BilibiliUploader", playlist_id: str) -> Dict[str, Any]:
    """
    批量将已上传但未入集的视频添加到 B站 合集，然后按 #数字 排序。
    同时执行诊断检查，发现并汇报异常状态的视频。
    """
    playlist_service = PlaylistService(db)
    videos = playlist_service.get_playlist_videos(playlist_id)

    upload_completed_no_aid = []
    for v in videos:
        meta = v.metadata or {}
        if not meta.get('bilibili_aid'):
            if db.is_step_completed(v.id, TaskStep.UPLOAD):
                upload_completed_no_aid.append((v.id, v.title[:40] if v.title else v.id))

    if upload_completed_no_aid:
        logger.warning(
            f"[诊断] {len(upload_completed_no_aid)} 个视频 upload 已完成但无 bilibili_aid"
            f"（上传成功但 DB 未记录 aid，需手动核实）:"
        )
        for vid, title in upload_completed_no_aid:
            logger.warning(f"  - {vid}: {title}")

    pending = []
    for v in videos:
        meta = v.metadata or {}
        aid = meta.get('bilibili_aid')
        target_season = meta.get('bilibili_target_season_id')
        already_added = meta.get('bilibili_season_added', False)

        if aid and target_season and not already_added:
            pending.append((v, int(aid), int(target_season)))

    result = {
        'total': len(pending),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'season_ids': set(),
        'failed_videos': [],
        'diagnostics': {
            'upload_completed_no_aid': upload_completed_no_aid,
            'aid_not_found_on_bilibili': [],
        },
    }

    if not pending:
        logger.info(f"Playlist {playlist_id}: 没有待同步的视频")
    else:
        logger.info(f"Playlist {playlist_id}: 找到 {len(pending)} 个待同步视频")

        for i, (video, aid, season_id) in enumerate(pending):
            result['season_ids'].add(season_id)
            try:
                add_result = uploader.add_to_season(aid, season_id)
                if add_result:
                    result['success'] += 1
                    updated_meta = dict(video.metadata or {})
                    updated_meta['bilibili_season_added'] = True
                    db.update_video(video.id, metadata=updated_meta)
                    logger.info(f"✓ {video.title or video.id} -> 合集 {season_id}")
                else:
                    result['failed'] += 1
                    result['failed_videos'].append(video.id)
                    logger.warning(f"✗ {video.title or video.id} -> 合集 {season_id} 失败")
            except Exception as e:
                result['failed'] += 1
                result['failed_videos'].append(video.id)
                logger.error(f"✗ {video.title or video.id} -> 合集 {season_id} 异常: {e}")
            if i < len(pending) - 1:
                time.sleep(3)

    aid_not_found = []
    for vid_id in result['failed_videos']:
        v = next((v for v in videos if v.id == vid_id), None)
        if not v:
            continue
        meta = v.metadata or {}
        aid = meta.get('bilibili_aid')
        if not aid:
            continue
        try:
            session = uploader._get_authenticated_session()
            resp = session.get(
                'https://member.bilibili.com/x/client/archive/view',
                params={'aid': int(aid)},
                timeout=10
            )
            data = resp.json()
            if data.get('code') != 0:
                aid_not_found.append((vid_id, int(aid), v.title[:40] if v.title else vid_id))
        except Exception:
            pass

    if aid_not_found:
        result['diagnostics']['aid_not_found_on_bilibili'] = aid_not_found
        logger.warning(
            f"[诊断] {len(aid_not_found)} 个视频有 bilibili_aid 但 B站查不到"
            f"（可能已被删除或 aid 记录错误）:"
        )
        for vid, aid, title in aid_not_found:
            logger.warning(f"  - {vid} (av{aid}): {title}")

    all_season_ids = set()
    for v in videos:
        meta = v.metadata or {}
        sid = meta.get('bilibili_target_season_id')
        if sid and meta.get('bilibili_aid') and meta.get('bilibili_season_added'):
            all_season_ids.add(int(sid))
    all_season_ids.update(result['season_ids'])

    desync_fixed = 0
    desync_failed = 0
    for sid in all_season_ids:
        try:
            season_data = uploader.get_season_episodes(sid)
            if not season_data:
                continue
            actual_aids = {ep['aid'] for ep in season_data.get('episodes', [])}

            for v in videos:
                meta = v.metadata or {}
                if (
                    meta.get('bilibili_season_added')
                    and meta.get('bilibili_target_season_id')
                    and int(meta['bilibili_target_season_id']) == sid
                ):
                    aid = meta.get('bilibili_aid')
                    if aid and aid not in actual_aids:
                        title = (v.title or v.id)[:40]
                        logger.warning(
                            f"[一致性修复] {v.id} (av{aid}) DB 标记已入集但实际不在合集 {sid}，重新添加..."
                        )
                        if uploader.add_to_season(int(aid), sid):
                            desync_fixed += 1
                            logger.info(f"  ✓ 重新添加成功: {title}")
                        else:
                            desync_failed += 1
                            updated_meta = dict(meta)
                            updated_meta['bilibili_season_added'] = False
                            db.update_video(v.id, metadata=updated_meta)
                            logger.error(f"  ✗ 重新添加失败: {title}（已修正 DB 标记为 False）")
                        time.sleep(3)
        except Exception as e:
            logger.warning(f"一致性校验合集 {sid} 异常: {e}")

    if desync_fixed or desync_failed:
        result['diagnostics']['desync_fixed'] = desync_fixed
        result['diagnostics']['desync_failed'] = desync_failed
        logger.info(
            f"[一致性校验] 修复 {desync_fixed} 个不一致视频"
            f"{f'，{desync_failed} 个修复失败' if desync_failed else ''}"
        )

    for season_id in all_season_ids:
        try:
            if uploader.auto_sort_season(season_id):
                logger.info(f"✓ 合集 {season_id} 排序完成")
            else:
                logger.warning(f"⚠ 合集 {season_id} 排序失败")
        except Exception as e:
            logger.warning(f"⚠ 合集 {season_id} 排序异常: {e}")

    diag = result['diagnostics']
    diag_msgs = []
    if diag['upload_completed_no_aid']:
        diag_msgs.append(f"{len(diag['upload_completed_no_aid'])} 个 upload 完成但无 aid")
    if diag['aid_not_found_on_bilibili']:
        diag_msgs.append(f"{len(diag['aid_not_found_on_bilibili'])} 个 aid 在B站查不到")
    if diag.get('desync_fixed'):
        diag_msgs.append(f"{diag['desync_fixed']} 个不一致视频已修复")
    if diag.get('desync_failed'):
        diag_msgs.append(f"{diag['desync_failed']} 个不一致视频修复失败")

    diag_str = f"，诊断问题: {'; '.join(diag_msgs)}" if diag_msgs else ""
    logger.info(
        f"Season sync 完成: {result['success']} 成功, "
        f"{result['failed']} 失败, {result['skipped']} 跳过{diag_str}"
    )
    return result


def resync_video_info(
    db: Any,
    uploader: "BilibiliUploader",
    config: Any,
    aid: int,
    callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    从 DB 和模板重新渲染视频元信息（title/desc/tags/tid）并同步到 B站。
    """
    _cb = callback or (lambda msg: None)
    result = {'success': False, 'title': '', 'message': ''}

    conn = sqlite3.connect(str(db.db_path))
    c = conn.cursor()
    c.execute("SELECT id, metadata FROM videos WHERE metadata LIKE ?", (f'%{aid}%',))
    rows = c.fetchall()
    conn.close()

    video_id = None
    for (vid, meta_str) in rows:
        if not meta_str:
            continue
        meta = json.loads(meta_str)
        if str(meta.get('bilibili_aid')) == str(aid):
            video_id = vid
            break

    if not video_id:
        result['message'] = f'DB 中未找到 av{aid} 对应的视频记录'
        _cb(result['message'])
        return result

    video = db.get_video(video_id)
    if not video:
        result['message'] = f'无法加载视频 {video_id}'
        _cb(result['message'])
        return result

    meta = video.metadata or {}
    translated = meta.get('translated', {})
    if not translated:
        result['message'] = f'视频 {video_id} 缺少翻译数据，无法渲染模板'
        _cb(result['message'])
        return result

    video_playlists = db.get_video_playlists(video_id)
    playlist_info = None
    if video_playlists:
        pl_id = video_playlists[0]
        playlist = db.get_playlist(pl_id)
        if playlist:
            pl_upload_config = (playlist.metadata or {}).get('upload_config', {})
            pv_info = db.get_playlist_video_info(pl_id, video_id)
            upload_order_index = pv_info.get('upload_order_index', 0) if pv_info else 0
            if not upload_order_index:
                upload_order_index = meta.get('upload_order_index', 0) or 0

            playlist_info = {
                'name': playlist.title,
                'id': pl_id,
                'index': upload_order_index,
                'uploader_name': pl_upload_config.get('uploader_name', ''),
            }

    bilibili_config = config.uploader.bilibili
    templates = {}
    if bilibili_config.templates:
        templates = {
            'title': bilibili_config.templates.title,
            'description': bilibili_config.templates.description,
            'custom_vars': bilibili_config.templates.custom_vars,
        }

    rendered = render_upload_metadata(video, templates, playlist_info)
    new_title = rendered['title'][:80]
    new_desc = rendered['description'][:2000]

    all_tags = []
    for t in (translated.get('tags_translated', []) or []):
        if t and t not in all_tags:
            all_tags.append(t)
    for t in (translated.get('tags_generated', []) or []):
        if t and t not in all_tags:
            all_tags.append(t)
    for t in (bilibili_config.default_tags or []):
        if t and t not in all_tags:
            all_tags.append(t)
    new_tags = all_tags[:12] if all_tags else None
    new_tid = translated.get('recommended_tid') or bilibili_config.default_tid

    _cb(f"渲染结果: title={new_title[:50]}...")
    _cb(f"  desc={len(new_desc)}字, tags={new_tags}, tid={new_tid}")

    ok = uploader.edit_video_info(
        aid=aid,
        title=new_title,
        desc=new_desc,
        tags=new_tags,
        tid=new_tid,
    )

    if ok:
        result['success'] = True
        result['title'] = new_title
        result['message'] = f'av{aid} 元信息已同步'
        _cb(f"  ✅ {result['message']}")
    else:
        result['message'] = f'av{aid} edit_video_info 调用失败'
        _cb(f"  ❌ {result['message']}")

    return result


def resync_season_video_infos(
    db: Any,
    uploader: "BilibiliUploader",
    config: Any,
    season_id: int,
    delay_seconds: float = 1.0,
    callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    按合集批量刷新视频元信息。
    """
    _cb = callback or (lambda msg: None)
    result = {
        'success': False,
        'season_id': season_id,
        'refreshed': 0,
        'failed': 0,
        'skipped': 0,
        'details': [],
        'message': '',
    }

    season_info = uploader.get_season_episodes(season_id)
    if not season_info:
        result['message'] = '无法获取合集信息'
        _cb(result['message'])
        return result

    episodes = season_info.get('episodes', [])
    if not episodes:
        result['success'] = True
        result['message'] = f'合集 {season_id} 中暂无视频'
        _cb(result['message'])
        return result

    total = len(episodes)
    _cb(f"开始同步合集 {season_id} 元信息，共 {total} 个视频")

    for idx, ep in enumerate(episodes):
        aid = ep.get('aid')
        if not aid:
            result['skipped'] += 1
            result['details'].append({
                'aid': None,
                'success': False,
                'title': '',
                'message': '合集条目缺少 aid，已跳过',
            })
            _cb(f"[{idx + 1}/{total}] 缺少 aid，跳过")
        else:
            _cb(f"[{idx + 1}/{total}] 开始同步 av{aid}")
            try:
                item = resync_video_info(db, uploader, config, int(aid), callback=_cb)
            except Exception as e:
                logger.error(f"批量同步合集 {season_id} 的 av{aid} 异常: {e}", exc_info=True)
                item = {'success': False, 'title': '', 'message': str(e)}

            if item.get('success'):
                result['refreshed'] += 1
            else:
                result['failed'] += 1

            result['details'].append({
                'aid': int(aid),
                'success': bool(item.get('success')),
                'title': item.get('title', ''),
                'message': item.get('message', ''),
            })

        if idx < total - 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

    result['success'] = True
    result['message'] = (
        f"合集 {season_id} 元信息同步完成：成功 {result['refreshed']}，"
        f"失败 {result['failed']}，跳过 {result['skipped']}"
    )
    _cb(result['message'])
    return result
