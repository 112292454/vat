"""
CLI tools 子命令组

提供统一的 tools 入口，支持 JobManager 作为子进程调用。
每个子命令包装现有功能逻辑，输出标准化进度标记：
  - [N%]      进度百分比
  - [SUCCESS]  任务成功完成
  - [FAILED]   任务失败

用法:
  vat tools fix-violation --aid 12345
  vat tools sync-playlist --playlist PL_xxx
  vat tools refresh-playlist --playlist PL_xxx
  vat tools retranslate-playlist --playlist PL_xxx
  vat tools upload-sync --playlist PL_xxx
  vat tools update-info --playlist PL_xxx
  vat tools sync-db --season 12345 --playlist PL_xxx
  vat tools season-sync --playlist PL_xxx
"""
import sys
import click
from pathlib import Path

from .commands import cli, get_config, get_logger


def _emit(msg: str):
    """输出消息到 stdout（确保子进程模式下日志可被 JobManager 读取）"""
    click.echo(msg, nl=True)
    sys.stdout.flush()


def _success(msg: str = ""):
    """输出成功标记"""
    _emit(f"[SUCCESS] {msg}" if msg else "[SUCCESS]")


def _failed(msg: str = ""):
    """输出失败标记"""
    _emit(f"[FAILED] {msg}" if msg else "[FAILED]")


def _progress(pct: int):
    """输出进度标记（0-100）"""
    _emit(f"[{pct}%]")


def _get_config_and_db(ctx):
    """获取 config 和 db 实例"""
    from ..config import load_config
    from ..database import Database
    config = load_config()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    return config, db


def _get_bilibili_uploader(config):
    """创建 BilibiliUploader 实例"""
    from ..uploaders.bilibili import BilibiliUploader
    bilibili_config = config.uploader.bilibili
    project_root = Path(__file__).parent.parent.parent
    cookies_file = project_root / bilibili_config.cookies_file
    return BilibiliUploader(
        cookies_file=str(cookies_file),
        line=bilibili_config.line,
        threads=bilibili_config.threads
    )


def _get_playlist_service(config, db):
    """创建 PlaylistService 实例"""
    from ..services.playlist_service import PlaylistService
    from ..downloaders import YouTubeDownloader
    downloader = YouTubeDownloader(
        proxy=config.get_stage_proxy("downloader"),
        video_format=config.downloader.youtube.format,
        cookies_file=config.downloader.youtube.cookies_file,
        remote_components=config.downloader.youtube.remote_components,
    )
    return PlaylistService(db, downloader)


# =============================================================================
# tools 命令组
# =============================================================================

@cli.group()
def tools():
    """工具任务（支持 job 化管理，可由 WebUI 或 CLI 发起）"""
    pass


# =============================================================================
# fix-violation: 修复被退回的B站稿件
# =============================================================================

@tools.command('fix-violation')
@click.option('--aid', required=True, type=int, help='B站稿件 AV号')
@click.option('--video-path', type=click.Path(), default=None, help='本地视频文件路径')
@click.option('--margin', default=1.0, type=float, help='违规区间安全边距（秒）')
@click.option('--mask-text', default='此处内容因平台合规要求已被遮罩', help='遮罩文字')
@click.option('--dry-run', is_flag=True, help='仅遮罩不上传')
@click.option('--max-rounds', default=10, type=int, help='最大修复轮次（默认10，设为1则不自动循环）')
@click.option('--wait-seconds', default=0, type=int,
              help='每轮修复后等待审核的秒数（默认0=上传耗时*2，下限900秒）')
def tools_fix_violation(aid, video_path, margin, mask_text, dry_run, max_rounds, wait_seconds):
    """修复被退回的B站稿件（累积式遮罩+上传替换，默认自动循环直到通过）
    
    自动循环流程：mask→上传→等待审核→check是否通过→如不通过则再次修复。
    等待时间默认为上传耗时的2倍（下限15分钟）。
    修复成功后自动尝试添加到B站合集（如有配置）。
    """
    import time as _time
    from ..config import load_config
    from ..database import Database

    config = load_config()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    logger = get_logger()

    try:
        uploader = _get_bilibili_uploader(config)

        # 查找本地视频（只做一次，所有轮次复用）
        local_video = None
        if video_path:
            local_video = Path(video_path)
            if not local_video.exists():
                _failed(f"文件不存在: {video_path}")
                return
        else:
            local_video = _find_local_video_for_aid(aid, config, db, uploader)
            if local_video:
                _emit(f"找到本地文件: {local_video}")
            else:
                _emit("⚠️ 本地文件未找到，将从B站下载（质量会降低）")

        from .commands import _get_previous_violation_ranges, _save_violation_ranges

        # 读取累积的 violation_ranges
        all_previous_ranges = _get_previous_violation_ranges(db, aid)
        if all_previous_ranges:
            _emit(f"历史已 mask: {len(all_previous_ranges)} 段")

        for round_num in range(max_rounds):
            _emit(f"\n{'='*40}")
            _emit(f"第 {round_num + 1}/{max_rounds} 轮修复")
            _emit(f"{'='*40}")

            result = uploader.fix_violation(
                aid=aid,
                video_path=local_video,
                mask_text=mask_text,
                margin_sec=margin,
                previous_ranges=all_previous_ranges,
                dry_run=dry_run,
                callback=_emit,
            )

            if not result['success']:
                _failed(result['message'])
                return

            # 更新累积 ranges 并持久化
            all_previous_ranges = result['all_ranges']
            _save_violation_ranges(db, aid, all_previous_ranges)

            # dry-run 模式不循环
            if dry_run:
                _progress(100)
                _success(f"dry-run 完成，遮罩文件: {result.get('masked_path', '')}")
                return

            # 最后一轮不再等待
            if round_num >= max_rounds - 1:
                _emit(f"已达最大轮次 ({max_rounds})，最后一轮修复已提交")
                break

            # 计算等待时间：用户指定 > 上传耗时*2 > 下限900秒(15分钟)
            upload_dur = result.get('upload_duration', 0)
            if wait_seconds > 0:
                actual_wait = wait_seconds
            else:
                actual_wait = max(int(upload_dur * 2), 900)
            _emit(f"上传耗时 {upload_dur:.0f}s，等待审核 {actual_wait}s ({actual_wait // 60} 分钟)...")

            # 等待，每10分钟输出一次日志
            waited = 0
            log_interval = 600  # 10分钟
            while waited < actual_wait:
                chunk = min(log_interval, actual_wait - waited)
                _time.sleep(chunk)
                waited += chunk
                remaining = actual_wait - waited
                if remaining > 0:
                    _emit(f"  等待审核中... 剩余 {remaining // 60} 分钟")

            # 检查是否仍被退回
            _emit("检查审核状态...")
            rejected = uploader.get_rejected_videos()
            still_rejected = any(v['aid'] == aid for v in rejected)

            if not still_rejected:
                _emit(f"av{aid} 已不在退回列表中（已通过审核或仍在审核中）")
                # 尝试添加到合集
                _try_add_to_season(db, uploader, aid)
                _progress(100)
                _success(f"av{aid} 修复流程完成")
                return

            _emit(f"av{aid} 仍被退回，准备下一轮修复...")
            # 下一轮 fix_violation 会重新 get_rejected_videos 获取最新违规信息

        # 所有轮次结束（最后一轮已提交但未等待检查）
        # 尝试添加到合集（视频可能在审核中，add_to_season 内部会处理已存在的情况）
        _try_add_to_season(db, uploader, aid)
        _progress(100)
        _success(f"av{aid} 已完成 {max_rounds} 轮修复提交")

    except Exception as e:
        logger.error(f"fix-violation 异常: {e}", exc_info=True)
        _failed(str(e))


def _try_add_to_season(db, uploader, aid: int):
    """修复成功后尝试将视频添加到B站合集（如果 DB 中配置了 target_season_id）"""
    import json, sqlite3
    logger = get_logger()
    try:
        conn = sqlite3.connect(str(db.db_path))
        c = conn.cursor()
        c.execute("SELECT id, metadata FROM videos WHERE metadata LIKE ?", (f'%{aid}%',))
        rows = c.fetchall()
        conn.close()

        for (video_id, meta_str) in rows:
            if not meta_str:
                continue
            meta = json.loads(meta_str)
            if str(meta.get('bilibili_aid')) != str(aid):
                continue
            season_id = meta.get('bilibili_target_season_id')
            if not season_id:
                _emit(f"av{aid} 未配置目标合集，跳过 add-to-season")
                return
            if meta.get('bilibili_season_added'):
                _emit(f"av{aid} 已在合集 {season_id} 中，跳过")
                return
            _emit(f"尝试将 av{aid} 添加到合集 {season_id}...")
            ok = uploader.add_to_season(aid, int(season_id))
            if ok:
                meta['bilibili_season_added'] = True
                db.update_video(video_id, metadata=meta)
                _emit(f"  已添加到合集 {season_id}")
            else:
                _emit(f"  添加到合集失败（不影响修复结果）")
            return
        _emit(f"DB 中未找到 av{aid} 对应的视频记录，跳过 add-to-season")
    except Exception as e:
        logger.warning(f"add-to-season 异常（不影响修复结果）: {e}")


def _find_local_video_for_aid(aid, config, db, uploader):
    """通过 B站稿件信息查找本地视频文件（复用 CLI 的查找逻辑）"""
    import re

    try:
        detail = uploader.get_archive_detail(aid)
        if detail:
            archive = detail.get('archive', {})
            source = archive.get('source', '')
            desc = archive.get('desc', '')
            full_desc = uploader._get_full_desc(aid)

            yt_video_id = None
            for text in [source, full_desc or desc, desc]:
                yt_match = re.search(r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)', text)
                if yt_match:
                    yt_video_id = yt_match.group(1)
                    break

            if yt_video_id:
                video = db.get_video(yt_video_id)
                if video:
                    candidates = []
                    if video.output_dir:
                        candidates.append(Path(video.output_dir) / "final.mp4")
                    vid_dir = Path(config.storage.output_dir) / video.id
                    candidates.append(vid_dir / "final.mp4")
                    candidates.append(vid_dir / f"{video.id}.mp4")
                    for c in candidates:
                        if c.exists():
                            return c
    except Exception:
        pass
    return None


# =============================================================================
# sync-playlist: 同步 Playlist（增量更新）
# =============================================================================

@tools.command('sync-playlist')
@click.option('--playlist', required=True, help='Playlist ID')
@click.option('--url', default=None, help='Playlist URL（首次添加时需要）')
@click.option('--fetch-dates', is_flag=True, default=True, help='获取上传日期')
def tools_sync_playlist(playlist, url, fetch_dates):
    """同步 YouTube Playlist（增量更新视频列表）"""
    from ..config import load_config
    from ..database import Database

    config = load_config()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)

    try:
        pl = db.get_playlist(playlist)
        if not pl:
            _failed(f"Playlist 不存在: {playlist}")
            return

        playlist_url = url or pl.source_url
        if not playlist_url:
            _failed(f"Playlist {playlist} 无 source_url，需通过 --url 指定")
            return

        service = _get_playlist_service(config, db)
        _emit(f"同步 Playlist: {pl.title}")
        _progress(10)

        result = service.sync_playlist(
            playlist_url,
            auto_add_videos=True,
            fetch_upload_dates=fetch_dates,
            progress_callback=_emit,
            target_playlist_id=playlist
        )

        _progress(100)
        _success(f"同步完成: 新增 {result.new_count}, 已存在 {result.existing_count}")

    except Exception as e:
        _failed(str(e))


# =============================================================================
# refresh-playlist: 刷新 Playlist 视频信息
# =============================================================================

@tools.command('refresh-playlist')
@click.option('--playlist', required=True, help='Playlist ID')
@click.option('--force-refetch', is_flag=True, help='强制重新获取所有字段')
@click.option('--force-retranslate', is_flag=True, help='强制重新翻译')
def tools_refresh_playlist(playlist, force_refetch, force_retranslate):
    """刷新 Playlist 视频信息（补全缺失的封面、时长、日期等）"""
    from ..config import load_config
    from ..database import Database

    config = load_config()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)

    try:
        pl = db.get_playlist(playlist)
        if not pl:
            _failed(f"Playlist 不存在: {playlist}")
            return

        service = _get_playlist_service(config, db)
        _emit(f"刷新 Playlist: {pl.title}")
        _progress(10)

        result = service.refresh_videos(
            playlist,
            force_refetch=force_refetch,
            force_retranslate=force_retranslate,
            callback=_emit
        )

        _progress(100)
        _success(f"刷新完成: 成功 {result['refreshed']}, 失败 {result['failed']}, 跳过 {result['skipped']}")

    except Exception as e:
        _failed(str(e))


# =============================================================================
# retranslate-playlist: 重新翻译 Playlist 视频标题/简介
# =============================================================================

@tools.command('retranslate-playlist')
@click.option('--playlist', required=True, help='Playlist ID')
def tools_retranslate_playlist(playlist):
    """重新翻译 Playlist 中所有视频的标题/简介"""
    from ..config import load_config
    from ..database import Database
    from ..services.playlist_service import PlaylistService

    config = load_config()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)

    try:
        service = PlaylistService(db)
        pl = service.get_playlist(playlist)
        if not pl:
            _failed(f"Playlist 不存在: {playlist}")
            return

        _emit(f"重新翻译 Playlist: {pl.title}")
        _progress(10)

        service.retranslate_videos(playlist)

        _progress(100)
        _success("重新翻译完成")

    except Exception as e:
        _failed(str(e))


# =============================================================================
# upload-sync: 将已上传视频批量添加到B站合集
# =============================================================================

@tools.command('upload-sync')
@click.option('--playlist', required=True, help='Playlist ID')
@click.option('--retry-delay', default=30, type=int, help='失败后重试等待（分钟），0=不重试')
def tools_upload_sync(playlist, retry_delay):
    """将已上传但未入集的视频批量添加到B站合集并排序"""
    from ..config import load_config
    from ..database import Database
    from ..services.playlist_service import PlaylistService

    config = load_config()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)

    try:
        service = PlaylistService(db)
        pl = service.get_playlist(playlist)
        if not pl:
            _failed(f"Playlist 不存在: {playlist}")
            return

        from ..uploaders.bilibili import season_sync
        uploader = _get_bilibili_uploader(config)

        _emit(f"开始 upload sync: {pl.title}")
        _progress(10)

        result = season_sync(db, uploader, playlist)
        _emit(f"结果: {result['success']} 成功, {result['failed']} 失败, 共 {result['total']} 个")

        if result['failed'] > 0 and retry_delay > 0:
            _emit(f"有 {result['failed']} 个失败，{retry_delay} 分钟后重试...")
            _progress(50)
            import time
            time.sleep(retry_delay * 60)

            uploader2 = _get_bilibili_uploader(config)
            _emit("开始重试...")
            result2 = season_sync(db, uploader2, playlist)
            _emit(f"重试结果: {result2['success']} 成功, {result2['failed']} 失败")

            if result2['failed'] > 0:
                _failed(f"仍有 {result2['failed']} 个视频失败")
                return

        _progress(100)
        if result['total'] == 0:
            _success("没有待同步的视频")
        else:
            _success(f"同步完成: {result['success']} 个视频已添加到合集")

    except Exception as e:
        _failed(str(e))


# =============================================================================
# update-info: 批量更新已上传视频的标题和简介
# =============================================================================

@tools.command('update-info')
@click.option('--playlist', required=True, help='Playlist ID')
@click.option('--dry-run', is_flag=True, help='仅预览不执行')
def tools_update_info(playlist, dry_run):
    """批量更新已上传视频的标题和简介（重新渲染模板）"""
    from ..config import load_config
    from ..database import Database
    from ..services.playlist_service import PlaylistService
    from ..uploaders.template import render_upload_metadata

    config = load_config()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)

    try:
        service = PlaylistService(db)
        pl = service.get_playlist(playlist)
        if not pl:
            _failed(f"Playlist 不存在: {playlist}")
            return

        bilibili_config = config.uploader.bilibili
        templates = {}
        if bilibili_config.templates:
            templates = {
                'title': bilibili_config.templates.title,
                'description': bilibili_config.templates.description,
                'custom_vars': bilibili_config.templates.custom_vars,
            }

        pl_upload_config = (pl.metadata or {}).get('upload_config', {})

        videos = service.get_playlist_videos(playlist)
        uploaded = []
        for v in videos:
            meta = v.metadata or {}
            aid = meta.get('bilibili_aid')
            if aid:
                uploaded.append((v, int(aid)))

        if not uploaded:
            _success("没有已上传到B站的视频")
            return

        _emit(f"已上传视频: {len(uploaded)} 个")
        _progress(10)

        # 构建更新列表
        update_list = []
        for v, aid in uploaded:
            pv_info = db.get_playlist_video_info(playlist, v.id)
            upload_order_index = pv_info.get('upload_order_index', 0) if pv_info else 0
            if not upload_order_index:
                upload_order_index = (v.metadata or {}).get('upload_order_index', 0) or v.playlist_index or 0

            playlist_info = {
                'name': pl.title,
                'id': pl.id,
                'index': upload_order_index,
                'uploader_name': pl_upload_config.get('uploader_name', ''),
            }
            rendered = render_upload_metadata(v, templates, playlist_info)

            meta = v.metadata or {}
            translated = meta.get('translated', {})

            update_list.append({
                'video': v,
                'aid': aid,
                'new_title': rendered['title'][:80],
                'new_desc': rendered['description'][:2000],
                'new_tags': translated.get('tags', None),
                'new_tid': translated.get('recommended_tid', None),
            })

        if dry_run:
            for i, item in enumerate(update_list, 1):
                _emit(f"[{i}] av{item['aid']}: {item['new_title'][:60]}")
            _success(f"dry-run 完成，共 {len(update_list)} 个视频待更新")
            return

        # 执行更新
        uploader = _get_bilibili_uploader(config)
        success = 0
        failed = 0
        total = len(update_list)
        for i, item in enumerate(update_list, 1):
            pct = 10 + int(90 * i / total)
            _progress(pct)
            title_short = item['new_title'][:40]
            _emit(f"[{i}/{total}] av{item['aid']}: {title_short}...")

            if uploader.edit_video_info(
                aid=item['aid'],
                title=item['new_title'],
                desc=item['new_desc'],
                tags=item['new_tags'],
                tid=item['new_tid'],
            ):
                success += 1
            else:
                failed += 1
                _emit(f"  ✗ 更新失败")

            import time
            time.sleep(1)

        _progress(100)
        if failed == 0:
            _success(f"完成: {success} 个视频已更新")
        else:
            _failed(f"{success} 成功, {failed} 失败")

    except Exception as e:
        _failed(str(e))


# =============================================================================
# sync-db: 从B站合集同步信息回数据库
# =============================================================================

@tools.command('sync-db')
@click.option('--season', required=True, type=int, help='B站合集ID')
@click.option('--playlist', required=True, help='Playlist ID')
@click.option('--dry-run', is_flag=True, help='仅预览不执行')
def tools_sync_db(season, playlist, dry_run):
    """将B站合集中的视频信息同步回数据库"""
    import re
    from ..config import load_config
    from ..database import Database
    from ..services.playlist_service import PlaylistService

    config = load_config()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)

    try:
        service = PlaylistService(db)
        pl = service.get_playlist(playlist)
        if not pl:
            _failed(f"Playlist 不存在: {playlist}")
            return

        uploader = _get_bilibili_uploader(config)

        _emit(f"获取合集 {season} 的视频列表...")
        season_info = uploader.get_season_episodes(season)
        if not season_info:
            _failed("无法获取合集信息")
            return

        episodes = season_info.get('episodes', [])
        _emit(f"合集中共 {len(episodes)} 个视频")
        _progress(10)

        # 建立映射
        pl_videos = service.get_playlist_videos(playlist)
        vid_to_video = {}
        aid_to_video = {}
        for v in pl_videos:
            vid_to_video[v.id] = v
            if v.source_url:
                yt_match = re.search(r'[?&]v=([a-zA-Z0-9_-]+)', v.source_url)
                if yt_match:
                    vid_to_video[yt_match.group(1)] = v
            meta = v.metadata or {}
            if meta.get('bilibili_aid'):
                aid_to_video[int(meta['bilibili_aid'])] = v

        # 匹配
        matched = []
        unmatched = []
        for i, ep in enumerate(episodes):
            aid = ep.get('aid')

            if aid in aid_to_video:
                matched.append((ep, aid_to_video[aid], 'aid'))
                continue

            detail = uploader.get_video_detail(aid)
            if detail:
                desc = detail.get('desc', '')
                yt_match = re.search(r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)', desc)
                if yt_match:
                    yt_vid = yt_match.group(1)
                    if yt_vid in vid_to_video:
                        matched.append((ep, vid_to_video[yt_vid], 'desc'))
                        continue

            unmatched.append(ep)
            pct = 10 + int(40 * (i + 1) / len(episodes))
            _progress(pct)

        _emit(f"匹配结果: {len(matched)} 匹配, {len(unmatched)} 未匹配")
        _progress(60)

        if dry_run:
            for ep, v, method in matched:
                _emit(f"  ✓ av{ep.get('aid')} → {v.id} ({method})")
            for ep in unmatched:
                _emit(f"  ✗ av{ep.get('aid')}: {ep.get('title', '')[:40]} (未匹配)")
            _success(f"dry-run 完成: {len(matched)} 匹配, {len(unmatched)} 未匹配")
            return

        # 写入 DB
        updated = 0
        for ep, v, method in matched:
            meta = v.metadata or {}
            meta['bilibili_aid'] = ep.get('aid')
            meta['bilibili_bvid'] = ep.get('bvid', '')
            meta['bilibili_target_season_id'] = season
            meta['bilibili_season_added'] = True
            db.update_video(v.id, metadata=meta)
            updated += 1

        _progress(100)
        _success(f"已更新 {updated} 条记录")

    except Exception as e:
        _failed(str(e))


# =============================================================================
# season-sync: B站合集同步（添加视频到合集+排序）
# =============================================================================

@tools.command('season-sync')
@click.option('--playlist', required=True, help='Playlist ID')
def tools_season_sync(playlist):
    """将 Playlist 中已上传的视频同步到B站合集（添加+排序）"""
    from ..config import load_config
    from ..database import Database

    config = load_config()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)

    try:
        from ..uploaders.bilibili import season_sync
        uploader = _get_bilibili_uploader(config)

        _emit(f"开始 season sync (playlist={playlist})...")
        _progress(10)

        result = season_sync(db, uploader, playlist)

        _emit(f"结果: {result['success']} 成功, {result['failed']} 失败, 共 {result['total']} 个")
        _progress(100)

        if result['failed'] == 0:
            _success(f"同步完成: {result['success']} 个视频已添加到合集")
        else:
            _failed(f"部分失败: {result['success']} 成功, {result['failed']} 失败")

    except Exception as e:
        _failed(str(e))
