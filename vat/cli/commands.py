"""
命令行接口实现
"""
import os
import click
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from tabulate import tabulate

from ..config import Config, load_config
from ..database import Database
from ..models import (
    Video, Task, SourceType, TaskStep, TaskStatus,
    STAGE_GROUPS, expand_stage_group, get_required_stages, DEFAULT_STAGE_SEQUENCE
)
from ..pipeline import create_video_from_url, VideoProcessor, schedule_videos
from ..downloaders import YouTubeDownloader
from ..services import PlaylistService
from ..utils.logger import setup_logger


# 全局配置
CONFIG = None
LOGGER = None
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

def get_config(config_path: Optional[str] = None) -> Config:
    """获取配置（延迟加载）
    
    注意：LLM环境变量（OPENAI_API_KEY/OPENAI_BASE_URL）已在config.py的
    LLMConfig.__post_init__中统一设置，无需在此处重复设置
    """
    global CONFIG
    if CONFIG is None:
        CONFIG = load_config(config_path)
        CONFIG.ensure_directories()
    return CONFIG


def get_logger():
    """获取日志器（使用统一的 utils/logger 格式）"""
    global LOGGER
    if LOGGER is None:
        LOGGER = setup_logger("cli")
    return LOGGER


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='配置文件路径')
@click.pass_context
def cli(ctx, config):
    """VAT - 视频自动化翻译流水线系统"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config


@cli.command()
@click.option('--output', '-o', default='config/config.yaml', help='输出配置文件路径')
def init(output):
    """初始化配置文件"""
    try:
        # 直接从默认配置读取并保存
        config = load_config()
        config.to_yaml(output)
        click.echo(f"✓ 已创建默认配置文件: {output}")
        click.echo(f"请编辑配置文件以设置API密钥等参数")
    except Exception as e:
        click.echo(f"✗ 创建配置文件失败: {e}", err=True)


@cli.command()
@click.option('--url', '-u', multiple=True, help='YouTube视频URL')
@click.option('--playlist', '-p', help='YouTube播放列表URL')
@click.option('--file', '-f', type=click.Path(exists=True), help='URL列表文件')
@click.pass_context
def download(ctx, url, playlist, file):
    """下载YouTube视频"""
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    # 收集URLs
    urls = list(url)
    
    # 从播放列表获取
    if playlist:
        logger.info(f"获取播放列表: {playlist}")
        downloader = YouTubeDownloader(
            proxy=config.get_stage_proxy("downloader"),
            video_format=config.downloader.youtube.format,
            cookies_file=config.downloader.youtube.cookies_file,
            remote_components=config.downloader.youtube.remote_components,
        )
        playlist_urls = downloader.get_playlist_urls(playlist)
        urls.extend(playlist_urls)
        logger.info(f"播放列表包含 {len(playlist_urls)} 个视频")
    
    # 从文件获取
    if file:
        with open(file, 'r', encoding='utf-8') as f:
            file_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            urls.extend(file_urls)
            logger.info(f"从文件读取 {len(file_urls)} 个URL")
    
    if not urls:
        click.echo("错误: 请提供至少一个URL", err=True)
        return
    
    logger.info(f"共 {len(urls)} 个视频待下载")
    
    # 创建视频记录
    video_ids = []
    for url_str in urls:
        try:
            video_id = create_video_from_url(url_str, db, SourceType.YOUTUBE)
            video_ids.append(video_id)
            logger.info(f"已添加: {url_str} (ID: {video_id})")
        except Exception as e:
            logger.error(f"添加失败: {url_str} - {e}")
    
    # 执行下载
    if video_ids:
        schedule_videos(config, video_ids, steps=['download'], use_multi_gpu=False)


@cli.command()
@click.option('--video-id', '-v', help='视频ID')
@click.option('--all', 'process_all', is_flag=True, help='处理所有已下载但未转录的视频')
@click.option('--force', '-f', is_flag=True, help='强制重新处理（即使已完成）')
@click.pass_context
def asr(ctx, video_id, process_all, force):
    """语音识别（转录字幕）"""
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    # 确定要处理的视频
    if video_id:
        video_ids = [video_id]
    elif process_all:
        # 查找已完成下载的视频
        video_ids = []
        for vid in [v.id for v in db.list_videos()]:
            if db.is_step_completed(vid, TaskStep.DOWNLOAD):
                # 如果使用 --force，包含所有已下载的；否则只包含未转录的
                if force or not db.is_step_completed(vid, TaskStep.SPLIT):
                    video_ids.append(vid)
        
        if force:
            logger.info(f"找到 {len(video_ids)} 个视频（强制重新处理）")
        else:
            logger.info(f"找到 {len(video_ids)} 个待转录视频")
    else:
        click.echo("错误: 请指定 --video-id 或使用 --all", err=True)
        return
    
    if not video_ids:
        click.echo("没有待处理的视频")
        return
    
    # 执行转录
    schedule_videos(config, video_ids, steps=['asr'], use_multi_gpu=True, force=force)


@cli.command()
@click.option('--video-id', '-v', help='视频ID')
@click.option('--all', 'process_all', is_flag=True, help='翻译所有已转录但未翻译的视频')
@click.option('--backend', '-b', type=click.Choice(['local', 'online', 'hybrid']), help='翻译后端')
@click.option('--force', '-f', is_flag=True, help='强制重新处理（即使已完成）')
@click.pass_context
def translate(ctx, video_id, process_all, backend, force):
    """翻译字幕"""
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    # 覆盖后端设置
    if backend:
        config.translator.default_backend = backend
    
    # 确定要处理的视频
    if video_id:
        video_ids = [video_id]
    elif process_all:
        video_ids = []
        for vid in [v.id for v in db.list_videos()]:
            if db.is_step_completed(vid, TaskStep.SPLIT):
                # 如果使用 --force，包含所有已转录的（split完成）；否则只包含未翻译的
                if force or not db.is_step_completed(vid, TaskStep.TRANSLATE):
                    video_ids.append(vid)
        
        if force:
            logger.info(f"找到 {len(video_ids)} 个视频（强制重新翻译）")
        else:
            logger.info(f"找到 {len(video_ids)} 个待翻译视频")
    else:
        click.echo("错误: 请指定 --video-id 或使用 --all", err=True)
        return
    
    if not video_ids:
        click.echo("没有待处理的视频")
        return
    
    # 执行翻译
    schedule_videos(config, video_ids, steps=['translate'], use_multi_gpu=True, force=force)


@cli.command()
@click.option('--video-id', '-v', help='视频ID')
@click.option('--all', 'process_all', is_flag=True, help='嵌入所有已翻译但未嵌入的视频')
@click.option('--force', '-f', is_flag=True, help='强制重新处理（即使已完成）')
@click.pass_context
def embed(ctx, video_id, process_all, force):
    """嵌入字幕到视频"""
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    # 确定要处理的视频
    if video_id:
        video_ids = [video_id]
    elif process_all:
        video_ids = []
        for vid in [v.id for v in db.list_videos()]:
            if db.is_step_completed(vid, TaskStep.TRANSLATE):
                # 如果使用 --force，包含所有已翻译的；否则只包含未嵌入的
                if force or not db.is_step_completed(vid, TaskStep.EMBED):
                    video_ids.append(vid)
        
        if force:
            logger.info(f"找到 {len(video_ids)} 个视频（强制重新嵌入）")
        else:
            logger.info(f"找到 {len(video_ids)} 个待嵌入视频")
    else:
        click.echo("错误: 请指定 --video-id 或使用 --all", err=True)
        return
    
    if not video_ids:
        click.echo("没有待处理的视频")
        return
    
    # 执行嵌入
    schedule_videos(config, video_ids, steps=['embed'], use_multi_gpu=False, force=force)


@cli.command()
@click.option('--url', '-u', multiple=True, help='YouTube视频URL')
@click.option('--playlist', '-p', help='YouTube播放列表URL')
@click.option('--file', '-f', type=click.Path(exists=True), help='URL列表文件')
@click.option('--gpus', help='使用的GPU列表（逗号分隔，如: 0,1,2）')
@click.option('--force', is_flag=True, help='强制重新处理（即使已完成）')
@click.pass_context
def pipeline(ctx, url, playlist, file, gpus, force):
    """完整流水线处理（下载→转录→翻译→嵌入）"""
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    # 设置GPU
    if gpus:
        config.concurrency.gpu_devices = [int(g.strip()) for g in gpus.split(',')]
    
    # 收集URLs
    urls = list(url)
    
    if playlist:
        logger.info(f"获取播放列表: {playlist}")
        downloader = YouTubeDownloader(
            proxy=config.get_stage_proxy("downloader"),
            video_format=config.downloader.youtube.format,
            cookies_file=config.downloader.youtube.cookies_file,
            remote_components=config.downloader.youtube.remote_components,
        )
        playlist_urls = downloader.get_playlist_urls(playlist)
        urls.extend(playlist_urls)
        logger.info(f"播放列表包含 {len(playlist_urls)} 个视频")
    
    if file:
        with open(file, 'r', encoding='utf-8') as f:
            file_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            urls.extend(file_urls)
            logger.info(f"从文件读取 {len(file_urls)} 个URL")
    
    if not urls:
        click.echo("错误: 请提供至少一个URL", err=True)
        return
    
    logger.info(f"共 {len(urls)} 个视频待处理")
    
    # 创建视频记录
    video_ids = []
    for url_str in urls:
        try:
            video_id = create_video_from_url(url_str, db, SourceType.YOUTUBE)
            video_ids.append(video_id)
            logger.info(f"已添加: {url_str} (ID: {video_id})")
        except Exception as e:
            logger.error(f"添加失败: {url_str} - {e}")
    
    # 执行完整流水线
    if video_ids:
        schedule_videos(
            config,
            video_ids,
            steps=['download', 'asr', 'translate', 'embed'],
            use_multi_gpu=len(config.concurrency.gpu_devices) > 1,
            force=force
            )



@cli.command()
@click.option('--video-id', '-v', help='查看特定视频的状态')
@click.option('--failed', 'filter_failed', is_flag=True, help='仅显示失败的任务')
@click.option('--pending', 'filter_pending', is_flag=True, help='仅显示待处理的任务')
@click.pass_context
def status(ctx, video_id, filter_failed, filter_pending):
    """查看处理状态"""
    config = get_config(ctx.obj.get('config_path'))
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    if video_id:
        # 显示特定视频的详细状态
        video = db.get_video(video_id)
        if not video:
            click.echo(f"错误: 视频不存在: {video_id}", err=True)
            return
        
        click.echo(f"\n视频信息:")
        click.echo(f"  ID: {video.id}")
        click.echo(f"  标题: {video.title or '(未知)'}")
        click.echo(f"  来源: {video.source_type.value}")
        click.echo(f"  URL: {video.source_url}")
        click.echo(f"  输出目录: {video.output_dir}")
        
        # 任务状态
        tasks = db.get_tasks(video_id)
        if tasks:
            click.echo(f"\n任务状态:")
            table_data = []
            for task in tasks:
                # 显示sub_phase（如果有）
                sub_phase_str = task.sub_phase.value if task.sub_phase else '-'
                table_data.append([
                    task.step.value,
                    task.status.value,
                    sub_phase_str,
                    task.gpu_id if task.gpu_id is not None else '-',
                    task.started_at.strftime('%Y-%m-%d %H:%M:%S') if task.started_at else '-',
                    task.completed_at.strftime('%Y-%m-%d %H:%M:%S') if task.completed_at else '-',
                    task.error_message[:40] if task.error_message else '-'
                ])
            
            headers = ['步骤', '状态', '子阶段', 'GPU', '开始时间', '完成时间', '错误']
            click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))
    else:
        # 显示所有视频的概览
        videos = db.list_videos()
        
        if not videos:
            click.echo("没有视频记录")
            return
        
        click.echo(f"\n共 {len(videos)} 个视频\n")
        
        table_data = []
        for video in videos:
            tasks = db.get_tasks(video.id)
            task_status = {}
            for task in tasks:
                if task.step not in task_status or task.id > task_status[task.step].id:
                    task_status[task.step] = task
            
            # 统计状态
            completed_count = sum(1 for t in task_status.values() if t.status == TaskStatus.COMPLETED)
            failed_count = sum(1 for t in task_status.values() if t.status == TaskStatus.FAILED)
            running_count = sum(1 for t in task_status.values() if t.status == TaskStatus.RUNNING)
            
            # 应用过滤
            if filter_failed and failed_count == 0:
                continue
            if filter_pending and completed_count >= len(DEFAULT_STAGE_SEQUENCE):
                continue
            
            table_data.append([
                video.id[:12],
                video.title[:30] if video.title else '-',
                video.source_type.value,
                f"{completed_count}/{len(DEFAULT_STAGE_SEQUENCE)}",
                "✗" if failed_count > 0 else ("⟳" if running_count > 0 else "✓"),
            ])
        
        if table_data:
            headers = ['ID', '标题', '来源', '进度', '状态']
            click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))
        else:
            click.echo("没有符合条件的视频")


@cli.command()
@click.option('--video-id', '-v', multiple=True, help='视频ID（可多次指定）')
@click.option('--all', 'clean_all', is_flag=True, help='清理所有视频')
@click.option('--records', is_flag=True, help='同时删除数据库记录')
@click.option('--include-source', is_flag=True, help='同时删除原始下载文件（需配合 --records）')
@click.option('--yes', '-y', is_flag=True, help='跳过确认')
@click.pass_context
def clean(ctx, video_id, clean_all, records, include_source, yes):
    """清理视频数据
    
    默认只删除处理产物（保留原始下载文件和数据库记录）。
    
    示例:
    
      # 清理处理产物（保留下载文件和记录）
      vat clean -v VIDEO_ID
      
      # 删除记录和处理产物（保留原始下载文件）
      vat clean -v VIDEO_ID --records
      
      # 完全删除（记录+所有文件）
      vat clean -v VIDEO_ID --records --include-source
      
      # 清理所有视频的处理产物
      vat clean --all
    """
    import shutil
    from vat.utils.file_ops import delete_processed_files
    
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    if include_source and not records:
        click.echo("错误: --include-source 需配合 --records 使用", err=True)
        return
    
    # 收集视频
    if clean_all:
        videos = db.list_videos()
    elif video_id:
        videos = []
        for vid in video_id:
            v = db.get_video(vid)
            if v:
                videos.append(v)
            else:
                click.echo(f"警告: 视频不存在: {vid}", err=True)
    else:
        click.echo("错误: 请指定 --video-id 或使用 --all", err=True)
        return
    
    if not videos:
        click.echo("没有待清理的视频")
        return
    
    # 描述操作
    if records and include_source:
        action = "删除记录+所有文件（含原始下载）"
    elif records:
        action = "删除记录+处理产物（保留原始下载文件）"
    else:
        action = "清理处理产物（保留原始下载文件和记录）"
    
    click.echo(f"操作: {action}")
    click.echo(f"视频数: {len(videos)}")
    if not yes and not click.confirm("确认?"):
        click.echo("已取消")
        return
    
    total_deleted = 0
    for v in videos:
        if v.output_dir:
            output_dir = Path(v.output_dir)
            if output_dir.exists():
                if records and include_source:
                    shutil.rmtree(output_dir)
                    total_deleted += 1
                else:
                    deleted = delete_processed_files(output_dir)
                    if deleted:
                        total_deleted += len(deleted)
                        if not records:
                            click.echo(f"  {v.id[:12]}: 删除 {len(deleted)} 个文件")
        
        if records:
            db.delete_video(v.id)
            title = v.title[:30] if v.title else v.id
            click.echo(f"  ✓ 已删除: {title}")
    
    if records:
        click.echo(f"\n删除完成，共处理 {len(videos)} 个视频")
    else:
        click.echo(f"\n清理完成，共删除 {total_deleted} 个处理产物")


# ==================== 新增命令：process (支持细粒度阶段) ====================

def parse_stages(stages_str: str) -> List[TaskStep]:
    """
    解析阶段参数
    
    支持格式：
    - 单个阶段: "whisper", "translate"
    - 阶段组: "asr" (展开为 whisper,split), "translate" (展开为 optimize,translate)
    - 多个阶段: "whisper,split,optimize"
    - 全部: "all"
    """
    if not stages_str or stages_str.lower() == 'all':
        return list(DEFAULT_STAGE_SEQUENCE)
    
    result = []
    for part in stages_str.split(','):
        part = part.strip().lower()
        if not part:
            continue
        
        # 尝试作为阶段组展开
        expanded = expand_stage_group(part)
        result.extend(expanded)
    
    # 去重并保持顺序
    seen = set()
    unique = []
    for step in result:
        if step not in seen:
            seen.add(step)
            unique.append(step)
    
    return unique



@cli.command()
@click.option('--video-id', '-v', multiple=True, help='视频ID（可多次指定）')
@click.option('--all', 'process_all', is_flag=True, help='处理所有待处理的视频')
@click.option('--playlist', '-p', help='处理指定 Playlist 中的视频')
@click.option('--stages', '-s', default='all', 
              help='要执行的阶段（逗号分隔）: download,whisper,split,optimize,translate,embed,upload 或阶段组 asr 或 all')
@click.option('--gpu', '-g', default='auto',
              help='GPU 设备: auto（自动选择）, cpu, cuda:0, cuda:1 等')
@click.option('--force', '-f', is_flag=True, help='强制重新处理（即使已完成）')
@click.option('--dry-run', is_flag=True, help='仅显示将要执行的操作，不实际执行')
@click.option('--concurrency', '-c', default=1, type=int, help='并发处理的视频数量（默认1，即串行）')
@click.option('--delay', '-d', default=None, type=float, help='视频间处理延迟（秒），防止 YouTube 限流。默认从配置读取')
@click.option('--upload-cron', default=None, help='定时上传 cron 表达式（仅当 stages 为 upload 时可用），如 "0 12,18 * * *"')
@click.option('--upload-batch-size', default=1, type=int, help='每次 cron 触发时上传的视频数量（默认1，仅与 --upload-cron 搭配使用）')
@click.option('--upload-mode', default='cron', type=click.Choice(['cron', 'dtime']),
              help='定时上传模式: cron=后台进程等待定时上传, dtime=立即全部上传但通过B站定时发布（需 >2h）')
@click.option('--fail-fast', is_flag=True, help='遇到视频处理失败时立即停止后续处理（多线程时不中断已运行的任务，但不再启动新任务）')
@click.pass_context
def process(ctx, video_id, process_all, playlist, stages, gpu, force, dry_run, concurrency, delay, upload_cron, upload_batch_size, upload_mode, fail_fast):
    """
    处理视频（支持细粒度阶段控制）
    
    示例:
    
      # 处理单个视频的所有阶段
      vat process -v VIDEO_ID
      
      # 只执行 ASR 阶段（whisper + split）
      vat process -v VIDEO_ID -s asr
      
      # 只执行翻译阶段（optimize + translate）
      vat process -v VIDEO_ID -s translate
      
      # 执行特定细粒度阶段
      vat process -v VIDEO_ID -s whisper,split
      
      # 使用指定 GPU
      vat process -v VIDEO_ID -g cuda:1
      
      # 处理 Playlist 中的所有视频
      vat process -p PLAYLIST_ID -s all
    """
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    # 解析阶段
    try:
        target_steps = parse_stages(stages)
    except Exception as e:
        click.echo(f"错误: 无效的阶段参数: {e}", err=True)
        return
    
    if not target_steps:
        click.echo("错误: 未指定任何阶段", err=True)
        return
    
    # 收集视频 ID
    video_ids = list(video_id)
    
    if playlist:
        # 从 Playlist 获取信息
        playlist_service = PlaylistService(db)
        pl = playlist_service.get_playlist(playlist)
        if not pl:
            click.echo(f"错误: Playlist 不存在: {playlist}", err=True)
            return
        
        # 应用 playlist 级别的 custom prompt 覆写（始终生效）
        if pl.metadata:
            config.apply_playlist_prompts(pl.metadata)
        
        # 仅在未显式指定 -v 时从 playlist 收集视频
        # 当同时指定 -v 和 -p 时，-p 仅作为 prompt context
        if not video_ids:
            pl_videos = playlist_service.get_playlist_videos(playlist)
            video_ids.extend([v.id for v in pl_videos])
            logger.info(f"Playlist '{pl.title}' 包含 {len(pl_videos)} 个视频")
    
    if process_all:
        # 获取所有待处理的视频
        all_videos = db.list_videos()
        for v in all_videos:
            if v.id not in video_ids:
                pending = db.get_pending_steps(v.id)
                # 检查是否有任何目标阶段待处理
                if any(step in pending for step in target_steps) or force:
                    video_ids.append(v.id)
        logger.info(f"找到 {len(video_ids)} 个待处理视频")
    
    if not video_ids:
        click.echo("错误: 请指定 --video-id, --playlist 或使用 --all", err=True)
        return
    
    # 去重
    video_ids = list(dict.fromkeys(video_ids))
    
    # ========== upload-cron 校验与分流 ==========
    if upload_cron:
        # 校验1: stages 必须仅包含 upload
        upload_only = all(s == TaskStep.UPLOAD for s in target_steps) and len(target_steps) == 1
        if not upload_only:
            click.echo("错误: --upload-cron 仅可用于 upload 阶段（-s upload）", err=True)
            return
        
        # 校验2: cron 表达式合法性
        try:
            from croniter import croniter
            if not croniter.is_valid(upload_cron):
                click.echo(f"错误: 无效的 cron 表达式: {upload_cron}", err=True)
                return
        except ImportError:
            click.echo("错误: croniter 未安装，请运行: pip install croniter", err=True)
            return
        
        # 校验3: 所有视频的 embed 阶段已完成
        not_ready = []
        for vid in video_ids:
            if not db.is_step_completed(vid, TaskStep.EMBED):
                v = db.get_video(vid)
                name = v.title[:30] if v and v.title else vid
                not_ready.append(name)
        if not_ready:
            click.echo(f"错误: 以下 {len(not_ready)} 个视频尚未完成 embed 阶段，无法创建定时上传任务:", err=True)
            for name in not_ready[:5]:
                click.echo(f"  - {name}", err=True)
            if len(not_ready) > 5:
                click.echo(f"  ... 还有 {len(not_ready) - 5} 个", err=True)
            return
        
        # 进入定时上传流程
        if upload_mode == 'dtime':
            _run_dtime_uploads(config, db, logger, video_ids, upload_cron, force, dry_run, playlist_id=playlist, batch_size=upload_batch_size)
        else:
            _run_scheduled_uploads(config, db, logger, video_ids, upload_cron, force, dry_run, playlist_id=playlist, batch_size=upload_batch_size)
        
        # 定时上传完成后，自动执行 season sync（与普通上传路径一致）
        if playlist and not dry_run:
            _auto_season_sync(config, db, logger, playlist)
        return
    
    # 显示执行计划
    plan_lines = [
        f"执行计划: 视频={len(video_ids)}, "
        f"阶段={','.join(s.value for s in target_steps)}, "
        f"GPU={gpu}, force={'是' if force else '否'}, 并发={concurrency}"
        + (f", fail-fast=是" if fail_fast else "")
    ]
    logger.info(plan_lines[0])
    
    if dry_run:
        logger.info("[DRY-RUN] 以下视频将被处理:")
        for vid in video_ids[:10]:
            video = db.get_video(vid)
            title = video.title[:40] if video and video.title else vid
            logger.info(f"  - {title}")
        if len(video_ids) > 10:
            logger.info(f"  ... 还有 {len(video_ids) - 10} 个视频")
        
        cli_cmd = _generate_process_cli(video_ids, stages, gpu, force)
        logger.info(f"等价 CLI 命令: {cli_cmd}")
        return
    
    # 设置 GPU
    config.gpu.device = gpu
    
    # 解析 GPU 设备为 gpu_id
    gpu_id = None
    if gpu and gpu != 'auto' and gpu != 'cpu':
        if gpu.startswith('cuda:'):
            try:
                gpu_id = int(gpu.split(':')[1])
            except (IndexError, ValueError):
                pass
    
    step_names = [s.value for s in target_steps]
    total = len(video_ids)
    
    # 确定视频间延迟：CLI 参数优先，否则从配置读取
    download_delay = delay if delay is not None else config.downloader.youtube.download_delay
    
    def process_one_video(args):
        """处理单个视频（可在线程池中并发调用）"""
        idx, vid = args
        video = db.get_video(vid)
        if not video:
            logger.warning(f"视频不存在: {vid}")
            return vid, False, "视频不存在"
        
        title = video.title[:30] if video.title else vid
        logger.info(f"[{idx + 1}/{total}] 开始处理: {title}")
        
        try:
            processor = VideoProcessor(
                video_id=vid,
                config=config,
                gpu_id=gpu_id,
                force=force,
                video_index=idx,
                total_videos=total,
                playlist_id=playlist
            )
            success = processor.process(steps=step_names)
            if success:
                logger.info(f"[{idx + 1}/{total}] 完成: {title}")
                return vid, True, None
            else:
                logger.warning(f"[{idx + 1}/{total}] 失败: {title}")
                return vid, False, "处理返回失败"
        except Exception as e:
            import traceback
            logger.error(f"[{idx + 1}/{total}] 失败: {title} - {e}\n{traceback.format_exc()}")
            return vid, False, str(e)
    
    def _run_batch(video_list):
        """执行一批视频处理，返回 (failed_vids, stopped_early)
        
        stopped_early: fail-fast 模式下因失败而提前终止时为 True
        """
        failed_vids = []
        stopped_early = False
        if concurrency <= 1:
            for i, (idx, vid) in enumerate(video_list):
                if i > 0 and download_delay > 0:
                    logger.info(f"等待 {download_delay:.0f} 秒后处理下一个视频...")
                    import time
                    time.sleep(download_delay)
                _, success, _ = process_one_video((idx, vid))
                if not success:
                    failed_vids.append(vid)
                    if fail_fast:
                        remaining = len(video_list) - i - 1
                        if remaining > 0:
                            logger.warning(f"fail-fast: 视频 {vid} 处理失败，跳过剩余 {remaining} 个视频")
                        stopped_early = True
                        break
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            _fail_fast_triggered = False
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                # 逐个提交任务，遇到失败后不再提交新任务
                pending_futures = {}
                video_iter = iter(video_list)
                
                # 先填满线程池
                for _ in range(min(concurrency, len(video_list))):
                    try:
                        idx, vid = next(video_iter)
                        future = executor.submit(process_one_video, (idx, vid))
                        pending_futures[future] = vid
                    except StopIteration:
                        break
                
                while pending_futures:
                    # 等待任意一个完成
                    done_futures = []
                    for future in as_completed(pending_futures):
                        done_futures.append(future)
                        break  # 每次只处理一个完成的
                    
                    for future in done_futures:
                        vid = pending_futures.pop(future)
                        try:
                            _, success, _ = future.result()
                            if not success:
                                failed_vids.append(vid)
                                if fail_fast:
                                    _fail_fast_triggered = True
                        except Exception as e:
                            logger.error(f"并发处理异常: {vid} - {e}")
                            failed_vids.append(vid)
                            if fail_fast:
                                _fail_fast_triggered = True
                    
                    # 如果没有触发 fail-fast，继续提交新任务
                    if not _fail_fast_triggered:
                        try:
                            idx, vid = next(video_iter)
                            future = executor.submit(process_one_video, (idx, vid))
                            pending_futures[future] = vid
                        except StopIteration:
                            pass
                    elif not pending_futures:
                        # 所有已提交的任务都完成了，退出
                        break
                    # 如果 fail-fast 已触发但还有运行中的任务，继续等待它们完成
                
                if _fail_fast_triggered:
                    stopped_early = True
                    logger.warning(f"fail-fast: 处理失败，不再启动新任务（已等待运行中的任务完成）")
        
        return failed_vids, stopped_early
    
    # 执行处理（失败的视频放到队尾重试，最多重试2轮）
    max_retry_rounds = 2
    logger.info(f"开始处理 {total} 个视频（并发: {concurrency}）")
    
    failed_vids, stopped_early = _run_batch(list(enumerate(video_ids)))
    
    if not (fail_fast and stopped_early):
        for retry_round in range(1, max_retry_rounds + 1):
            if not failed_vids:
                break
            logger.info(f"第 {retry_round} 轮重试: {len(failed_vids)} 个失败视频")
            retry_list = [(video_ids.index(vid), vid) for vid in failed_vids]
            failed_vids, stopped_early = _run_batch(retry_list)
            if fail_fast and stopped_early:
                break
    elif failed_vids:
        logger.info(f"fail-fast 模式：跳过重试")
    
    if failed_vids:
        logger.warning(f"处理完成，{len(failed_vids)} 个视频最终失败: {', '.join(failed_vids[:5])}")
    else:
        logger.info("处理完成，全部成功")
    
    # ========== 批量上传后自动 upload sync ==========
    # 条件：stages 包含 upload 且有 playlist 上下文
    has_upload = any(s == TaskStep.UPLOAD for s in target_steps)
    if has_upload and playlist:
        _auto_season_sync(config, db, logger, playlist)


def _auto_season_sync(config, db, logger, playlist_id: str, retry_delay_minutes: int = 30):
    """
    批量上传后自动将视频添加到合集并排序。
    
    流程：
    1. 立即尝试一次 season-sync
    2. 如果有失败的（视频可能尚未被B站索引），等待后自动重试一次
    
    Args:
        config: 配置对象
        db: 数据库实例
        logger: 日志器
        playlist_id: Playlist ID
        retry_delay_minutes: 重试等待时间（分钟），默认30分钟
    """
    try:
        from ..uploaders.bilibili import BilibiliUploader, season_sync, BILIUP_AVAILABLE
        
        if not BILIUP_AVAILABLE:
            return
        
        bilibili_config = config.uploader.bilibili
        project_root = Path(__file__).parent.parent.parent
        cookies_file = project_root / bilibili_config.cookies_file
        
        uploader = BilibiliUploader(
            cookies_file=str(cookies_file),
            line=bilibili_config.line,
            threads=bilibili_config.threads
        )
        
        # 第一次尝试
        logger.info(f"=== Season Sync: 第1次尝试 (playlist={playlist_id}) ===")
        result = season_sync(db, uploader, playlist_id)
        
        if result['total'] == 0:
            logger.info("没有待同步的视频，跳过 season-sync")
            return
        
        if result['failed'] == 0:
            logger.info(f"Season sync 完成: 全部 {result['success']} 个视频已添加到合集")
            return
        
        # 有失败的，等待后重试
        logger.info(
            f"Season sync 第1次: {result['success']} 成功, {result['failed']} 失败"
            f"（可能B站尚未索引）。{retry_delay_minutes} 分钟后自动重试..."
        )
        
        import time
        time.sleep(retry_delay_minutes * 60)
        
        # 第二次尝试（重新创建 uploader 防止 session 过期）
        uploader2 = BilibiliUploader(
            cookies_file=str(cookies_file),
            line=bilibili_config.line,
            threads=bilibili_config.threads
        )
        
        logger.info(f"=== Season Sync: 第2次尝试 (playlist={playlist_id}) ===")
        result2 = season_sync(db, uploader2, playlist_id)
        
        if result2['failed'] == 0:
            logger.info(f"Season sync 重试完成: 全部成功")
        else:
            logger.warning(
                f"Season sync 重试后仍有 {result2['failed']} 个视频失败，"
                f"请稍后手动运行: vat upload sync -p {playlist_id}"
            )
    except Exception as e:
        logger.error(f"Season sync 异常: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _run_scheduled_uploads(config, db, logger, video_ids, cron_expr, force, dry_run, playlist_id=None, batch_size=1):
    """
    定时上传：按 cron 表达式批量上传视频
    
    每次 cron 触发时间到达后上传队列中的 batch_size 个视频。
    已完成上传的视频会被跳过（支持断点续传）。
    
    Args:
        config: 配置对象
        db: 数据库实例
        logger: 日志器
        video_ids: 视频ID有序列表（决定上传顺序）
        cron_expr: cron 表达式
        force: 是否强制重新上传
        dry_run: 仅预览
        playlist_id: 发起任务的 Playlist ID（上传时用于确定正确的 playlist 上下文）
        batch_size: 每次 cron 触发时上传的视频数量（默认1）
    """
    import time
    from datetime import datetime
    from croniter import croniter
    import math
    
    batch_size = max(1, batch_size)
    total = len(video_ids)
    
    # 构建上传队列：跳过已完成上传的视频（除非 force）
    queue = []
    for vid in video_ids:
        if not force and db.is_step_completed(vid, TaskStep.UPLOAD):
            video = db.get_video(vid)
            title = video.title[:30] if video and video.title else vid
            logger.info(f"跳过已上传: {title}")
            continue
        queue.append(vid)
    
    if not queue:
        logger.info("所有视频已上传完成，无需定时上传")
        return
    
    # 按 batch_size 分组
    batches = []
    for i in range(0, len(queue), batch_size):
        batches.append(queue[i:i + batch_size])
    
    logger.info(f"定时上传任务: {len(queue)}/{total} 个视频待上传")
    logger.info(f"Cron 表达式: {cron_expr}, 每次上传 {batch_size} 个, 共 {len(batches)} 批次")
    
    # 预览模式：显示上传计划
    cron = croniter(cron_expr, datetime.now())
    if dry_run:
        logger.info("[DRY-RUN] 上传计划:")
        vid_idx = 0
        for batch_idx, batch in enumerate(batches):
            next_time = cron.get_next(datetime)
            logger.info(f"  批次 {batch_idx + 1} @ {next_time.strftime('%Y-%m-%d %H:%M')} ({len(batch)} 个):")
            for vid in batch:
                vid_idx += 1
                video = db.get_video(vid)
                title = video.title[:40] if video and video.title else vid
                logger.info(f"    {vid_idx}. {title}")
        return
    
    # 按批次上传
    uploaded = 0
    failed = 0
    cron = croniter(cron_expr, datetime.now())
    
    for batch_idx, batch in enumerate(batches):
        next_time = cron.get_next(datetime)
        
        # 等待到触发时间
        now = datetime.now()
        wait_seconds = (next_time - now).total_seconds()
        
        if wait_seconds > 0:
            batch_titles = []
            for vid in batch[:3]:
                video = db.get_video(vid)
                batch_titles.append(video.title[:20] if video and video.title else vid)
            preview = ', '.join(batch_titles)
            if len(batch) > 3:
                preview += f' 等{len(batch)}个'
            
            logger.info(
                f"[UPLOAD-SCHEDULE] 等待批次 {batch_idx + 1}/{len(batches)} "
                f"@ {next_time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(还需等待 {_format_duration(wait_seconds)}): {preview}"
            )
            # 分段 sleep，每 60 秒输出一次心跳日志
            while True:
                remaining = (next_time - datetime.now()).total_seconds()
                if remaining <= 0:
                    break
                sleep_chunk = min(remaining, 60.0)
                time.sleep(sleep_chunk)
        
        # 依次上传本批次中的视频
        logger.info(f"[UPLOAD-SCHEDULE] 开始批次 {batch_idx + 1}/{len(batches)} ({len(batch)} 个视频)")
        for vid in batch:
            video = db.get_video(vid)
            title = video.title[:30] if video and video.title else vid
            logger.info(f"[UPLOAD-SCHEDULE] 上传 ({uploaded + failed + 1}/{len(queue)}): {title}")
            try:
                processor = VideoProcessor(
                    video_id=vid,
                    config=config,
                    gpu_id=None,
                    force=force,
                    video_index=uploaded + failed,
                    total_videos=len(queue),
                    playlist_id=playlist_id
                )
                success = processor.process(steps=['upload'])
                if success:
                    uploaded += 1
                    logger.info(f"[UPLOAD-SCHEDULE] 上传成功 ({uploaded}/{len(queue)}): {title}")
                else:
                    failed += 1
                    logger.warning(f"[UPLOAD-SCHEDULE] 上传失败 ({title})，继续下一个")
            except Exception as e:
                failed += 1
                logger.error(f"[UPLOAD-SCHEDULE] 上传异常 ({title}): {e}")
    
    logger.info(
        f"[UPLOAD-SCHEDULE] 定时上传完成: "
        f"成功 {uploaded}, 失败 {failed}, 总计 {len(queue)}"
    )


def _run_dtime_uploads(config, db, logger, video_ids, cron_expr, force, dry_run, playlist_id=None, batch_size=1):
    """
    B站定时发布模式：计算每个视频的发布时间，一次性全部上传（通过 B站 dtime 参数定时发布）
    
    与 cron 模式不同，此模式不需要后台进程等待。所有视频立即上传到 B站，
    但通过 dtime 参数指定不同的定时发布时间，B站会在指定时间自动发布。
    
    约束：
    - B站 dtime 必须距离当前时间 > 2小时
    - B站 dtime 上限约 15 天（超过可能被 API 拒绝）
    
    Args:
        config: 配置对象
        db: 数据库实例
        logger: 日志器
        video_ids: 视频ID有序列表
        cron_expr: cron 表达式（用于计算发布时间）
        force: 是否强制重新上传
        dry_run: 仅预览
        playlist_id: 发起任务的 Playlist ID
        batch_size: 每次 cron 触发时发布的视频数量
    """
    import time as _time
    from datetime import datetime
    from croniter import croniter
    
    batch_size = max(1, batch_size)
    total = len(video_ids)
    
    # 构建上传队列
    queue = []
    for vid in video_ids:
        if not force and db.is_step_completed(vid, TaskStep.UPLOAD):
            video = db.get_video(vid)
            title = video.title[:30] if video and video.title else vid
            logger.info(f"跳过已上传: {title}")
            continue
        queue.append(vid)
    
    if not queue:
        logger.info("所有视频已上传完成，无需定时上传")
        return
    
    # 按 batch_size 分组
    batches = []
    for i in range(0, len(queue), batch_size):
        batches.append(queue[i:i + batch_size])
    
    # 计算每批次的发布时间（dtime）
    now = datetime.now()
    cron = croniter(cron_expr, now)
    batch_dtimes = []  # [(batch, dtime_timestamp), ...]
    
    DTIME_MIN_OFFSET = 7200 + 60  # 至少 2小时1分钟（留余量）
    DTIME_MAX_OFFSET = 15 * 86400  # 最多 15 天
    
    for batch in batches:
        next_time = cron.get_next(datetime)
        dtime_ts = int(next_time.timestamp())
        batch_dtimes.append((batch, dtime_ts, next_time))
    
    # 校验所有 dtime
    now_ts = int(_time.time())
    too_early = []
    too_late = []
    for batch, dtime_ts, next_time in batch_dtimes:
        offset = dtime_ts - now_ts
        if offset < DTIME_MIN_OFFSET:
            too_early.append(next_time)
        if offset > DTIME_MAX_OFFSET:
            too_late.append(next_time)
    
    if too_early:
        logger.error(
            f"以下发布时间距现在不足 2 小时（B站要求 dtime > 2小时）:\n"
            + "\n".join(f"  {t.strftime('%Y-%m-%d %H:%M')}" for t in too_early)
        )
        logger.error("请调整 cron 表达式，确保首次触发时间距现在至少 2 小时")
        return
    
    if too_late:
        logger.warning(
            f"以下发布时间距现在超过 15 天，B站可能拒绝:\n"
            + "\n".join(f"  {t.strftime('%Y-%m-%d %H:%M')}" for t in too_late)
        )
    
    logger.info(f"[DTIME] B站定时发布模式: {len(queue)}/{total} 个视频待上传")
    logger.info(f"[DTIME] Cron 表达式: {cron_expr}, 每次 {batch_size} 个, 共 {len(batches)} 批次")
    logger.info(f"[DTIME] 发布时间范围: "
                f"{batch_dtimes[0][2].strftime('%Y-%m-%d %H:%M')} ~ "
                f"{batch_dtimes[-1][2].strftime('%Y-%m-%d %H:%M')}")
    
    # 预览模式
    if dry_run:
        logger.info("[DRY-RUN] 定时发布计划:")
        vid_idx = 0
        for batch, dtime_ts, next_time in batch_dtimes:
            logger.info(f"  批次 @ {next_time.strftime('%Y-%m-%d %H:%M')} ({len(batch)} 个):")
            for vid in batch:
                vid_idx += 1
                video = db.get_video(vid)
                title = video.title[:40] if video and video.title else vid
                logger.info(f"    {vid_idx}. {title}")
        return
    
    # 立即逐个上传，每个视频携带对应的 dtime
    uploaded = 0
    failed = 0
    
    for batch_idx, (batch, dtime_ts, next_time) in enumerate(batch_dtimes):
        logger.info(
            f"[DTIME] 上传批次 {batch_idx + 1}/{len(batches)} "
            f"(定时发布 @ {next_time.strftime('%Y-%m-%d %H:%M')}, {len(batch)} 个视频)"
        )
        for vid in batch:
            video = db.get_video(vid)
            title = video.title[:30] if video and video.title else vid
            logger.info(f"[DTIME] 上传 ({uploaded + failed + 1}/{len(queue)}): {title}")
            try:
                processor = VideoProcessor(
                    video_id=vid,
                    config=config,
                    gpu_id=None,
                    force=force,
                    video_index=uploaded + failed,
                    total_videos=len(queue),
                    playlist_id=playlist_id,
                    upload_dtime=dtime_ts
                )
                success = processor.process(steps=['upload'])
                if success:
                    uploaded += 1
                    logger.info(
                        f"[DTIME] 上传成功 ({uploaded}/{len(queue)}): {title} "
                        f"→ 定时 {next_time.strftime('%m-%d %H:%M')}"
                    )
                else:
                    failed += 1
                    logger.warning(f"[DTIME] 上传失败 ({title})，继续下一个")
            except Exception as e:
                failed += 1
                logger.error(f"[DTIME] 上传异常 ({title}): {e}")
    
    logger.info(
        f"[DTIME] 定时发布上传完成: "
        f"成功 {uploaded}, 失败 {failed}, 总计 {len(queue)}"
    )
    if uploaded > 0:
        logger.info(
            f"[DTIME] 视频已全部上传到 B站，将在对应时间自动发布 "
            f"({batch_dtimes[0][2].strftime('%m-%d %H:%M')} ~ "
            f"{batch_dtimes[-1][2].strftime('%m-%d %H:%M')})"
        )


def _format_duration(seconds: float) -> str:
    """格式化等待时长为人类可读字符串"""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}秒"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}分{seconds % 60}秒"
    hours = minutes // 60
    remaining_min = minutes % 60
    if hours < 24:
        return f"{hours}时{remaining_min}分"
    days = hours // 24
    remaining_hours = hours % 24
    return f"{days}天{remaining_hours}时{remaining_min}分"


def _generate_process_cli(video_ids: List[str], stages: str, gpu: str, force: bool) -> str:
    """生成等价的 CLI 命令"""
    parts = ["python -m vat process"]
    
    for vid in video_ids:
        parts.append(f"-v {vid}")
    
    if stages != 'all':
        parts.append(f"-s {stages}")
    
    if gpu != 'auto':
        parts.append(f"-g {gpu}")
    
    if force:
        parts.append("-f")
    
    return " ".join(parts)


# ==================== Playlist 管理命令 ====================

@cli.group()
def playlist():
    """Playlist 管理"""
    pass


@playlist.command('add')
@click.argument('url')
@click.option('--sync/--no-sync', default=True, help='是否立即同步')
@click.pass_context
def playlist_add(ctx, url, sync):
    """添加 Playlist"""
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    downloader = YouTubeDownloader(
        proxy=config.get_stage_proxy("downloader"),
        video_format=config.downloader.youtube.format,
        cookies_file=config.downloader.youtube.cookies_file,
        remote_components=config.downloader.youtube.remote_components,
    )
    
    playlist_service = PlaylistService(db, downloader)
    
    try:
        result = playlist_service.sync_playlist(
            url,
            auto_add_videos=sync,
            progress_callback=lambda msg: click.echo(msg)
        )
        
        click.echo(f"\n✓ Playlist 已添加: {result.playlist_id}")
        click.echo(f"  新增视频: {result.new_count}")
        click.echo(f"  已存在: {result.existing_count}")
        click.echo(f"  总数: {result.total_videos}")
        
    except Exception as e:
        click.echo(f"✗ 添加失败: {e}", err=True)


@playlist.command('list')
@click.pass_context
def playlist_list(ctx):
    """列出所有 Playlist"""
    config = get_config(ctx.obj.get('config_path'))
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    playlists = db.list_playlists()
    
    if not playlists:
        click.echo("没有 Playlist")
        return
    
    table_data = []
    for pl in playlists:
        table_data.append([
            pl.id[:12],
            pl.title[:30] if pl.title else '-',
            pl.channel[:20] if pl.channel else '-',
            pl.video_count or 0,
            pl.last_synced_at.strftime('%Y-%m-%d %H:%M') if pl.last_synced_at else '-'
        ])
    
    headers = ['ID', '标题', '频道', '视频数', '最后同步']
    click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))


@playlist.command('sync')
@click.argument('playlist_id')
@click.pass_context
def playlist_sync(ctx, playlist_id):
    """同步 Playlist（增量更新）"""
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    pl = db.get_playlist(playlist_id)
    if not pl:
        click.echo(f"错误: Playlist 不存在: {playlist_id}", err=True)
        return
    
    downloader = YouTubeDownloader(
        proxy=config.get_stage_proxy("downloader"),
        video_format=config.downloader.youtube.format,
        cookies_file=config.downloader.youtube.cookies_file,
        remote_components=config.downloader.youtube.remote_components,
    )
    
    playlist_service = PlaylistService(db, downloader)
    
    try:
        result = playlist_service.sync_playlist(
            pl.source_url,
            auto_add_videos=True,
            progress_callback=lambda msg: click.echo(msg),
            target_playlist_id=playlist_id
        )
        
        click.echo(f"\n✓ 同步完成")
        click.echo(f"  新增视频: {result.new_count}")
        click.echo(f"  已存在: {result.existing_count}")
        
    except Exception as e:
        click.echo(f"✗ 同步失败: {e}", err=True)


@playlist.command('refresh')
@click.argument('playlist_id')
@click.option('--force-refetch', is_flag=True, help='强制重新获取所有字段（覆盖已有值，但保留翻译结果）')
@click.option('--force-retranslate', is_flag=True, help='强制重新翻译标题/简介（需配合 --force-refetch）')
@click.pass_context
def playlist_refresh(ctx, playlist_id, force_refetch, force_retranslate):
    """刷新 Playlist 视频信息（补全缺失的封面、时长、日期等）
    
    默认 merge 模式：仅补全缺失字段，不破坏已有数据。
    
    示例:
    
      # 补全缺失信息（默认 merge）
      vat playlist refresh PLAYLIST_ID
      
      # 强制重新获取所有信息
      vat playlist refresh PLAYLIST_ID --force-refetch
      
      # 强制重新获取 + 重新翻译
      vat playlist refresh PLAYLIST_ID --force-refetch --force-retranslate
    """
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    pl = db.get_playlist(playlist_id)
    if not pl:
        click.echo(f"错误: Playlist 不存在: {playlist_id}", err=True)
        return
    
    if force_retranslate and not force_refetch:
        click.echo("提示: --force-retranslate 需配合 --force-refetch 使用。"
                    "如只需重新翻译，请使用 'vat playlist retranslate'", err=True)
        return
    
    downloader = YouTubeDownloader(
        proxy=config.get_stage_proxy("downloader"),
        video_format=config.downloader.youtube.format,
        cookies_file=config.downloader.youtube.cookies_file,
        remote_components=config.downloader.youtube.remote_components,
    )
    
    playlist_service = PlaylistService(db, downloader)
    
    try:
        result = playlist_service.refresh_videos(
            playlist_id,
            force_refetch=force_refetch,
            force_retranslate=force_retranslate,
            callback=lambda msg: click.echo(msg)
        )
        
        click.echo(f"\n✓ 刷新完成")
        click.echo(f"  成功: {result['refreshed']}")
        click.echo(f"  失败: {result['failed']}")
        click.echo(f"  跳过: {result['skipped']}")
        
    except Exception as e:
        click.echo(f"✗ 刷新失败: {e}", err=True)


@playlist.command('show')
@click.argument('playlist_id')
@click.pass_context
def playlist_show(ctx, playlist_id):
    """显示 Playlist 详情"""
    config = get_config(ctx.obj.get('config_path'))
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    playlist_service = PlaylistService(db)
    
    pl = playlist_service.get_playlist(playlist_id)
    if not pl:
        click.echo(f"错误: Playlist 不存在: {playlist_id}", err=True)
        return
    
    click.echo(f"\nPlaylist: {pl.title}")
    click.echo(f"  ID: {pl.id}")
    click.echo(f"  URL: {pl.source_url}")
    click.echo(f"  频道: {pl.channel or '-'}")
    click.echo(f"  视频数: {pl.video_count or 0}")
    click.echo(f"  最后同步: {pl.last_synced_at.strftime('%Y-%m-%d %H:%M') if pl.last_synced_at else '-'}")
    
    # 显示进度统计
    progress = playlist_service.get_playlist_progress(playlist_id)
    click.echo(f"\n处理进度:")
    click.echo(f"  完成: {progress['completed']}/{progress['total']}")
    click.echo(f"  待处理: {progress['pending']}")
    
    # 显示视频列表
    videos = playlist_service.get_playlist_videos(playlist_id)
    if videos:
        click.echo(f"\n视频列表:")
        table_data = []
        for v in videos[:20]:
            pending = db.get_pending_steps(v.id)
            status = "✓" if not pending else f"待: {len(pending)}"
            table_data.append([
                v.playlist_index or '-',
                v.id[:12],
                v.title[:35] if v.title else '-',
                status
            ])
        
        headers = ['#', 'ID', '标题', '状态']
        click.echo(tabulate(table_data, headers=headers, tablefmt='simple'))
        
        if len(videos) > 20:
            click.echo(f"  ... 还有 {len(videos) - 20} 个视频")


@playlist.command('delete')
@click.argument('playlist_id')
@click.option('--delete-videos', is_flag=True, help='同时删除关联的视频记录和处理产物')
@click.option('--yes', '-y', is_flag=True, help='跳过确认')
@click.pass_context
def playlist_delete(ctx, playlist_id, delete_videos, yes):
    """删除 Playlist"""
    config = get_config(ctx.obj.get('config_path'))
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    playlist_service = PlaylistService(db)
    pl = playlist_service.get_playlist(playlist_id)
    if not pl:
        click.echo(f"错误: Playlist 不存在: {playlist_id}", err=True)
        return
    
    action = "删除 Playlist + 关联视频记录和处理产物" if delete_videos else "仅删除 Playlist 记录（保留视频）"
    click.echo(f"Playlist: {pl.title}")
    click.echo(f"操作: {action}")
    
    if not yes and not click.confirm("确认?"):
        click.echo("已取消")
        return
    
    result = playlist_service.delete_playlist(playlist_id, delete_videos=delete_videos)
    click.echo(f"✓ 已删除 Playlist: {pl.title}")
    if delete_videos:
        click.echo(f"  删除视频: {result.get('deleted_videos', 0)} 个")


# =============================================================================
# 上传命令组
# =============================================================================

@cli.group()
def upload():
    """上传相关命令（单视频上传、批量上传、合集同步等）"""
    pass


@upload.command('video')
@click.argument('video_id')
@click.option('--platform', '-p', default='bilibili', help='上传平台 (目前仅支持 bilibili)')
@click.option('--playlist', 'upload_playlist_id', default=None, help='指定上传上下文的 Playlist ID（视频属于多个 playlist 时用于确定正确的上下文）')
@click.option('--season', '-s', type=int, help='添加到合集ID (上传后自动添加)')
@click.option('--dry-run', is_flag=True, help='仅预览，不实际上传')
@click.pass_context
def upload_video(ctx, video_id, upload_playlist_id, platform, season, dry_run):
    """上传单个视频到指定平台
    
    VIDEO_ID: 视频ID
    
    示例:
    
      vat upload video VIDEO_ID
      vat upload video VIDEO_ID --season 7376902
      vat upload video VIDEO_ID --playlist PLAYLIST_ID --season 7376902
    """
    config = get_config(ctx.obj.get('config_path'))
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    logger = get_logger()
    
    # 获取视频记录
    video = db.get_video(video_id)
    if not video:
        click.echo(f"✗ 视频不存在: {video_id}", err=True)
        return
    
    # 检查视频文件
    if not video.output_dir:
        click.echo(f"✗ 视频未处理完成，没有输出目录", err=True)
        return
    
    final_video = Path(video.output_dir) / "final.mp4"
    if not final_video.exists():
        click.echo(f"✗ 最终视频文件不存在: {final_video}", err=True)
        return
    
    # 渲染上传元数据
    from ..uploaders.template import render_upload_metadata
    
    bilibili_config = config.uploader.bilibili
    templates = {}
    if bilibili_config.templates:
        templates = {
            'title': bilibili_config.templates.title,
            'description': bilibili_config.templates.description,
            'custom_vars': bilibili_config.templates.custom_vars,
        }
    
    # 获取播放列表信息
    # 优先使用显式传入的 playlist_id，回退到 playlist_videos 关联表查询
    effective_playlist_id = upload_playlist_id
    if not effective_playlist_id:
        video_playlists = db.get_video_playlists(video_id)
        if len(video_playlists) == 1:
            effective_playlist_id = video_playlists[0]
        elif len(video_playlists) > 1:
            click.echo(
                f"✗ 视频 {video_id} 属于多个 playlist ({video_playlists})，"
                "请通过 --playlist 指定目标 playlist", err=True
            )
            return
    playlist_info = None
    if effective_playlist_id:
        playlist_service = PlaylistService(db)
        pl = playlist_service.get_playlist(effective_playlist_id)
        if pl:
            pl_upload_config = (pl.metadata or {}).get('upload_config', {})
            # upload_order_index: 只从 playlist_videos 关联表读取（per-playlist）
            pv_info = db.get_playlist_video_info(effective_playlist_id, video_id)
            upload_order_index = pv_info.get('upload_order_index', 0) if pv_info else 0
            assert upload_order_index, (
                f"视频 {video_id} 在 playlist {effective_playlist_id} 中缺少 upload_order_index，"
                "请先执行 playlist sync 以分配时间顺序索引"
            )
            playlist_info = {
                'name': pl.title,
                'id': pl.id,
                'index': upload_order_index,
                'uploader_name': pl_upload_config.get('uploader_name', ''),
            }
    
    # 渲染元数据
    rendered = render_upload_metadata(video, templates, playlist_info)
    
    click.echo(f"\n视频: {video_id}")
    click.echo(f"文件: {final_video}")
    click.echo(f"大小: {final_video.stat().st_size / 1024 / 1024:.1f} MB")
    click.echo()
    click.echo("上传信息:")
    click.echo(f"  标题: {rendered['title'][:80]}")
    click.echo(f"  简介: {rendered['description'][:100]}...")
    click.echo(f"  平台: {platform}")
    
    if dry_run:
        click.echo()
        click.echo("--dry-run 模式，跳过实际上传")
        click.echo("✓ 预览完成")
        return
    
    # 确认上传
    if not click.confirm("\n确认上传?"):
        click.echo("已取消")
        return
    
    # 执行上传
    if platform == 'bilibili':
        from ..uploaders.bilibili import BilibiliUploader
        
        project_root = Path(__file__).parent.parent.parent
        cookies_file = project_root / bilibili_config.cookies_file
        
        uploader = BilibiliUploader(
            cookies_file=str(cookies_file),
            line=bilibili_config.line,
            threads=bilibili_config.threads
        )
        
        # 查找封面
        cover_path = None
        if bilibili_config.auto_cover:
            for cover_name in ['thumbnail.jpg', 'thumbnail.png', 'cover.jpg', 'cover.png']:
                potential = Path(video.output_dir) / cover_name
                if potential.exists():
                    cover_path = potential
                    break
        
        # 获取其他配置
        copyright_type = bilibili_config.copyright
        default_tid = bilibili_config.default_tid
        default_tags = bilibili_config.default_tags
        
        # 从翻译结果获取标签和分区
        metadata = video.metadata or {}
        translated = metadata.get('translated', {})
        tags = translated.get('tags', default_tags)
        tid = translated.get('recommended_tid', default_tid)
        
        click.echo()
        click.echo("开始上传...")
        
        result = uploader.upload(
            video_path=final_video,
            title=rendered['title'][:80],
            description=rendered['description'][:2000],
            tid=tid,
            tags=tags,
            copyright=copyright_type,
            source=video.source_url if copyright_type == 2 else '',
            cover_path=cover_path,
        )
        
        if result.success:
            click.echo()
            click.echo("=" * 50)
            click.echo("✓ 上传成功!")
            click.echo(f"  BV号: {result.bvid}")
            click.echo(f"  链接: https://www.bilibili.com/video/{result.bvid}")
            click.echo("=" * 50)
            
            # 更新数据库（合并到现有 metadata）
            video_obj = db.get_video(video_id)
            updated_metadata = dict(video_obj.metadata) if video_obj and video_obj.metadata else {}
            updated_metadata.update({
                'bilibili_bvid': result.bvid,
                'bilibili_aid': result.aid or 0,
                'bilibili_url': f"https://www.bilibili.com/video/{result.bvid}",
                'uploaded_at': datetime.now().isoformat(),
            })
            
            # 添加到合集（尝试一次，不阻塞重试）
            season_added = False
            if season:
                updated_metadata['bilibili_target_season_id'] = season
                aid = result.aid if result.aid else None
                if aid:
                    click.echo(f"\n添加到合集 {season} (AV号: {aid})...")
                    try:
                        if uploader.add_to_season(aid, season):
                            click.echo(f"✓ 已添加到合集")
                            season_added = True
                        else:
                            click.echo(f"⚠ 添加到合集失败（视频可能尚未索引），请稍后运行 vat upload sync", err=True)
                    except Exception as e:
                        click.echo(f"⚠ 添加到合集异常: {e}，请稍后运行 vat upload sync", err=True)
                else:
                    click.echo(f"⚠ 上传响应中无 AV号，请稍后运行 vat upload sync", err=True)
                updated_metadata['bilibili_season_added'] = season_added
            
            db.update_video(video_id, metadata=updated_metadata)
        else:
            click.echo(f"✗ 上传失败: {result.error}", err=True)
    else:
        click.echo(f"✗ 不支持的平台: {platform}", err=True)


@upload.command('sync')
@click.option('--playlist', '-p', required=True, help='Playlist ID（必须指定）')
@click.option('--retry-delay', default=30, type=int, help='失败后自动重试的等待时间（分钟），0=不自动重试')
@click.pass_context
def upload_sync(ctx, playlist, retry_delay):
    """将已上传但未入集的视频批量添加到B站合集并排序
    
    查找指定 playlist 中所有已上传到B站（有 aid）但尚未添加到目标合集的视频，
    批量执行 add_to_season，然后对合集按 #数字 自动排序。
    
    示例:
    
      # 立即同步
      vat upload sync -p PLAYLIST_ID
      
      # 同步，失败后等60分钟自动重试
      vat upload sync -p PLAYLIST_ID --retry-delay 60
      
      # 同步，不自动重试
      vat upload sync -p PLAYLIST_ID --retry-delay 0
    """
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    # 验证 playlist 存在
    playlist_service = PlaylistService(db)
    pl = playlist_service.get_playlist(playlist)
    if not pl:
        click.echo(f"错误: Playlist 不存在: {playlist}", err=True)
        return
    
    click.echo(f"Playlist: {pl.title} ({playlist})")
    
    try:
        from ..uploaders.bilibili import BilibiliUploader, season_sync, BILIUP_AVAILABLE
        
        if not BILIUP_AVAILABLE:
            click.echo("错误: biliup 库不可用，请安装: pip install biliup", err=True)
            return
        
        bilibili_config = config.uploader.bilibili
        project_root = Path(__file__).parent.parent.parent
        cookies_file = project_root / bilibili_config.cookies_file
        
        uploader = BilibiliUploader(
            cookies_file=str(cookies_file),
            line=bilibili_config.line,
            threads=bilibili_config.threads
        )
        
        # 第一次同步
        click.echo("开始 upload sync...")
        result = season_sync(db, uploader, playlist)
        
        click.echo(f"\n结果: {result['success']} 成功, {result['failed']} 失败, 共 {result['total']} 个")
        
        if result['failed'] > 0 and retry_delay > 0:
            click.echo(f"\n有 {result['failed']} 个视频同步失败，{retry_delay} 分钟后自动重试...")
            import time
            time.sleep(retry_delay * 60)
            
            # 重试（重新创建 uploader）
            uploader2 = BilibiliUploader(
                cookies_file=str(cookies_file),
                line=bilibili_config.line,
                threads=bilibili_config.threads
            )
            click.echo("开始重试...")
            result2 = season_sync(db, uploader2, playlist)
            click.echo(f"重试结果: {result2['success']} 成功, {result2['failed']} 失败")
            
            if result2['failed'] > 0:
                click.echo(f"仍有 {result2['failed']} 个视频失败，请稍后再次运行此命令")
        elif result['failed'] > 0:
            click.echo(f"有 {result['failed']} 个视频失败，请稍后重新运行: vat upload sync -p {playlist}")
        
        if result['total'] == 0:
            click.echo("没有待同步的视频")
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        logger.debug(f"upload sync 异常: {e}", exc_info=True)


@upload.command('update-info')
@click.option('--playlist', '-p', required=True, help='Playlist ID（必须指定）')
@click.option('--dry-run', is_flag=True, help='仅预览，不实际修改')
@click.option('--yes', '-y', is_flag=True, help='跳过确认')
@click.pass_context
def upload_update_info(ctx, playlist, dry_run, yes):
    """批量更新已上传视频的标题和简介（重新渲染模板）
    
    读取 playlist 中所有已上传到B站的视频，使用当前模板和翻译结果
    重新渲染标题/简介，然后调用B站 API 更新。
    
    适用场景：修改了翻译提示词并重新翻译后，需要同步更新B站视频信息。
    
    示例:
    
      vat upload update-info -p PLAYLIST_ID --dry-run
      vat upload update-info -p PLAYLIST_ID -y
    """
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    from ..uploaders.template import render_upload_metadata
    
    playlist_service = PlaylistService(db)
    pl = playlist_service.get_playlist(playlist)
    if not pl:
        click.echo(f"错误: Playlist 不存在: {playlist}", err=True)
        return
    
    # 获取模板配置
    bilibili_config = config.uploader.bilibili
    templates = {}
    if bilibili_config.templates:
        templates = {
            'title': bilibili_config.templates.title,
            'description': bilibili_config.templates.description,
            'custom_vars': bilibili_config.templates.custom_vars,
        }
    
    pl_upload_config = (pl.metadata or {}).get('upload_config', {})
    
    # 筛选已上传的视频
    videos = playlist_service.get_playlist_videos(playlist)
    uploaded = []
    for v in videos:
        meta = v.metadata or {}
        aid = meta.get('bilibili_aid')
        if aid:
            uploaded.append((v, int(aid)))
    
    if not uploaded:
        click.echo("没有已上传到B站的视频")
        return
    
    click.echo(f"Playlist: {pl.title}")
    click.echo(f"已上传视频: {len(uploaded)} 个")
    
    # 预览模式：显示新旧对比
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
        new_title = rendered['title'][:80]
        new_desc = rendered['description'][:2000]
        
        # 获取翻译结果中的标签和分区
        meta = v.metadata or {}
        translated = meta.get('translated', {})
        new_tags = translated.get('tags', None)
        new_tid = translated.get('recommended_tid', None)
        
        update_list.append({
            'video': v,
            'aid': aid,
            'new_title': new_title,
            'new_desc': new_desc,
            'new_tags': new_tags,
            'new_tid': new_tid,
        })
    
    if dry_run:
        click.echo("\n--dry-run 模式，预览更新:")
        for i, item in enumerate(update_list, 1):
            click.echo(f"\n  [{i}] av{item['aid']}")
            click.echo(f"    新标题: {item['new_title'][:60]}")
            click.echo(f"    新简介: {item['new_desc'][:80]}...")
        click.echo(f"\n共 {len(update_list)} 个视频待更新")
        return
    
    if not yes and not click.confirm(f"\n确认更新 {len(update_list)} 个视频的标题和简介?"):
        click.echo("已取消")
        return
    
    # 执行更新
    try:
        from ..uploaders.bilibili import BilibiliUploader
        
        project_root = Path(__file__).parent.parent.parent
        cookies_file = project_root / bilibili_config.cookies_file
        uploader = BilibiliUploader(
            cookies_file=str(cookies_file),
            line=bilibili_config.line,
            threads=bilibili_config.threads
        )
        
        success = 0
        failed = 0
        for i, item in enumerate(update_list, 1):
            v = item['video']
            title_short = item['new_title'][:40]
            click.echo(f"[{i}/{len(update_list)}] av{item['aid']}: {title_short}...")
            
            if uploader.edit_video_info(
                aid=item['aid'],
                title=item['new_title'],
                desc=item['new_desc'],
                tags=item['new_tags'],
                tid=item['new_tid'],
            ):
                success += 1
                click.echo(f"  ✓ 已更新")
            else:
                failed += 1
                click.echo(f"  ✗ 更新失败", err=True)
            
            # 简单限速，避免触发B站反爬
            import time
            time.sleep(1)
        
        click.echo(f"\n完成: {success} 成功, {failed} 失败")
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        logger.debug(f"update-info 异常: {e}", exc_info=True)


@upload.command('sync-db')
@click.option('--season', '-s', required=True, type=int, help='B站合集ID')
@click.option('--playlist', '-p', required=True, help='对应的 Playlist ID')
@click.option('--dry-run', is_flag=True, help='仅预览匹配结果，不实际修改DB')
@click.pass_context
def upload_sync_db(ctx, season, playlist, dry_run):
    """将B站合集中的视频信息同步回数据库
    
    从B站合集获取所有视频的 aid/bvid，通过转载来源（source URL）匹配到
    数据库中的视频记录，补全 bilibili_aid、bilibili_bvid、
    bilibili_target_season_id、bilibili_season_added 等字段。
    
    适用场景：数据库重构后，已上传视频的B站信息丢失，需要重新同步。
    
    示例:
    
      vat upload sync-db -s 7376902 -p PLAYLIST_ID --dry-run
      vat upload sync-db -s 7376902 -p PLAYLIST_ID
    """
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    playlist_service = PlaylistService(db)
    pl = playlist_service.get_playlist(playlist)
    if not pl:
        click.echo(f"错误: Playlist 不存在: {playlist}", err=True)
        return
    
    try:
        from ..uploaders.bilibili import BilibiliUploader
        
        bilibili_config = config.uploader.bilibili
        project_root = Path(__file__).parent.parent.parent
        cookies_file = project_root / bilibili_config.cookies_file
        uploader = BilibiliUploader(
            cookies_file=str(cookies_file),
            line=bilibili_config.line,
            threads=bilibili_config.threads
        )
        
        # 1. 获取合集中的所有视频
        click.echo(f"获取合集 {season} 的视频列表...")
        season_info = uploader.get_season_episodes(season)
        if not season_info:
            click.echo("错误: 无法获取合集信息", err=True)
            return
        
        episodes = season_info.get('episodes', [])
        click.echo(f"合集中共 {len(episodes)} 个视频")
        
        # 2. 获取 playlist 中的视频，建立 source_url → video 映射
        pl_videos = playlist_service.get_playlist_videos(playlist)
        url_to_video = {}
        aid_to_video = {}
        for v in pl_videos:
            if v.source_url:
                # 标准化 YouTube URL（去除多余参数）
                normalized = v.source_url.split('&')[0] if '&' in v.source_url else v.source_url
                url_to_video[normalized] = v
                url_to_video[v.source_url] = v
            # 如果 DB 中已有 bilibili_aid，也建立映射
            meta = v.metadata or {}
            if meta.get('bilibili_aid'):
                aid_to_video[int(meta['bilibili_aid'])] = v
        
        # 3. 建立 video_id → video 映射（从 source_url 提取 YouTube video ID）
        import re
        vid_to_video = {}
        for v in pl_videos:
            if v.source_url:
                yt_match = re.search(r'[?&]v=([a-zA-Z0-9_-]+)', v.source_url)
                if yt_match:
                    vid_to_video[yt_match.group(1)] = v
            # 也用 video.id 本身（通常就是 YouTube video ID）
            vid_to_video[v.id] = v
        
        # 4. 对每个 episode，尝试匹配 DB 记录
        matched = []
        unmatched = []
        
        for ep in episodes:
            aid = ep.get('aid')
            bvid_ep = ep.get('bvid', '')
            
            # 方法1: aid 直接匹配（DB 中已有 bilibili_aid 的情况）
            if aid in aid_to_video:
                matched.append((ep, aid_to_video[aid], 'aid'))
                continue
            
            # 方法2: 从B站视频描述中提取 YouTube video ID 匹配
            # 公共 API 的 desc 中包含上传模板渲染的 source_url
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
            import time
            time.sleep(0.3)  # 限速
        
        # 4. 显示匹配结果
        click.echo(f"\n匹配结果: {len(matched)} 匹配, {len(unmatched)} 未匹配")
        
        if matched:
            click.echo("\n已匹配:")
            for ep, v, method in matched:
                meta = v.metadata or {}
                has_info = '已有' if meta.get('bilibili_aid') else '缺失'
                click.echo(f"  av{ep['aid']} → {v.id[:12]} ({v.title[:30] if v.title else '-'}) [{method}] bilibili信息:{has_info}")
        
        if unmatched:
            click.echo("\n未匹配（需手动处理）:")
            for ep in unmatched:
                click.echo(f"  av{ep['aid']}: {ep.get('title', '?')[:40]}")
        
        if dry_run:
            click.echo("\n--dry-run 模式，不修改数据库")
            return
        
        # 5. 更新 DB
        updated = 0
        for ep, v, method in matched:
            meta = dict(v.metadata) if v.metadata else {}
            aid = ep['aid']
            
            # 获取 bvid（如果还没有）
            bvid = meta.get('bilibili_bvid', '')
            if not bvid:
                detail = uploader.get_video_detail(aid)
                if detail:
                    bvid = detail.get('bvid', '')
            
            meta['bilibili_aid'] = aid
            if bvid:
                meta['bilibili_bvid'] = bvid
                meta['bilibili_url'] = f"https://www.bilibili.com/video/{bvid}"
            meta['bilibili_target_season_id'] = season
            meta['bilibili_season_added'] = True
            
            db.update_video(v.id, metadata=meta)
            updated += 1
        
        click.echo(f"\n✓ 已更新 {updated} 个视频的数据库记录")
        
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        import traceback
        logger.debug(traceback.format_exc())


# =============================================================================
# B站子命令组
# =============================================================================

@cli.group()
@click.pass_context
def bilibili(ctx):
    """B站相关功能（登录、合集管理等）"""
    pass


def _get_bilibili_uploader(ctx):
    """获取 B站上传器实例"""
    from ..uploaders.bilibili import BilibiliUploader
    
    config = get_config(ctx.obj.get('config_path'))
    bilibili_config = config.uploader.bilibili
    project_root = Path(__file__).parent.parent.parent
    cookies_file = project_root / bilibili_config.cookies_file
    
    return BilibiliUploader(cookies_file=str(cookies_file))


@bilibili.command('seasons')
@click.pass_context
def bilibili_list_seasons(ctx):
    """列出合集列表"""
    uploader = _get_bilibili_uploader(ctx)
    seasons = uploader.list_seasons()
    
    if not seasons:
        click.echo("没有找到合集，或获取失败")
        return
    
    click.echo(f"\n找到 {len(seasons)} 个合集:\n")
    
    table_data = []
    for s in seasons:
        name = s.get('name') or '(未命名)'
        table_data.append([
            s['season_id'],
            name[:30],
            s['total'],
        ])
    
    headers = ['ID', '名称', '视频数']
    click.echo(tabulate(table_data, headers=headers, tablefmt='simple'))
    click.echo()
    click.echo("使用示例: vat upload video VIDEO_ID --season SEASON_ID")


@bilibili.command('create-season')
@click.argument('title')
@click.option('--desc', '-d', default='', help='合集简介')
@click.pass_context
def bilibili_create_season(ctx, title, desc):
    """创建新合集"""
    uploader = _get_bilibili_uploader(ctx)
    
    season_id = uploader.create_season(title, desc)
    
    if season_id:
        click.echo(f"✓ 合集创建成功!")
        click.echo(f"  ID: {season_id}")
        click.echo(f"  标题: {title}")
    else:
        click.echo("✗ 创建合集失败", err=True)


@bilibili.command('login')
@click.pass_context
def bilibili_login(ctx):
    """扫码登录B站账号"""
    config = get_config(ctx.obj.get('config_path'))
    bilibili_config = config.uploader.bilibili
    project_root = Path(__file__).parent.parent.parent
    cookies_file = project_root / bilibili_config.cookies_file
    
    # 确保目录存在
    cookies_file.parent.mkdir(parents=True, exist_ok=True)
    
    click.echo("正在获取登录二维码...")
    
    try:
        import stream_gears
        
        # 获取二维码
        qr_data = stream_gears.get_qrcode(None)
        
        import json
        qr_info = json.loads(qr_data)
        qr_url = qr_info.get('data', {}).get('url', '')
        
        if not qr_url:
            click.echo("✗ 获取二维码失败", err=True)
            return
        
        # 生成二维码
        import qrcode
        qr = qrcode.QRCode(version=1, box_size=1, border=1)
        qr.add_data(qr_url)
        qr.make(fit=True)
        
        click.echo("\n请使用B站APP扫描以下二维码登录:\n")
        qr.print_ascii(invert=True)
        click.echo(f"\n或访问: {qr_url}")
        click.echo("\n等待扫码登录...")
        
        # 等待登录
        result = stream_gears.login_by_qrcode(qr_data, None)
        
        # 保存 cookie
        with open(cookies_file, 'w', encoding='utf-8') as f:
            f.write(result)
        
        click.echo(f"\n✓ 登录成功!")
        click.echo(f"Cookie 已保存到: {cookies_file}")
        
    except ImportError:
        click.echo("✗ 需要安装 stream_gears 和 qrcode: pip install stream_gears qrcode", err=True)
    except Exception as e:
        click.echo(f"✗ 登录失败: {e}", err=True)


@bilibili.command('status')
@click.pass_context
def bilibili_status(ctx):
    """检查登录状态"""
    uploader = _get_bilibili_uploader(ctx)
    
    if uploader.validate_credentials():
        click.echo("✓ Cookie 有效")
        
        # 尝试获取用户信息
        try:
            session = uploader._get_authenticated_session()
            resp = session.get('https://api.bilibili.com/x/web-interface/nav', timeout=5)
            data = resp.json()
            
            if data.get('code') == 0:
                user_data = data.get('data', {})
                click.echo(f"  用户: {user_data.get('uname', '未知')}")
                click.echo(f"  UID: {user_data.get('mid', '未知')}")
                click.echo(f"  等级: Lv{user_data.get('level_info', {}).get('current_level', 0)}")
        except:
            pass
    else:
        click.echo("✗ Cookie 无效或已过期，请重新登录")
        click.echo("  运行: vat bilibili login")


@bilibili.command('rejected')
@click.option('--keyword', '-k', default='', help='搜索关键词')
@click.pass_context
def bilibili_rejected(ctx, keyword):
    """列出被退回的稿件及违规详情
    
    显示所有被退回的稿件，包括违规时间段和退回原因。
    带有具体违规时间段的视频可以通过 `vat bilibili fix` 命令自动修复。
    
    示例:
    
      vat bilibili rejected
      vat bilibili rejected -k 漆黒
    """
    uploader = _get_bilibili_uploader(ctx)
    
    rejected = uploader.get_rejected_videos(keyword=keyword)
    if not rejected:
        click.echo("没有被退回的稿件")
        return
    
    click.echo(f"\n被退回的稿件: {len(rejected)} 个\n")
    
    fixable_count = 0
    for v in rejected:
        all_ranges = []
        is_full = False
        for p in v['problems']:
            all_ranges.extend(p['time_ranges'])
            if p['is_full_video']:
                is_full = True
        
        if all_ranges and not is_full:
            status = "🔧 可修复"
            fixable_count += 1
        elif is_full:
            status = "❌ 全片违规"
        else:
            status = "⚠️  未知"
        
        click.echo(f"  {status} | aid={v['aid']} | {v['title'][:55]}")
        for p in v['problems']:
            click.echo(f"    原因: {p['reason'][:60]}")
            if p['violation_time']:
                click.echo(f"    时间: {p['violation_time']} → {p['time_ranges']}")
            if p['is_full_video']:
                click.echo(f"    位置: {p['violation_position']}（全片违规，无法自动修复）")
            if p['modify_advise']:
                click.echo(f"    建议: {p['modify_advise'][:60]}")
        click.echo()
    
    if fixable_count > 0:
        click.echo(f"其中 {fixable_count} 个可自动修复，使用:")
        click.echo(f"  vat bilibili fix --aid <AID>")
        click.echo(f"  vat bilibili fix --aid <AID> --dry-run  # 仅遮罩不上传")


def _find_local_video_cli(aid: int, config, db, uploader) -> Optional[Path]:
    """
    CLI 侧：根据 aid 查找本地视频文件路径。
    
    查找策略（按优先级）：
    1. 从 B站稿件 source URL 提取 YouTube video ID → DB 查找视频记录 → 本地 final.mp4
    2. DB 中通过 bilibili_aid 匹配 → 本地 final.mp4
    3. 通过 B站稿件标题匹配 DB 翻译标题 → 本地 final.mp4
    4. 直接按 YouTube video ID 查找 output 目录
    """
    import re
    
    yt_video_id = None
    bili_title = None
    
    # 方法1: source URL → YouTube video ID → DB
    try:
        detail = uploader.get_archive_detail(aid)
        if detail:
            archive = detail.get('archive', {})
            bili_title = archive.get('title', '')
            source = archive.get('source', '')
            yt_match = re.search(r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)', source)
            if yt_match:
                yt_video_id = yt_match.group(1)
                click.echo(f"  稿件对应 YouTube 视频: {yt_video_id}")
                
                video = db.get_video(yt_video_id)
                if video:
                    path = _resolve_video_file_cli(video, config)
                    if path:
                        click.echo(f"  通过 source URL 找到本地视频: {path}")
                        return path
    except Exception as e:
        click.echo(f"  通过 source URL 查找失败: {e}", err=True)
    
    # 方法2 + 3: 遍历 DB
    videos = db.list_videos()
    
    # 方法2: bilibili_aid 匹配
    for v in videos:
        meta = v.metadata or {}
        if str(meta.get('bilibili_aid', '')) == str(aid):
            path = _resolve_video_file_cli(v, config)
            if path:
                click.echo(f"  通过 bilibili_aid 找到本地视频: {path}")
                return path
    
    # 方法3: 标题匹配
    if bili_title:
        clean_title = re.sub(r'\s*\|\s*#\d+\s*$', '', bili_title).strip()
        for v in videos:
            meta = v.metadata or {}
            translated = meta.get('translated', {})
            t_title = translated.get('title_translated', '') if translated else ''
            if t_title and clean_title and (clean_title in t_title or t_title in clean_title):
                path = _resolve_video_file_cli(v, config)
                if path:
                    click.echo(f"  通过标题匹配找到本地视频: {v.id} → {path}")
                    return path
    
    # 方法4: output 目录直接查找
    if yt_video_id:
        vid_dir = Path(config.storage.output_dir) / yt_video_id
        for name in ['final.mp4', f'{yt_video_id}.mp4']:
            candidate = vid_dir / name
            if candidate.exists():
                click.echo(f"  通过 output 目录找到视频: {candidate}")
                return candidate
    
    return None


def _resolve_video_file_cli(video, config) -> Optional[Path]:
    """从视频记录解析本地视频文件路径（final.mp4 优先）"""
    candidates = []
    if video.output_dir:
        candidates.append(Path(video.output_dir) / "final.mp4")
    vid_dir = Path(config.storage.output_dir) / video.id
    candidates.append(vid_dir / "final.mp4")
    candidates.append(vid_dir / f"{video.id}.mp4")
    
    for c in candidates:
        if c.exists():
            return c
    return None


def _download_from_bilibili_cli(aid: int, bvid: str, config, logger) -> Optional[Path]:
    """
    CLI 侧：从 B站下载视频作为 fallback。
    使用 yt-dlp 下载 B站视频到临时目录。
    """
    import subprocess
    import tempfile
    
    url = f"https://www.bilibili.com/video/{bvid}" if bvid else f"https://www.bilibili.com/video/av{aid}"
    
    click.echo(f"  ⚠️ 本地视频文件未找到，将从 B站下载原视频: {url}")
    click.echo(f"  （建议在清理本地文件前完成审核修复）")
    
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bilibili_fix_{aid}_"))
    output_template = str(tmp_dir / f"av{aid}.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        url,
    ]
    
    try:
        click.echo(f"  下载中...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            click.echo(f"  ✗ 下载失败: {result.stderr[-300:]}", err=True)
            return None
        
        mp4_files = list(tmp_dir.glob("*.mp4"))
        if mp4_files:
            size_mb = mp4_files[0].stat().st_size / 1024 / 1024
            click.echo(f"  ✓ 下载成功: {mp4_files[0].name} ({size_mb:.1f}MB)")
            return mp4_files[0]
        
        click.echo(f"  ✗ 下载完成但未找到 mp4 文件", err=True)
        return None
        
    except subprocess.TimeoutExpired:
        click.echo(f"  ✗ 下载超时 (>10分钟)", err=True)
        return None
    except Exception as e:
        click.echo(f"  ✗ 下载异常: {e}", err=True)
        return None


@bilibili.command('fix')
@click.option('--aid', required=True, type=int, help='要修复的稿件 AV号')
@click.option('--video-path', type=click.Path(exists=True), help='本地视频文件路径（默认自动查找）')
@click.option('--margin', default=1.0, type=float, help='违规区间前后扩展的安全边距（秒），默认1.0')
@click.option('--mask-text', default='此处内容因平台合规要求已被遮罩', help='遮罩区域显示的文字')
@click.option('--dry-run', is_flag=True, help='仅执行遮罩处理，不上传替换')
@click.option('--keep-masked', is_flag=True, help='上传后保留遮罩文件（默认删除）')
@click.option('--yes', '-y', is_flag=True, help='跳过确认')
@click.pass_context
def bilibili_fix(ctx, aid, video_path, margin, mask_text, dry_run, keep_masked, yes):
    """修复被退回的稿件：遮罩违规片段 + 重新上传
    
    自动获取审核退回信息，用 ffmpeg 遮罩违规时间段（黑屏+说明文字+静音），
    然后上传新视频文件替换原稿件并重新提交审核。
    
    流程:
      1. 从 B站 API 获取退回原因和违规时间段
      2. 定位本地视频文件（从 DB 或手动指定）
      3. ffmpeg 遮罩违规片段（GPU 加速）
      4. 上传新视频并编辑稿件替换（除非 --dry-run）
    
    示例:
    
      vat bilibili fix --aid 116089795185839
      vat bilibili fix --aid 116089795185839 --dry-run
      vat bilibili fix --aid 116089795185839 --video-path /path/to/video.mp4
      vat bilibili fix --aid 116089795185839 --margin 2.0
    """
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    uploader = _get_bilibili_uploader(ctx)
    
    # Step 1: 获取审核信息
    click.echo(f"获取稿件 av{aid} 的审核信息...")
    rejected = uploader.get_rejected_videos()
    target = [v for v in rejected if v['aid'] == aid]
    
    if not target:
        click.echo(f"✗ 未找到 aid={aid} 的退回稿件", err=True)
        click.echo("提示: 使用 `vat bilibili rejected` 查看所有退回稿件")
        return
    
    t = target[0]
    click.echo(f"  标题: {t['title'][:60]}")
    
    # 收集违规时间段
    all_ranges = []
    is_full = False
    for p in t['problems']:
        all_ranges.extend(p['time_ranges'])
        if p['is_full_video']:
            is_full = True
        click.echo(f"  原因: {p['reason'][:60]}")
        if p['violation_time']:
            click.echo(f"  时间: {p['violation_time']}")
        if p['is_full_video']:
            click.echo(f"  位置: {p['violation_position']}")
    
    if is_full:
        click.echo(f"\n✗ 全片违规，无法通过遮罩修复", err=True)
        return
    
    if not all_ranges:
        click.echo(f"\n✗ 无具体违规时间段，无法自动修复", err=True)
        return
    
    click.echo(f"\n  违规区间: {all_ranges}")
    click.echo(f"  安全边距: ±{margin}s")
    
    # Step 2: 定位本地视频文件
    if video_path:
        local_video = Path(video_path)
    else:
        local_video = _find_local_video_cli(aid, config, db, uploader)
        
        if not local_video:
            # fallback: 从 B站下载原视频
            click.echo(f"\n  ⚠️ 本地文件未找到，尝试从B站下载原视频...")
            local_video = _download_from_bilibili_cli(aid, t.get('bvid', ''), config, logger)
        
        if not local_video:
            click.echo(f"\n✗ 无法自动定位本地视频文件，且从B站下载失败", err=True)
            click.echo(f"  请使用 --video-path 手动指定视频文件路径")
            return
    
    # Step 3: ffmpeg 遮罩
    from ..embedder.ffmpeg_wrapper import FFmpegWrapper
    
    ffmpeg = FFmpegWrapper()
    info = ffmpeg.get_video_info(local_video)
    if not info:
        click.echo(f"✗ 无法获取视频信息: {local_video}", err=True)
        return
    
    click.echo(f"\n  原视频: {info['duration']:.0f}s, "
               f"{info.get('video', {}).get('width', '?')}x{info.get('video', {}).get('height', '?')}, "
               f"{local_video.stat().st_size / 1024 / 1024:.0f}MB")
    
    masked_video = local_video.parent / f"{local_video.stem}_masked{local_video.suffix}"
    
    if not yes and not dry_run:
        if not click.confirm(f"\n确认遮罩 {len(all_ranges)} 个违规区间并上传替换?"):
            click.echo("已取消")
            return
    elif not yes:
        if not click.confirm(f"\n确认遮罩 {len(all_ranges)} 个违规区间?"):
            click.echo("已取消")
            return
    
    click.echo(f"\n开始遮罩处理...")
    ok = ffmpeg.mask_violation_segments(
        video_path=local_video,
        output_path=masked_video,
        violation_ranges=all_ranges,
        mask_text=mask_text,
        margin_sec=margin,
    )
    
    if not ok:
        click.echo(f"\n✗ 遮罩处理失败", err=True)
        return
    
    out_size = masked_video.stat().st_size / 1024 / 1024
    click.echo(f"  ✓ 遮罩完成: {masked_video.name} ({out_size:.0f}MB)")
    
    # Step 4: 上传替换
    if dry_run:
        click.echo(f"\n--dry-run 模式，跳过上传")
        click.echo(f"  遮罩文件: {masked_video}")
        return
    
    click.echo(f"\n开始上传替换...")
    replace_ok = uploader.replace_video(aid, masked_video)
    
    if replace_ok:
        click.echo(f"\n✅ 稿件 av{aid} 已修复并重新提交审核")
        if not keep_masked:
            masked_video.unlink(missing_ok=True)
            click.echo(f"  已清理遮罩临时文件")
    else:
        click.echo(f"\n✗ 上传替换失败，遮罩文件保留在: {masked_video}", err=True)


@upload.command('playlist')
@click.argument('playlist_id')
@click.option('--platform', '-p', default='bilibili', help='上传平台')
@click.option('--season', '-s', type=int, help='添加到合集ID')
@click.option('--limit', '-n', type=int, help='最大上传数量')
@click.option('--dry-run', is_flag=True, help='仅预览，不实际上传')
@click.pass_context
def upload_playlist(ctx, playlist_id, platform, season, limit, dry_run):
    """批量上传播放列表中的视频
    
    PLAYLIST_ID: 播放列表ID
    
    示例:
    
      vat upload playlist PLAYLIST_ID
      vat upload playlist PLAYLIST_ID --season 7376902
      vat upload playlist PLAYLIST_ID --limit 5 --dry-run
    """
    config = get_config(ctx.obj.get('config_path'))
    db = Database(config.storage.database_path, output_base_dir=config.storage.output_dir)
    
    playlist_service = PlaylistService(db)
    pl = playlist_service.get_playlist(playlist_id)
    
    if not pl:
        click.echo(f"✗ 播放列表不存在: {playlist_id}", err=True)
        return
    
    # 获取已完成处理的视频
    videos = playlist_service.get_playlist_videos(playlist_id)
    ready_videos = []
    
    for v in videos:
        if v.output_dir:
            final_video = Path(v.output_dir) / "final.mp4"
            if final_video.exists():
                # 检查是否已上传
                metadata = v.metadata or {}
                if not metadata.get('bilibili_bvid'):
                    ready_videos.append(v)
    
    if not ready_videos:
        click.echo("没有待上传的视频")
        return
    
    if limit:
        ready_videos = ready_videos[:limit]
    
    click.echo(f"\n播放列表: {pl.title}")
    click.echo(f"待上传视频: {len(ready_videos)} 个")
    
    if dry_run:
        click.echo("\n--dry-run 模式，显示待上传列表:")
        for i, v in enumerate(ready_videos, 1):
            click.echo(f"  {i}. [{v.id[:8]}] {v.title[:40]}")
        return
    
    if not click.confirm(f"\n确认上传 {len(ready_videos)} 个视频?"):
        click.echo("已取消")
        return
    
    # 逐个上传
    success_count = 0
    for i, v in enumerate(ready_videos, 1):
        click.echo(f"\n[{i}/{len(ready_videos)}] 上传: {v.title[:40]}...")
        ctx.invoke(upload_video, video_id=v.id, upload_playlist_id=playlist_id, platform=platform, season=season, dry_run=False)
        success_count += 1
    
    click.echo(f"\n完成: 成功上传 {success_count}/{len(ready_videos)} 个视频")


if __name__ == '__main__':
    cli(obj={})
