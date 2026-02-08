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
from .logger import setup_logger


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
    """获取日志器"""
    global LOGGER
    if LOGGER is None:
        config = get_config()
        LOGGER = setup_logger(
            level=config.logging.level,
            log_file=config.logging.file,
            log_format=config.logging.format
        )
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
    db = Database(config.storage.database_path)
    
    # 收集URLs
    urls = list(url)
    
    # 从播放列表获取
    if playlist:
        logger.info(f"获取播放列表: {playlist}")
        downloader = YouTubeDownloader(
            proxy=config.proxy.get_proxy(),
            video_format=config.downloader.youtube.format
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
    db = Database(config.storage.database_path)
    
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
    db = Database(config.storage.database_path)
    
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
    db = Database(config.storage.database_path)
    
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
    db = Database(config.storage.database_path)
    
    # 设置GPU
    if gpus:
        config.concurrency.gpu_devices = [int(g.strip()) for g in gpus.split(',')]
    
    # 收集URLs
    urls = list(url)
    
    if playlist:
        logger.info(f"获取播放列表: {playlist}")
        downloader = YouTubeDownloader(
            proxy=config.proxy.get_proxy(),
            video_format=config.downloader.youtube.format
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
@click.option('--video-id', '-v', help='视频ID')
@click.option('--all-completed', is_flag=True, help='上传所有已完成的视频')
@click.option('--title', help='视频标题')
@click.option('--desc', help='视频描述')
@click.option('--tags', help='标签（逗号分隔）')
@click.pass_context
def upload(ctx, video_id, all_completed, title, desc, tags):
    """上传视频到B站"""
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    
    click.echo("上传功能尚未完全实现")
    # TODO: 实现上传逻辑


@cli.command()
@click.option('--video-id', '-v', help='查看特定视频的状态')
@click.option('--failed', 'filter_failed', is_flag=True, help='仅显示失败的任务')
@click.option('--pending', 'filter_pending', is_flag=True, help='仅显示待处理的任务')
@click.pass_context
def status(ctx, video_id, filter_failed, filter_pending):
    """查看处理状态"""
    config = get_config(ctx.obj.get('config_path'))
    db = Database(config.storage.database_path)
    
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
@click.option('--all', 'clean_all', is_flag=True, help='清理所有视频的处理产物')
@click.option('--yes', '-y', is_flag=True, help='跳过确认')
@click.pass_context
def clean(ctx, video_id, clean_all, yes):
    """清理处理产物（保留原始下载文件）"""
    from vat.utils.file_ops import delete_processed_files
    
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path)
    
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
    
    click.echo(f"将清理 {len(videos)} 个视频的处理产物（保留原始下载文件）")
    if not yes and not click.confirm("确认?"):
        click.echo("已取消")
        return
    
    total_deleted = 0
    for v in videos:
        if v.output_dir:
            output_dir = Path(v.output_dir)
            if output_dir.exists():
                deleted = delete_processed_files(output_dir)
                if deleted:
                    total_deleted += len(deleted)
                    click.echo(f"  {v.id[:12]}: 删除 {len(deleted)} 个文件")
    
    click.echo(f"\n清理完成，共删除 {total_deleted} 个处理产物")


@cli.command('delete')
@click.option('--video-id', '-v', multiple=True, help='视频ID（可多次指定）')
@click.option('--delete-files', is_flag=True, help='同时删除所有文件（包括原始下载）')
@click.option('--yes', '-y', is_flag=True, help='跳过确认')
@click.pass_context
def delete_video(ctx, video_id, delete_files, yes):
    """删除视频记录（默认保留原始下载文件）"""
    import shutil
    from vat.utils.file_ops import delete_processed_files
    
    config = get_config(ctx.obj.get('config_path'))
    logger = get_logger()
    db = Database(config.storage.database_path)
    
    if not video_id:
        click.echo("错误: 请指定 --video-id", err=True)
        return
    
    videos = []
    for vid in video_id:
        v = db.get_video(vid)
        if v:
            videos.append(v)
        else:
            click.echo(f"警告: 视频不存在: {vid}", err=True)
    
    if not videos:
        click.echo("没有待删除的视频")
        return
    
    action = "删除记录+所有文件" if delete_files else "删除记录+处理产物（保留原始下载）"
    click.echo(f"将{action}，共 {len(videos)} 个视频")
    if not yes and not click.confirm("确认?"):
        click.echo("已取消")
        return
    
    for v in videos:
        if v.output_dir:
            output_dir = Path(v.output_dir)
            if output_dir.exists():
                if delete_files:
                    shutil.rmtree(output_dir)
                else:
                    delete_processed_files(output_dir)
        
        db.delete_video(v.id)
        title = v.title[:30] if v.title else v.id
        click.echo(f"  ✓ 已删除: {title}")
    
    click.echo(f"\n删除完成")


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
              help='要执行的阶段（逗号分隔）: download,whisper,split,optimize,translate,embed,upload 或阶段组 asr,translate 或 all')
@click.option('--gpu', '-g', default='auto',
              help='GPU 设备: auto（自动选择）, cpu, cuda:0, cuda:1 等')
@click.option('--force', '-f', is_flag=True, help='强制重新处理（即使已完成）')
@click.option('--dry-run', is_flag=True, help='仅显示将要执行的操作，不实际执行')
@click.option('--concurrency', '-c', default=1, type=int, help='并发处理的视频数量（默认1，即串行）')
@click.pass_context
def process(ctx, video_id, process_all, playlist, stages, gpu, force, dry_run, concurrency):
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
    db = Database(config.storage.database_path)
    
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
        # 从 Playlist 获取视频
        playlist_service = PlaylistService(db)
        pl = playlist_service.get_playlist(playlist)
        if not pl:
            click.echo(f"错误: Playlist 不存在: {playlist}", err=True)
            return
        
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
    
    # 显示执行计划
    click.echo(f"\n执行计划:")
    click.echo(f"  视频数量: {len(video_ids)}")
    click.echo(f"  执行阶段: {', '.join(s.value for s in target_steps)}")
    click.echo(f"  GPU 设备: {gpu}")
    click.echo(f"  强制模式: {'是' if force else '否'}")
    click.echo(f"  并发数量: {concurrency}")
    
    if dry_run:
        click.echo(f"\n[DRY-RUN] 以下视频将被处理:")
        for vid in video_ids[:10]:
            video = db.get_video(vid)
            title = video.title[:40] if video and video.title else vid
            click.echo(f"  - {title}")
        if len(video_ids) > 10:
            click.echo(f"  ... 还有 {len(video_ids) - 10} 个视频")
        
        # 生成等价 CLI 命令
        cli_cmd = _generate_process_cli(video_ids, stages, gpu, force)
        click.echo(f"\n等价 CLI 命令:")
        click.echo(f"  {cli_cmd}")
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
    
    def process_one_video(args):
        """处理单个视频（可在线程池中并发调用）"""
        idx, vid = args
        video = db.get_video(vid)
        if not video:
            logger.warning(f"视频不存在: {vid}")
            return vid, False, "视频不存在"
        
        title = video.title[:30] if video.title else vid
        click.echo(f"\n[{idx + 1}/{total}] 处理: {title}")
        
        try:
            processor = VideoProcessor(
                video_id=vid,
                config=config,
                gpu_id=gpu_id,
                force=force,
                video_index=idx,
                total_videos=total
            )
            processor.process(steps=step_names)
            click.echo(f"  ✓ [{idx + 1}/{total}] 完成: {title}")
            return vid, True, None
        except Exception as e:
            logger.error(f"处理失败: {vid} - {e}")
            click.echo(f"  ✗ [{idx + 1}/{total}] 失败: {title} - {e}")
            return vid, False, str(e)
    
    # 执行处理
    click.echo(f"\n开始处理 {total} 个视频（并发: {concurrency}）...")
    
    if concurrency <= 1:
        # 串行处理
        for idx, vid in enumerate(video_ids):
            process_one_video((idx, vid))
    else:
        # 并发处理
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(process_one_video, (idx, vid)): vid 
                      for idx, vid in enumerate(video_ids)}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    vid = futures[future]
                    logger.error(f"并发处理异常: {vid} - {e}")
    
    click.echo(f"\n处理完成")


def _generate_process_cli(video_ids: List[str], stages: str, gpu: str, force: bool) -> str:
    """生成等价的 CLI 命令"""
    parts = ["vat", "process"]
    
    for vid in video_ids[:3]:
        parts.append(f"-v {vid}")
    
    if len(video_ids) > 3:
        parts.append(f"# ... 还有 {len(video_ids) - 3} 个视频")
    
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
    db = Database(config.storage.database_path)
    
    downloader = YouTubeDownloader(
        proxy=config.proxy.get_proxy(),
        video_format=config.downloader.youtube.format
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
    db = Database(config.storage.database_path)
    
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
    db = Database(config.storage.database_path)
    
    pl = db.get_playlist(playlist_id)
    if not pl:
        click.echo(f"错误: Playlist 不存在: {playlist_id}", err=True)
        return
    
    downloader = YouTubeDownloader(
        proxy=config.proxy.get_proxy(),
        video_format=config.downloader.youtube.format
    )
    
    playlist_service = PlaylistService(db, downloader)
    
    try:
        result = playlist_service.sync_playlist(
            pl.source_url,
            auto_add_videos=True,
            progress_callback=lambda msg: click.echo(msg)
        )
        
        click.echo(f"\n✓ 同步完成")
        click.echo(f"  新增视频: {result.new_count}")
        click.echo(f"  已存在: {result.existing_count}")
        
    except Exception as e:
        click.echo(f"✗ 同步失败: {e}", err=True)


@playlist.command('show')
@click.argument('playlist_id')
@click.pass_context
def playlist_show(ctx, playlist_id):
    """显示 Playlist 详情"""
    config = get_config(ctx.obj.get('config_path'))
    db = Database(config.storage.database_path)
    
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
    db = Database(config.storage.database_path)
    
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
# 上传命令
# =============================================================================

@cli.command()
@click.argument('video_id')
@click.option('--platform', '-p', default='bilibili', help='上传平台 (目前仅支持 bilibili)')
@click.option('--season', '-s', type=int, help='添加到合集ID (上传后自动添加)')
@click.option('--dry-run', is_flag=True, help='仅预览，不实际上传')
@click.pass_context
def upload(ctx, video_id, platform, season, dry_run):
    """上传视频到指定平台
    
    VIDEO_ID: 视频ID
    """
    config = get_config(ctx.obj.get('config_path'))
    db = Database(config.storage.database_path)
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
    playlist_info = None
    if video.playlist_id:
        playlist_service = PlaylistService(db)
        pl = playlist_service.get_playlist(video.playlist_id)
        if pl:
            playlist_info = {
                'name': pl.title,
                'id': pl.id,
                'index': video.playlist_index or 0,
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
            
            # 添加到合集
            if season:
                click.echo(f"\n添加到合集 {season}...")
                aid = uploader.bvid_to_aid(result.bvid)
                if aid:
                    if uploader.add_to_season(aid, season):
                        click.echo(f"✓ 已添加到合集")
                    else:
                        click.echo(f"⚠ 添加到合集失败", err=True)
                else:
                    click.echo(f"⚠ 无法获取AV号，跳过添加合集", err=True)
            
            # 更新数据库（合并到现有 metadata）
            video_obj = db.get_video(video_id)
            updated_metadata = dict(video_obj.metadata) if video_obj and video_obj.metadata else {}
            updated_metadata.update({
                'bilibili_bvid': result.bvid,
                'bilibili_url': f"https://www.bilibili.com/video/{result.bvid}",
                'uploaded_at': datetime.now().isoformat(),
                'bilibili_season_id': season,
            })
            db.update_video(video_id, metadata=updated_metadata)
        else:
            click.echo(f"✗ 上传失败: {result.error}", err=True)
    else:
        click.echo(f"✗ 不支持的平台: {platform}", err=True)


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
    click.echo("使用示例: vat upload VIDEO_ID --season SEASON_ID")


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


@cli.command('upload-playlist')
@click.argument('playlist_id')
@click.option('--platform', '-p', default='bilibili', help='上传平台')
@click.option('--season', '-s', type=int, help='添加到合集ID')
@click.option('--limit', '-n', type=int, help='最大上传数量')
@click.option('--dry-run', is_flag=True, help='仅预览，不实际上传')
@click.pass_context
def upload_playlist(ctx, playlist_id, platform, season, limit, dry_run):
    """批量上传播放列表中的视频
    
    PLAYLIST_ID: 播放列表ID
    """
    config = get_config(ctx.obj.get('config_path'))
    db = Database(config.storage.database_path)
    
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
        ctx.invoke(upload, video_id=v.id, platform=platform, season=season, dry_run=False)
        success_count += 1
    
    click.echo(f"\n完成: 成功上传 {success_count}/{len(ready_videos)} 个视频")


if __name__ == '__main__':
    cli(obj={})
