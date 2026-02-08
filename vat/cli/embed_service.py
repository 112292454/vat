"""
异步嵌字服务命令行工具

用于启动、管理独立的嵌字服务
"""
import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional
import click

from ..embedder.async_embedder import init_global_queue, get_global_queue
from ..config import Config


@click.group()
def embed_service():
    """异步嵌字服务管理"""
    pass


@embed_service.command()
@click.option(
    '--config',
    '-c',
    type=click.Path(exists=True),
    default='config/default.yaml',
    help='配置文件路径'
)
@click.option(
    '--persist-file',
    type=click.Path(),
    default='data/embed_queue.json',
    help='任务持久化文件路径'
)
def start(config: str, persist_file: str):
    """启动异步嵌字服务"""
    click.echo("正在启动异步嵌字服务...")
    
    # 加载配置
    cfg = Config.from_yaml(config)
    
    # 初始化队列
    queue = init_global_queue(
        gpu_devices=cfg.concurrency.gpu_devices,
        max_concurrent_per_gpu=cfg.concurrency.max_concurrent_per_gpu,
        persist_file=Path(persist_file).expanduser()
    )
    
    # 运行服务
    asyncio.run(_run_service(queue))


async def _run_service(queue):
    """运行服务主循环"""
    # 启动队列
    await queue.start()
    
    # 注册信号处理
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        click.echo("\n收到停止信号，正在优雅退出...")
        asyncio.create_task(queue.stop())
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    click.echo("异步嵌字服务已启动")
    click.echo(f"GPU设备: {queue.gpu_devices}")
    click.echo(f"每GPU并发数: {queue.max_concurrent_per_gpu}")
    click.echo("按 Ctrl+C 停止服务")
    
    # 定期打印统计信息
    try:
        while queue.running:
            await asyncio.sleep(30)
            stats = queue.get_stats()
            click.echo(
                f"[统计] 提交: {stats['total_submitted']}, "
                f"待处理: {stats['pending_count']}, "
                f"处理中: {stats['processing_count']}, "
                f"已完成: {stats['total_completed']}, "
                f"失败: {stats['total_failed']}, "
                f"平均耗时: {stats['avg_processing_time']:.2f}秒"
            )
    except asyncio.CancelledError:
        pass
    
    click.echo("异步嵌字服务已停止")


@embed_service.command()
@click.option(
    '--config',
    '-c',
    type=click.Path(exists=True),
    default='config/default.yaml',
    help='配置文件路径'
)
def status(config: str):
    """查看服务状态"""
    # TODO: 实现通过IPC或API查询服务状态
    click.echo("状态查询功能待实现")


if __name__ == '__main__':
    embed_service()
