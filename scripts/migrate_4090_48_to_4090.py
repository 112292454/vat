#!/usr/bin/env python3
"""
一次性迁移脚本：将 4090-48 (NFS) 上的视频文件同步到 4090 (本地)

背景：
  数据库中 76 个视频的 output_dir 指向 /local/gzy/4090-48/vat/data/videos/{id}，
  这是 NFS 挂载路径。最近的处理（翻译、嵌入）都在 4090-48 上执行，产生了较新的文件。
  本脚本将这些文件迁移到本地 /local/gzy/4090/vat/data/videos/{id}。

策略：
  1. 从数据库中读取 output_dir 指向 4090-48 的视频 ID 列表
  2. 对每个视频：
     a. 清空 4090 目录中的旧文件（避免新旧混杂，缺失即可发现）
     b. rsync 从 4090-48 复制到 4090（保留时间戳）
  3. DB migration v5 会在代码重启时自动清空 output_dir 列

注意：
  - 此脚本在 DB migration v5 之前运行（v5 会清空 output_dir，之后无法从 DB 中识别 4090-48 路径）
  - 4090-48 是 NFS，rsync 大文件可能较慢
  - --dry-run 模式仅预览，不实际操作
"""
import sqlite3
import subprocess
import shutil
import sys
import os
from pathlib import Path
from datetime import datetime

# 路径配置
DB_PATH = "/local/gzy/4090/vat/data/database.db"
OLD_BASE = "/local/gzy/4090-48/vat/data/videos"
NEW_BASE = "/local/gzy/4090/vat/data/videos"
OLD_PREFIX = "/local/gzy/4090-48/"

DRY_RUN = "--dry-run" in sys.argv


def get_videos_on_old_path(db_path: str) -> list:
    """从数据库读取 output_dir 指向 4090-48 的视频 ID 列表"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, output_dir FROM videos WHERE output_dir LIKE ?",
        (f"{OLD_PREFIX}%",)
    )
    rows = [(row['id'], row['output_dir']) for row in cursor.fetchall()]
    conn.close()
    return rows


def compare_dirs(src: Path, dst: Path) -> dict:
    """比较两个目录中的文件，返回差异摘要"""
    src_files = {f.name: f for f in src.iterdir() if f.is_file()} if src.exists() else {}
    dst_files = {f.name: f for f in dst.iterdir() if f.is_file()} if dst.exists() else {}
    
    only_src = set(src_files) - set(dst_files)
    only_dst = set(dst_files) - set(src_files)
    common = set(src_files) & set(dst_files)
    
    newer_in_src = []
    same = []
    for name in common:
        src_stat = src_files[name].stat()
        dst_stat = dst_files[name].stat()
        # 比较 mtime 和 size
        if abs(src_stat.st_mtime - dst_stat.st_mtime) < 1 and src_stat.st_size == dst_stat.st_size:
            same.append(name)
        elif src_stat.st_mtime > dst_stat.st_mtime:
            newer_in_src.append(name)
        else:
            # dst 比 src 新——理论上不应发生，标记为异常
            newer_in_src.append(f"{name} (WARNING: dst is newer)")
    
    return {
        "only_src": only_src,
        "only_dst": only_dst,
        "newer_in_src": newer_in_src,
        "same": same,
        "src_count": len(src_files),
        "dst_count": len(dst_files),
    }


def migrate_video(video_id: str, src_dir: Path, dst_dir: Path, dry_run: bool) -> bool:
    """迁移单个视频目录
    
    步骤：
    1. 清空 dst 目录中的所有文件（避免新旧混杂）
    2. rsync 从 src 复制到 dst
    
    Returns:
        True if successful
    """
    if not src_dir.exists():
        print(f"  WARNING: 源目录不存在: {src_dir}")
        return False
    
    # Step 1: 清空 dst 目录
    if dst_dir.exists():
        if dry_run:
            dst_files = list(dst_dir.iterdir())
            print(f"  [DRY-RUN] 将清空 {len(dst_files)} 个文件: {dst_dir}")
        else:
            # 删除目录内所有文件，保留目录本身
            for f in dst_dir.iterdir():
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
    else:
        if dry_run:
            print(f"  [DRY-RUN] 将创建目录: {dst_dir}")
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 2: rsync 从 src 到 dst（保留时间戳）
    cmd = [
        "rsync", "-a",  # archive mode: 保留权限、时间戳等
        "--info=progress2",  # 显示总体进度
        f"{src_dir}/",  # trailing slash: 复制目录内容
        f"{dst_dir}/",
    ]
    
    if dry_run:
        print(f"  [DRY-RUN] rsync {src_dir}/ → {dst_dir}/")
        return True
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: rsync 失败: {result.stderr}")
        return False
    
    return True


def verify_migration(video_id: str, src_dir: Path, dst_dir: Path) -> bool:
    """验证迁移结果：dst 中每个文件与 src 大小一致"""
    if not dst_dir.exists():
        return False
    
    src_files = {f.name: f.stat().st_size for f in src_dir.iterdir() if f.is_file()}
    dst_files = {f.name: f.stat().st_size for f in dst_dir.iterdir() if f.is_file()}
    
    if set(src_files) != set(dst_files):
        missing = set(src_files) - set(dst_files)
        extra = set(dst_files) - set(src_files)
        if missing:
            print(f"  VERIFY FAIL: 缺失文件: {missing}")
        if extra:
            print(f"  VERIFY FAIL: 多余文件: {extra}")
        return False
    
    for name in src_files:
        if src_files[name] != dst_files[name]:
            print(f"  VERIFY FAIL: 大小不一致: {name} (src={src_files[name]}, dst={dst_files[name]})")
            return False
    
    return True


def main():
    print(f"{'='*60}")
    print(f"4090-48 → 4090 视频文件迁移脚本")
    print(f"{'='*60}")
    print(f"数据库: {DB_PATH}")
    print(f"源路径: {OLD_BASE}")
    print(f"目标路径: {NEW_BASE}")
    print(f"模式: {'DRY-RUN（仅预览）' if DRY_RUN else '实际执行'}")
    print()
    
    # Step 1: 读取需要迁移的视频
    videos = get_videos_on_old_path(DB_PATH)
    print(f"需要迁移的视频: {len(videos)} 个")
    print()
    
    if not videos:
        print("无需迁移")
        return
    
    # Step 2: 分析差异
    needs_migration = []
    already_synced = []
    
    for video_id, old_path in videos:
        src_dir = Path(OLD_BASE) / video_id
        dst_dir = Path(NEW_BASE) / video_id
        
        diff = compare_dirs(src_dir, dst_dir)
        has_diff = diff["only_src"] or diff["newer_in_src"]
        
        if has_diff:
            needs_migration.append((video_id, old_path, diff))
        else:
            already_synced.append((video_id, diff))
    
    print(f"需要同步文件的视频: {len(needs_migration)} 个")
    print(f"两边已一致的视频: {len(already_synced)} 个（仍需 clear+rsync 以确保一致性）")
    print()
    
    # 显示需要迁移的详情
    if needs_migration:
        print("--- 有差异的视频 ---")
        for video_id, old_path, diff in needs_migration:
            print(f"  {video_id}:")
            if diff["only_src"]:
                print(f"    仅在 4090-48: {diff['only_src']}")
            if diff["newer_in_src"]:
                print(f"    4090-48 更新: {diff['newer_in_src']}")
        print()
    
    if not DRY_RUN:
        # 确认执行
        total_videos = len(videos)
        answer = input(f"将对 {total_videos} 个视频执行: 清空4090目录 → rsync从4090-48。继续? [y/N] ")
        if answer.lower() != 'y':
            print("已取消")
            return
    
    # Step 3: 执行迁移（所有 76 个都 clear+rsync，确保一致性）
    success = 0
    failed = 0
    
    for i, (video_id, old_path) in enumerate(videos, 1):
        src_dir = Path(OLD_BASE) / video_id
        dst_dir = Path(NEW_BASE) / video_id
        
        print(f"[{i}/{len(videos)}] {video_id}")
        
        ok = migrate_video(video_id, src_dir, dst_dir, DRY_RUN)
        
        if ok and not DRY_RUN:
            # 验证
            if verify_migration(video_id, src_dir, dst_dir):
                print(f"  ✓ 迁移完成并验证通过")
                success += 1
            else:
                print(f"  ✗ 迁移完成但验证失败")
                failed += 1
        elif ok:
            success += 1
        else:
            failed += 1
    
    print()
    print(f"{'='*60}")
    print(f"迁移结果: 成功 {success}, 失败 {failed}")
    if not DRY_RUN:
        print(f"\n下一步: 重启应用，DB migration v5 将自动清空 output_dir 列。")
        print(f"之后所有视频的 output_dir 将由 config.storage.output_dir / video_id 计算。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
