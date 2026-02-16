#!/usr/bin/env python3
"""补齐缺失的视频封面（统一保存为 thumbnail.jpg）

扫描视频目录，对缺少本地封面的视频尝试下载：
1. 优先使用 metadata 中的 thumbnail URL
2. 对标准 YouTube ID，直接构造 ytimg.com URL（maxres → hq → mq 降级）
3. 对哈希 ID，从 source_url 提取 YouTube ID 后构造 URL

下载后自动转为 JPG 格式保存为 thumbnail.jpg。

用法:
    python scripts/fix_thumbnails.py                    # 扫描并修复所有
    python scripts/fix_thumbnails.py --dry-run          # 仅报告缺失
    python scripts/fix_thumbnails.py --convert-existing # 将现有 webp/png 封面转为 jpg
    python scripts/fix_thumbnails.py -v VIDEO_ID        # 修复指定视频
"""
import argparse
import io
import json
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

# ============ 配置 ============
DB_PATH = "/local/gzy/4090/vat/data/database.db"
BASE_DIR = "/local/gzy/4090/vat/data/videos"
PROXY = "http://localhost:7990"
TIMEOUT = 20
MAX_WORKERS = 8

COVER_NAMES = ["thumbnail", "cover"]
COVER_EXTS = ["jpg", "jpeg", "png", "webp"]
YT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
YT_URL_PATTERN = re.compile(r"[?&]v=([A-Za-z0-9_-]{11})")


def get_proxies():
    return {"http": PROXY, "https": PROXY} if PROXY else None


def find_local_cover(out_dir: Path) -> Path | None:
    """查找本地封面文件，返回路径或 None"""
    for name in COVER_NAMES:
        for ext in COVER_EXTS:
            p = out_dir / f"{name}.{ext}"
            if p.exists() and p.stat().st_size > 500:
                return p
    return None


def save_as_jpg(image_bytes: bytes, target_path: Path) -> int:
    """将图片字节数据转为 JPG 并保存，返回文件大小"""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    img.save(target_path, "JPEG", quality=90)
    return target_path.stat().st_size


def convert_cover_to_jpg(cover_path: Path) -> bool:
    """将现有非 JPG 封面转为 thumbnail.jpg，删除旧文件"""
    out_dir = cover_path.parent
    target = out_dir / "thumbnail.jpg"

    if cover_path == target:
        return False  # 已是 jpg
    if cover_path.suffix.lower() in (".jpg", ".jpeg") and cover_path.name.startswith("thumbnail"):
        # 只需重命名
        cover_path.rename(target)
        return True

    try:
        img = Image.open(cover_path)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.save(target, "JPEG", quality=90)
        # 删除旧文件（不同路径时）
        if cover_path != target and cover_path.exists():
            cover_path.unlink()
        return True
    except Exception as e:
        print(f"  转换失败 {cover_path}: {e}")
        return False


def get_thumbnail_urls(video_id, source_url, metadata):
    """根据视频信息构建候选 thumbnail URL 列表"""
    urls = []

    # 1. metadata 中的 URL
    thumb = metadata.get("thumbnail", "")
    if thumb:
        urls.append(thumb)
        if "maxresdefault" in thumb:
            urls.append(thumb.replace("maxresdefault", "hqdefault"))
            urls.append(thumb.replace("maxresdefault", "mqdefault"))
        return urls

    # 2. 从 video ID 或 source_url 提取 YouTube ID
    yt_id = None
    if YT_ID_PATTERN.match(video_id):
        yt_id = video_id
    elif source_url:
        m = YT_URL_PATTERN.search(source_url)
        if m:
            yt_id = m.group(1)

    if yt_id:
        for res in ["maxresdefault", "hqdefault", "mqdefault"]:
            urls.append(f"https://i.ytimg.com/vi/{yt_id}/{res}.jpg")
    return urls


def download_one(video_id, source_url, metadata):
    """下载单个视频封面，统一保存为 thumbnail.jpg"""
    out_dir = Path(BASE_DIR) / video_id
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    candidate_urls = get_thumbnail_urls(video_id, source_url, metadata)
    if not candidate_urls:
        return video_id, "no_url", ""

    proxies = get_proxies()
    for url in candidate_urls:
        try:
            resp = requests.get(url, timeout=TIMEOUT, proxies=proxies)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            if len(resp.content) < 500:
                continue

            target = out_dir / "thumbnail.jpg"
            size = save_as_jpg(resp.content, target)
            return video_id, "ok", f"thumbnail.jpg ({size // 1024}KB)"
        except Exception:
            continue

    return video_id, "fail", "all URLs failed"


def main():
    parser = argparse.ArgumentParser(description="补齐缺失的视频封面")
    parser.add_argument("-v", "--video-id", nargs="*", help="指定视频 ID")
    parser.add_argument("--dry-run", action="store_true", help="仅报告缺失，不下载")
    parser.add_argument("--convert-existing", action="store_true",
                        help="将现有 webp/png 封面统一转为 thumbnail.jpg")
    parser.add_argument("-w", "--workers", type=int, default=MAX_WORKERS, help="并发线程数")
    args = parser.parse_args()

    base_dir = Path(BASE_DIR)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 获取视频列表
    if args.video_id:
        placeholders = ",".join("?" * len(args.video_id))
        cursor.execute(f"SELECT id, source_url, metadata FROM videos WHERE id IN ({placeholders})",
                       args.video_id)
    else:
        cursor.execute("SELECT id, source_url, metadata FROM videos")

    rows = cursor.fetchall()
    conn.close()
    print(f"扫描 {len(rows)} 个视频")

    # ---- 模式1：转换现有封面为 jpg ----
    if args.convert_existing:
        converted = 0
        skipped = 0
        for row in rows:
            out_dir = base_dir / row["id"]
            cover = find_local_cover(out_dir)
            if not cover:
                continue
            # 已是 thumbnail.jpg 则跳过
            if cover.name == "thumbnail.jpg":
                skipped += 1
                continue
            if convert_cover_to_jpg(cover):
                converted += 1
            else:
                skipped += 1
        print(f"转换完成: 转换={converted}, 已是jpg={skipped}")
        return

    # ---- 模式2：下载缺失封面 ----
    missing = []
    for row in rows:
        vid = row["id"]
        meta = json.loads(row["metadata"] or "{}")
        if meta.get("unavailable"):
            continue
        out_dir = base_dir / vid
        if find_local_cover(out_dir):
            continue
        missing.append((vid, row["source_url"] or "", meta))

    print(f"缺少封面: {len(missing)} 个")

    if not missing:
        print("所有视频都有本地封面 ✓")
        return

    if args.dry_run:
        for vid, src, meta in missing:
            thumb = meta.get("thumbnail", "")
            print(f"  缺失: {vid}  URL={'有' if thumb else '无'}")
        return

    ok = fail = no_url = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, vid, src, meta): vid
                   for vid, src, meta in missing}
        for i, future in enumerate(as_completed(futures), 1):
            vid, status, msg = future.result()
            if status == "ok":
                ok += 1
            elif status == "no_url":
                no_url += 1
                print(f"  跳过 {vid}: 无可用 URL")
            else:
                fail += 1
                print(f"  失败 {vid}: {msg}")

            if i % 50 == 0 or i == len(missing):
                print(f"  [{i}/{len(missing)}] 成功={ok} 失败={fail} 无URL={no_url}")

    print(f"\n完成: 下载={ok}, 失败={fail}, 无URL={no_url}")


if __name__ == "__main__":
    main()
