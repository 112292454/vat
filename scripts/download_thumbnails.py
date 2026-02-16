#!/usr/bin/env python3
"""紧急脚本：批量下载所有视频的 thumbnail 到本地 output_dir/thumbnail.jpg"""
import sqlite3
import requests
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

DB_PATH = "/local/gzy/4090/vat/data/database.db"
BASE_DIR = "/local/gzy/4090/vat/data/videos"
PROXY = "http://localhost:7990"
TIMEOUT = 20
MAX_WORKERS = 8


def get_videos_with_thumbnails():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, metadata FROM videos WHERE metadata IS NOT NULL")
    results = []
    for row in cursor.fetchall():
        meta = json.loads(row['metadata']) if row['metadata'] else {}
        thumb_url = meta.get('thumbnail', '')
        if thumb_url and thumb_url.startswith('http'):
            results.append((row['id'], thumb_url))
    conn.close()
    return results


def download_one(video_id, thumb_url):
    output_dir = Path(BASE_DIR) / video_id
    target = output_dir / "thumbnail.jpg"
    
    if target.exists() and target.stat().st_size > 1000:
        return video_id, "skip", "already exists"
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        resp = requests.get(thumb_url, timeout=TIMEOUT, proxies={"http": PROXY, "https": PROXY})
        resp.raise_for_status()
        
        # 确定扩展名
        ct = resp.headers.get('content-type', '')
        ext = '.jpg'
        if 'png' in ct:
            ext = '.png'
        elif 'webp' in ct:
            ext = '.webp'
        
        target = output_dir / f"thumbnail{ext}"
        target.write_bytes(resp.content)
        return video_id, "ok", f"{len(resp.content)//1024}KB"
    except Exception as e:
        return video_id, "fail", str(e)


def main():
    videos = get_videos_with_thumbnails()
    print(f"共 {len(videos)} 个视频有 thumbnail URL")
    
    # 检查已存在的
    existing = 0
    for vid, _ in videos:
        d = Path(BASE_DIR) / vid
        if any((d / f"thumbnail{ext}").exists() for ext in ['.jpg', '.png', '.webp']):
            existing += 1
    print(f"已有本地封面: {existing}, 需下载: {len(videos) - existing}")
    
    ok = 0
    skip = 0
    fail = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_one, vid, url): vid for vid, url in videos}
        for i, future in enumerate(as_completed(futures), 1):
            vid, status, msg = future.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"  FAIL [{vid}]: {msg}")
            
            if i % 100 == 0 or i == len(videos):
                print(f"[{i}/{len(videos)}] ok={ok} skip={skip} fail={fail}")
    
    print(f"\n完成: 下载={ok}, 已存在={skip}, 失败={fail}")


if __name__ == "__main__":
    main()
