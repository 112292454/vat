#!/usr/bin/env python3
"""
B站上传测试脚本

使用方法:
    python scripts/bilibili_upload_test.py [video_path]

如果不指定视频路径，会使用数据库中已完成的视频进行测试。
"""
import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from biliup.plugins.bili_webup import BiliBili, Data


def load_cookie(cookie_file: Path) -> dict:
    """加载并解析 cookie 文件（兼容 social-auto-upload 格式）"""
    with open(cookie_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取关键 cookie
    keys_to_extract = ["SESSDATA", "bili_jct", "DedeUserID__ckMd5", "DedeUserID", "access_token"]
    extracted = {}
    
    # 从 cookie_info 中提取
    if 'cookie_info' in data and 'cookies' in data['cookie_info']:
        for cookie in data['cookie_info']['cookies']:
            if cookie.get('name') in keys_to_extract:
                extracted[cookie['name']] = cookie['value']
    
    # 提取 access_token
    if 'token_info' in data and 'access_token' in data['token_info']:
        extracted['access_token'] = data['token_info']['access_token']
    
    # 如果是简单格式，直接使用
    if not extracted:
        for key in keys_to_extract:
            if key in data:
                extracted[key] = data[key]
    
    return extracted


def find_test_video() -> tuple:
    """从数据库中查找一个已完成的视频用于测试"""
    from vat.config import load_config
    from vat.database import Database
    
    config = load_config()
    db = Database(config.storage.database_path)
    
    # 查找 test playlist 中的视频
    test_playlist_id = "PLb-FxUvMeZO1jkvW8Ybzd8rCH2C1aoIGG"
    video_ids = db.get_playlist_video_ids(test_playlist_id)
    
    for vid in video_ids:
        video = db.get_video(vid)
        if video and video.output_dir:
            output_dir = Path(video.output_dir)
            final_video = output_dir / "final.mp4"
            if final_video.exists():
                metadata = video.metadata or {}
                translated = metadata.get('translated', {})
                
                # 查找封面图片（优先使用下载的缩略图）
                cover_path = None
                for cover_name in ['thumbnail.jpg', 'thumbnail.png', 'cover.jpg', 'cover.png']:
                    potential_cover = output_dir / cover_name
                    if potential_cover.exists():
                        cover_path = potential_cover
                        break
                
                return final_video, {
                    'title': translated.get('title', video.title or vid),
                    'desc': translated.get('description', ''),
                    'tags': translated.get('tags', ['VTuber', '日本']),
                    'tid': translated.get('recommended_tid', 21),
                    'source': video.source_url,
                    'cover': cover_path,
                }
    
    return None, None


def main():
    parser = argparse.ArgumentParser(description='B站上传测试')
    parser.add_argument('video_path', nargs='?', help='视频文件路径')
    parser.add_argument('--title', help='视频标题')
    parser.add_argument('--tid', type=int, default=21, help='分区ID (默认: 21 日常)')
    parser.add_argument('--copyright', type=int, choices=[1, 2], default=2, 
                        help='类型: 1=自制, 2=转载 (默认: 2)')
    parser.add_argument('--cover', help='封面图片路径')
    parser.add_argument('--dry-run', action='store_true', help='仅验证，不实际上传')
    args = parser.parse_args()
    
    # Cookie 文件路径
    cookie_file = project_root / "cookies" / "bilibili" / "account.json"
    
    if not cookie_file.exists():
        print(f"✗ Cookie 文件不存在: {cookie_file}")
        print()
        print("请先运行登录脚本获取 Cookie:")
        print("  python scripts/bilibili_login.py")
        sys.exit(1)
    
    # 加载 cookie
    print(f"加载 Cookie: {cookie_file}")
    cookie_data = load_cookie(cookie_file)
    
    required_keys = ["SESSDATA", "bili_jct", "DedeUserID"]
    missing = [k for k in required_keys if k not in cookie_data]
    if missing:
        print(f"✗ Cookie 缺少必要字段: {missing}")
        sys.exit(1)
    
    print(f"✓ Cookie 加载成功")
    print()
    
    # 确定视频文件和元数据
    if args.video_path:
        video_path = Path(args.video_path)
        metadata = {
            'title': args.title or video_path.stem+"[自动上传测试]",
            'desc': '测试上传',
            'tags': ['测试'],
            'tid': args.tid,
            'source': '',
            'cover': Path(args.cover) if args.cover else None,
        }
    else:
        print("未指定视频，从数据库查找测试视频...")
        video_path, metadata = find_test_video()
        
        if not video_path:
            print("✗ 未找到可用的测试视频")
            print()
            print("请指定视频路径:")
            print("  python scripts/bilibili_upload_test.py /path/to/video.mp4")
            sys.exit(1)
    
    # 命令行参数覆盖
    if args.cover:
        metadata['cover'] = Path(args.cover)
    
    copyright_type = args.copyright
    
    if not video_path.exists():
        print(f"✗ 视频文件不存在: {video_path}")
        sys.exit(1)
    
    print(f"视频文件: {video_path}")
    print(f"文件大小: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("上传信息:")
    print(f"  标题: {metadata['title']}")
    print(f"  类型: {'自制' if copyright_type == 1 else '转载'}")
    print(f"  分区: {metadata['tid']}")
    print(f"  标签: {metadata['tags']}")
    if metadata.get('source') and copyright_type == 2:
        print(f"  来源: {metadata['source']}")
    if metadata.get('cover'):
        print(f"  封面: {metadata['cover']}")
    print()
    
    if args.dry_run:
        print("--dry-run 模式，跳过实际上传")
        print("✓ 验证通过")
        return
    
    # 确认上传
    confirm = input("确认上传? (y/N): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    print()
    print("开始上传...")
    
    # 准备上传数据
    data = Data()
    data.copyright = copyright_type
    data.title = metadata['title'][:80]  # B站标题限制80字符
    data.desc = metadata.get('desc', '')[:2000]  # 描述限制
    data.tid = metadata['tid']
    data.set_tag(metadata['tags'][:12])  # 标签限制12个
    
    # 转载来源
    if copyright_type == 2 and metadata.get('source'):
        data.source = metadata['source']
    
    data.dtime = 0  # 立即发布
    
    try:
        with BiliBili(data) as bili:
            # 登录
            bili.login_by_cookies(cookie_data)
            if 'access_token' in cookie_data:
                bili.access_token = cookie_data['access_token']
            
            # 验证登录
            try:
                info = bili.myinfo()
                print(f"登录账号: {info.get('uname', 'unknown')}")
            except Exception as e:
                print(f"⚠ 无法获取用户信息: {e}")
            
            # 上传封面
            cover_path = metadata.get('cover')
            if cover_path and Path(cover_path).exists():
                print()
                print(f"上传封面: {cover_path}")
                try:
                    cover_url = bili.cover_up(str(cover_path))
                    data.cover = cover_url.replace('http:', '')
                    print("✓ 封面上传成功")
                except Exception as e:
                    print(f"⚠ 封面上传失败: {e}")
            
            # 上传视频
            print()
            print("上传视频文件...")
            video_part = bili.upload_file(str(video_path), lines='AUTO', tasks=3)
            video_part['title'] = 'P1'
            data.append(video_part)
            
            print("提交视频（使用Web端API）...")
            ret = bili.submit_web()  # 使用 Web 端 API，TV 端已停用
            
            if ret.get('code') == 0:
                bvid = ret.get('data', {}).get('bvid', '')
                aid = ret.get('data', {}).get('aid', '')
                print()
                print("=" * 50)
                print("✓ 上传成功!")
                print(f"  BV号: {bvid}")
                print(f"  AV号: {aid}")
                print(f"  链接: https://www.bilibili.com/video/{bvid}")
                print("=" * 50)
            else:
                print()
                print(f"✗ 上传失败: {ret.get('message', '未知错误')}")
                print(f"  错误码: {ret.get('code')}")
                print(f"  完整响应: {ret}")
                
    except Exception as e:
        print(f"✗ 上传异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
