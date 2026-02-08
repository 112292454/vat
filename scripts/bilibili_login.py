#!/usr/bin/env python3
"""
B站登录脚本 - 通过二维码获取 Cookie

使用方法:
    python scripts/bilibili_login.py

执行后会显示二维码链接，用手机 B 站 APP 扫码登录。
登录成功后，Cookie 会保存到 cookies/bilibili/ 目录。
"""
import sys
from pathlib import Path
import json
import time

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import stream_gears


def main():
    # Cookie 保存目录
    cookie_dir = project_root / "cookies" / "bilibili"
    cookie_dir.mkdir(parents=True, exist_ok=True)
    cookie_file = cookie_dir / "account.json"
    
    print("=" * 50)
    print("B站二维码登录")
    print("=" * 50)
    print()
    print(f"Cookie 将保存到: {cookie_file}")
    print()
    
    # 获取二维码（使用 stream_gears，proxy=None 表示不使用代理）
    print("获取二维码...")
    qr_str = stream_gears.get_qrcode(None)
    qr_data = json.loads(qr_str)
    
    if qr_data.get('code') != 0:
        print(f"✗ 获取二维码失败: {qr_data}")
        return
    
    url = qr_data['data']['url']
    auth_code = qr_data['data']['auth_code']
    
    print()
    print("请用手机 B 站 APP 扫描以下链接的二维码:")
    print()
    print(f"  {url}")
    print()
    print("或者在浏览器中打开上述链接，然后用手机扫描页面上的二维码")
    print()
    print("等待登录（请在浏览器中完成登录，支持扫码或短信验证）...")
    print()
    
    try:
        # login_by_qrcode 会阻塞直到登录成功或超时
        result_str = stream_gears.login_by_qrcode(qr_str, None)
        result = json.loads(result_str)
        
        # 检查是否有 cookie_info（登录成功的标志）
        if 'cookie_info' in result and result['cookie_info'].get('cookies'):
            print("✓ 登录成功!")
            
            # 获取用户信息
            if 'token_info' in result:
                mid = result['token_info'].get('mid', 'N/A')
                print(f"  UID: {mid}")
            
            # 保存完整的登录响应（包含 cookie_info 和 token_info）
            with open(cookie_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print()
            print(f"✓ Cookie 已保存到: {cookie_file}")
            print()
            print("现在可以运行上传测试:")
            print("  python scripts/bilibili_upload_test.py --dry-run")
        else:
            print(f"✗ 登录失败: {result}")
            
    except Exception as e:
        print(f"✗ 登录异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
