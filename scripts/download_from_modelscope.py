"""
从 ModelScope 下载 fuzhen/emind-0.5b 模型。

用法：
  1. 设置环境变量: export MODELSCOPE_API_TOKEN=ms-30159468-0600-4910-96fe-f317c2abe100
  2. 运行: python scripts/download_from_modelscope.py
"""
import os
import sys

from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

MODEL_ID = "fuzhen/emind-0.5b"
OUTPUT_DIR = "checkpoints/emind-0.5b-merged"
TOKEN = "ms-30159468-0600-4910-96fe-f317c2abe100"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 先登录
    api = HubApi()
    api.login(TOKEN)

    # 验证模型是否存在
    try:
        info = api.get_model(model_id=MODEL_ID)
        print(f"[OK] Model found: {MODEL_ID}")
        print(f"     Files size: {info.get('StorageSize', 'unknown')} bytes")
        print(f"     SDK Version: {info.get('Revision', 'unknown')}")
    except Exception as e:
        print(f"[ERROR] Model not found or not accessible: {e}")

    # 下载
    print(f"\nDownloading {MODEL_ID} to {OUTPUT_DIR} ...")
    try:
        local_dir = snapshot_download(
            MODEL_ID,
            local_dir=OUTPUT_DIR,
        )
        print(f"\n[OK] Download complete: {local_dir}")
        for f in os.listdir(local_dir):
            size = os.path.getsize(os.path.join(local_dir, f))
            print(f"  {f:30s} {size/1024/1024:.1f} MB")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nTrying alternative method...")
        try:
            files = api.get_model_files(model_id=MODEL_ID)
            print(f"Model files: {files}")
        except Exception as e2:
            print(f"  Also failed: {e2}")
        sys.exit(1)


if __name__ == "__main__":
    main()
