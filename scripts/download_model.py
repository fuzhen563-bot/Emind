"""从 ModelScope 下载 fuzhen/emind-0.5b"""
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download
import os

TOKEN = "ms-30159468-0600-4910-96fe-f317c2abe100"
OUTPUT_DIR = "checkpoints/emind-0.5b-merged"

api = HubApi()
api.login(TOKEN)

os.makedirs(OUTPUT_DIR, exist_ok=True)

local_dir = snapshot_download(
    'fuzhen/emind-0.5b',
    local_dir=OUTPUT_DIR,
    user_agent={"auth_token": TOKEN},
)

print(f"Downloaded to: {local_dir}")
print("Contents:", os.listdir(local_dir))
