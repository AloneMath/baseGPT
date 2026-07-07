"""
Downloads shards of the karpathy/climbmix-400b-shuffle dataset from HuggingFace.

Usage:
    python train_dataset.py --num-shards 170 --data-dir data
"""
import argparse
import os
import time

import requests
from tqdm import tqdm

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542  # last available shard index in the dataset (shard_06542.parquet)
DEFAULT_NUM_SHARDS = 170
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NUM_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


def shard_filename(index: int) -> str:
    return f"shard_{index:05d}.parquet"


def download_shard(index: int, data_dir: str) -> str:
    """Downloads a single shard, skipping it if already present. Returns the local file path."""
    filename = shard_filename(index)
    final_path = os.path.join(data_dir, filename)
    if os.path.exists(final_path):
        return final_path

    url = f"{BASE_URL}/{filename}"
    tmp_path = final_path + ".tmp"

    last_error = None
    for attempt in range(1, NUM_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                with open(tmp_path, "wb") as f, tqdm(
                    total=total_size, unit="B", unit_scale=True, desc=filename, leave=False
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        pbar.update(len(chunk))
            os.replace(tmp_path, final_path)
            return final_path
        except requests.RequestException as e:
            last_error = e
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if attempt < NUM_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"Failed to download {filename} after {NUM_RETRIES} attempts: {last_error}")


def download_shards(num_shards: int, data_dir: str) -> None:
    num_shards = min(num_shards, MAX_SHARD + 1)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading {num_shards} shards from {BASE_URL} into {data_dir}")
    for index in tqdm(range(num_shards), desc="shards"):
        download_shard(index, data_dir)
    print(f"Done. {num_shards} shards available in {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download ClimbMix dataset shards from HuggingFace")
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS, help="number of shards to download")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="directory to store downloaded shards")
    args = parser.parse_args()
    download_shards(args.num_shards, args.data_dir)


if __name__ == "__main__":
    main()
