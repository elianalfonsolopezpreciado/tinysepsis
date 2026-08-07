"""Download PhysioNet Challenge 2019 sepsis dataset from the public open S3 bucket.

Open-access dataset, no PhysioNet account or credentialing required
(verified via https://physionet.org/content/challenge-2019/1.0.0/).
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "physionet-open"
PREFIX = "challenge-2019/1.0.0/training/"
DEST = Path(__file__).resolve().parent.parent / "data" / "raw"
REGION = "us-east-1"


def make_client():
    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 5, "mode": "adaptive"}),
    )


def list_keys():
    s3 = make_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".psv"):
                keys.append((obj["Key"], obj["Size"]))
    return keys


def download_one(key, size):
    rel = key[len(PREFIX):]
    dest_path = DEST / rel
    if dest_path.exists() and dest_path.stat().st_size == size:
        return "skip"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    s3 = make_client()
    s3.download_file(BUCKET, key, str(dest_path))
    return "ok"


def main():
    print("Listing objects...", flush=True)
    keys = list_keys()
    print(f"Found {len(keys)} .psv files under s3://{BUCKET}/{PREFIX}", flush=True)
    total_bytes = sum(size for _, size in keys)
    print(f"Total size: {total_bytes / 1e6:.1f} MB", flush=True)

    done = 0
    skipped = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = {ex.submit(download_one, k, s): k for k, s in keys}
        for fut in as_completed(futs):
            result = fut.result()
            done += 1
            if result == "skip":
                skipped += 1
            if done % 1000 == 0 or done == len(keys):
                elapsed = time.time() - t0
                print(f"{done}/{len(keys)} ({skipped} skipped) - {elapsed:.0f}s", flush=True)

    print(f"Done. Skipped {skipped} already-present files.", flush=True)


if __name__ == "__main__":
    main()
