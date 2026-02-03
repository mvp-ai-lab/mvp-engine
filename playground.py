import json
import os
import tarfile
from io import BytesIO
from time import sleep

import numpy as np
import webdataset as wd
from PIL import Image

from mvp_engine.dataset.webdataset import WebDatasetBuilder

# ==============================================
# 模拟多 node 环境的配置
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

print(f"[Node Info] WORLD_SIZE={WORLD_SIZE}, RANK={RANK}, LOCAL_RANK={LOCAL_RANK}")

# ==============================================
# Generate a dataset consisting of 4 shards, each shard contains 8 images
os.makedirs("test_data", exist_ok=True)

for i in range(4):
    with tarfile.open(f"test_data/shard_img_00000{i}.tar", "w") as tar:
        for j in range(8):
            idx = i * 8 + j
            # Create sample image
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img_bytes = BytesIO()
            img.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            tarinfo = tarfile.TarInfo(name=f"{idx:06d}.jpg")
            tarinfo.size = len(img_bytes.getvalue())
            tar.addfile(tarinfo, img_bytes)

            # Create sample metadata
            metadata = {"id": idx, "label": idx % 3}
            metadata_bytes = BytesIO(json.dumps(metadata).encode("utf-8"))
            metadata_tarinfo = tarfile.TarInfo(name=f"{idx:06d}.json")
            metadata_tarinfo.size = len(metadata_bytes.getvalue())
            tar.addfile(metadata_tarinfo, metadata_bytes)

for i in range(4):
    with tarfile.open(f"test_data/shard_depth_00000{i}.tar", "w") as tar:
        for j in range(8):
            idx = i * 8 + j
            # Create sample image
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img_bytes = BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            tarinfo = tarfile.TarInfo(name=f"{idx:06d}.png")
            tarinfo.size = len(img_bytes.getvalue())
            tar.addfile(tarinfo, img_bytes)


# ==============================================
# Read this dataset
def make_sample(sample):
    return {
        "id": sample["json"]["id"],
        "image": sample["jpg"],
        "label": sample["json"]["label"],
        "__url__": sample["__url__"],
    }


ds = WebDatasetBuilder("./test_data/shard_img_{000000..000003}.tar").build(
    batch_size=1,
    shuffle_buffer=-1,
    make_sample_fn=make_sample,
)

TOTAL_SAMPLES = 4 * 8
NUM_WORKERS = 2

loader = wd.WebLoader(ds, batch_size=None, num_workers=NUM_WORKERS)

EPOCH_ITERS = TOTAL_SAMPLES // WORLD_SIZE
i = 0


seen_ids = set()
for batch in loader:
    for data in batch:
        print(f"Rank {RANK} got sample ID: {data['id']} from {data['__url__']}")

        seen_ids.add(data["id"])

        i += 1
        if i == EPOCH_ITERS:
            i = 0
            print(f"=> Epoch complete. Seen {len(seen_ids)} unique IDs: {sorted(seen_ids)}")
            seen_ids.clear()

    sleep(0.1)
