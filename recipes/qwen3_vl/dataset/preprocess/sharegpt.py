from io import BytesIO

from PIL import Image


def sharegpt_mapping_fn(sample: dict) -> dict:
    mapped = {}
    mapped["id"] = sample["id"]
    mapped["conversations"] = sample["conversations"]
    mapped["image"] = Image.open(BytesIO(sample["image"]))
    mapped["image"].load()
    mapped["image"] = mapped["image"].convert("RGB")
    return mapped
