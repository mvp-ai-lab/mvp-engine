"""
WebDataset Viewer - A Flask web application to visualize webdataset samples.

This tool allows users to browse and inspect webdataset tar files,
displaying images, depth maps, and metadata in a web interface.
"""

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import webdataset as wds
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)

# Global state
DATASET_PATH = None
SHARD_PATHS = []
CURRENT_SAMPLES = []
SAMPLE_INDEX = 0


def find_shards(dataset_path: str) -> List[str]:
    """Find all tar files in the dataset path."""
    path = Path(dataset_path)
    if not path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    shards = sorted([str(p) for p in path.rglob("*.tar")])
    if not shards:
        raise ValueError(f"No tar files found in {dataset_path}")

    return shards


def decode_image_data(data: bytes) -> Optional[Image.Image]:
    """Decode image data from bytes."""
    try:
        img = Image.open(io.BytesIO(data))
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string for web display."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def depth_to_image(depth_data: bytes) -> Optional[Image.Image]:
    """Convert depth data to a viewable image."""
    try:
        # Try to decode as image first
        depth_img = Image.open(io.BytesIO(depth_data))

        # Convert to numpy array for visualization
        depth_array = np.array(depth_img)

        # Normalize to 0-255 range for visualization
        if depth_array.max() > 0:
            depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255
        else:
            depth_normalized = depth_array

        # Apply colormap (using a simple grayscale to color mapping)
        depth_color = np.zeros((*depth_array.shape[:2], 3), dtype=np.uint8)
        depth_color[..., 0] = depth_normalized  # R channel
        depth_color[..., 1] = depth_normalized * 0.7  # G channel
        depth_color[..., 2] = 255 - depth_normalized  # B channel (inverted)

        return Image.fromarray(depth_color, mode="RGB")
    except Exception as e:
        print(f"Error processing depth data: {e}")
        return None


def load_samples(num_samples: int = 100) -> List[Dict]:
    """Load samples from the webdataset."""
    global SHARD_PATHS, CURRENT_SAMPLES

    samples = []
    dataset = wds.WebDataset(SHARD_PATHS, shardshuffle=False)

    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        processed_sample = {
            "__key__": sample.get("__key__", f"sample_{i}"),
            "meta": None,
            "image": None,
            "depth": None,
            "image_key": None,
            "depth_key": None,
            "all_keys": list(sample.keys()),
        }

        # Extract metadata
        if "meta.json" in sample:
            try:
                if isinstance(sample["meta.json"], bytes):
                    processed_sample["meta"] = json.loads(sample["meta.json"].decode("utf-8"))
                else:
                    processed_sample["meta"] = sample["meta.json"]
            except Exception as e:
                print(f"Error parsing metadata: {e}")

        # Find and process image
        for key in sample.keys():
            if key.startswith("images."):
                processed_sample["image_key"] = key
                processed_sample["image"] = sample[key]
                break

        # Find and process depth
        for key in sample.keys():
            if key.startswith("depths."):
                processed_sample["depth_key"] = key
                processed_sample["depth"] = sample[key]
                break

        samples.append(processed_sample)

    CURRENT_SAMPLES = samples
    return samples


@app.route("/")
def index():
    """Serve the main viewer page."""
    return render_template("index.html")


@app.route("/api/init", methods=["POST"])
def init_dataset():
    """Initialize the dataset from the provided path."""
    global DATASET_PATH, SHARD_PATHS, CURRENT_SAMPLES, SAMPLE_INDEX

    data = request.json
    dataset_path = data.get("dataset_path", "")

    if not dataset_path:
        return jsonify({"error": "No dataset path provided"}), 400

    try:
        DATASET_PATH = dataset_path
        SHARD_PATHS = find_shards(dataset_path)
        SAMPLE_INDEX = 0

        # Load initial samples
        load_samples(num_samples=100)

        return jsonify(
            {
                "success": True,
                "num_shards": len(SHARD_PATHS),
                "num_samples": len(CURRENT_SAMPLES),
                "shards": SHARD_PATHS[:10],  # Return first 10 shard paths
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample/<int:index>")
def get_sample(index: int):
    """Get a specific sample by index."""
    global CURRENT_SAMPLES

    if not CURRENT_SAMPLES:
        return jsonify({"error": "No samples loaded. Initialize dataset first."}), 400

    if index < 0 or index >= len(CURRENT_SAMPLES):
        return jsonify({"error": f"Index {index} out of range [0, {len(CURRENT_SAMPLES)}]"}), 400

    sample = CURRENT_SAMPLES[index]

    # Process image
    image_b64 = None
    if sample["image"] is not None:
        img = decode_image_data(sample["image"])
        if img:
            image_b64 = image_to_base64(img)

    # Process depth
    depth_b64 = None
    if sample["depth"] is not None:
        depth_img = depth_to_image(sample["depth"])
        if depth_img:
            depth_b64 = image_to_base64(depth_img)

    return jsonify(
        {
            "index": index,
            "total": len(CURRENT_SAMPLES),
            "key": sample["__key__"],
            "meta": sample["meta"],
            "image": image_b64,
            "depth": depth_b64,
            "image_key": sample["image_key"],
            "depth_key": sample["depth_key"],
            "all_keys": sample["all_keys"],
        }
    )


@app.route("/api/stats")
def get_stats():
    """Get dataset statistics."""
    global DATASET_PATH, SHARD_PATHS, CURRENT_SAMPLES

    return jsonify(
        {
            "dataset_path": DATASET_PATH,
            "num_shards": len(SHARD_PATHS),
            "num_samples_loaded": len(CURRENT_SAMPLES),
            "sample_keys": CURRENT_SAMPLES[0]["all_keys"] if CURRENT_SAMPLES else [],
        }
    )


def main():
    parser = argparse.ArgumentParser(description="WebDataset Viewer")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data/potato_v1/data",
        help="Path to the webdataset directory",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    # Initialize with provided dataset path
    if args.dataset_path:
        global DATASET_PATH, SHARD_PATHS
        try:
            DATASET_PATH = args.dataset_path
            SHARD_PATHS = find_shards(args.dataset_path)
            print(f"Found {len(SHARD_PATHS)} shards in {args.dataset_path}")
            load_samples(num_samples=100)
            print(f"Loaded {len(CURRENT_SAMPLES)} samples")
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")
            print("You can still start the server and provide the path through the web interface.")

    print(f"Starting WebDataset Viewer on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
