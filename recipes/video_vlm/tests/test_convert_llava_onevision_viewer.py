"""Tests for converting LLaVA-OneVision viewer parquet into Video VLM rows."""

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from recipes.video_vlm.tools.convert_llava_onevision_viewer import main


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_convert_viewer_parquet(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "input" / "viewer"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "spatial-1",
                    "task": "demo",
                    "image": {"bytes": _png_bytes(), "path": "demo.png"},
                    "preview": "[user] Where is the object? [assistant] On the table.",
                }
            ]
        ),
        input_dir / "spatial.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "video": {"bytes": b"fake-video-bytes", "path": "demo.mp4"},
                    "preview": "[user] Describe the video. [assistant] A short clip.",
                }
            ]
        ),
        input_dir / "caption_gt10min.parquet",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "convert",
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(output_dir),
            "--image-limit",
            "1",
            "--video-limit",
            "1",
        ],
    )

    main()

    rows = pq.read_table(output_dir / "demo.parquet").to_pylist()
    assert len(rows) == 2
    assert rows[0]["messages"][0]["content"].startswith("<image>\n")
    assert rows[0]["images"] == [str((output_dir / "media/images/spatial_0000.png").resolve())]
    assert rows[0]["image_size"] == [[8, 6]]
    assert rows[1]["messages"][0]["content"].startswith("<video>\n")
    assert rows[1]["images"] == []
    assert rows[1]["image_size"] == []
    assert rows[1]["videos"] == [str((output_dir / "media/videos/caption_gt10min_0000.mp4").resolve())]
    assert Path(rows[0]["images"][0]).is_file()
    assert Path(rows[1]["videos"][0]).read_bytes() == b"fake-video-bytes"
