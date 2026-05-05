"""Dataset processing utilities for the OpenBee recipe."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from mvp_dataset import Dataset
from mvp_dataset.core import RuntimeContext

from ..configs.schema import OpenbeeConfig
from ..guards.data import build_dataguard
from .packing import build_packed_sample_assembler, finalize_packed_sample_group
from .preprocess import convert_images_to_pixel_values, process_sample
from .skip import SkipMode, build_skip_by_worker, build_skip_recorder


def build_dataset(
    config: OpenbeeConfig,
    *,
    processor: Any,
    process_fn: Any = process_sample,
    resample: bool = True,
    resolve_refs: bool = True,
    skip_mode: SkipMode = "off",
    skip_counts: Mapping[int, int] | None = None,
) -> Dataset:
    """Build the training dataset pipeline for the recipe.

    Args:
        config: Recipe config with dataset, runtime, and shuffle settings.
        processor: Hugging Face processor used during sample processing.
        process_fn: Function to process individual samples.
        resample: Whether to loop dataset shards indefinitely across rounds.
        resolve_refs: Whether to resolve references in the dataset.
        skip_mode: Optional post-pack fast-resume mode.
        skip_counts: Worker-slot skip counts used by ``skip_mode="perform"``.
    Returns:
        An ``mvp_dataset.Dataset`` pipeline with parquet loading, processing, and
        sample-level shuffling.
    """
    dataset_path = config.data.train_path
    if dataset_path is None:
        raise ValueError("Missing `data.train_path` for the OpenBee recipe.")

    context = RuntimeContext.from_runtime(seed=int(config.seed))

    # 1. Create the data source.
    dataset = Dataset.from_source(
        "lance",
        dataset_path,
        context=context,
        resample=resample,
        shuffle_mode="fragment_aware",
    )

    # 2. Data guard for handling invalid/bad samples.
    dataset = dataset.assemble(
        partial(
            build_dataguard,
            check_basic_formats=True,
            check_input_ids=False,
            check_image_sizes=True,
        )
    )

    # 3. Pre-process the data samples.
    process_kwargs: dict[str, Any] = {
        "processor": processor,
        "max_length": int(config.data.max_seq_len),
        "thinking_mode": config.data.thinking_mode,
    }
    dataset = dataset.map(partial(process_fn, **process_kwargs))

    # 4. Guard to filter out any bad samples
    dataset = dataset.assemble(
        partial(
            build_dataguard,
            check_basic_formats=False,
            check_input_ids=True,
            check_image_sizes=False,
            record=False,
        )
    )

    # 5. Optionally pack samples into longer sequences before resolving image refs.
    if config.data.packing:
        dataset = dataset.assemble(
            partial(
                build_packed_sample_assembler,
                max_length=config.data.max_seq_len,
                selection_strategy=config.data.packing_selection_strategy,
                open_pack_limit=config.data.packing_open_pack_limit,
                pack_buffer_size=config.data.packing_buffer_size,
                defer_finalize=True,
            )
        )

    # 6. Optional post-pack resume skip boundary. The recorder path returns here
    # so the lightweight resume pass never resolves image references.
    if skip_mode == "pre_calculate":
        return dataset.assemble(build_skip_recorder)
    elif skip_mode == "perform":
        dataset = dataset.assemble(
            partial(
                build_skip_by_worker, skip_counts={str(slot): int(count) for slot, count in (skip_counts or {}).items()}
            )
        )
    elif skip_mode != "off":
        raise ValueError(f"Invalid skip_mode: {skip_mode}. Must be one of 'off', 'pre_calculate', or 'perform'.")

    # 7. Resolve references after packing so invalid/short samples avoid image IO.
    if resolve_refs:
        # TODO: add error handeling inside the mvp-dataset
        # TODO: what about pure text data?
        dataset = dataset.resolve_ref(ref_names=config.data.ref_columns).map(
            partial(convert_images_to_pixel_values, processor=processor)
        )

    # 8. Drop sentinels created by late image decode/materialization failures.
    # In packed mode this also drops empty packed groups before finalization.
    dataset = dataset.assemble(
        partial(
            build_dataguard,
            check_basic_formats=False,
            check_input_ids=True,
            check_image_sizes=False,
            record=False,
        )
    )

    # 9. Materialize deferred packs after references have been resolved.
    if config.data.packing:
        dataset = dataset.map(finalize_packed_sample_group)

    return dataset


if __name__ == "__main__":
    from transformers import AutoProcessor

    from ..configs.schema import OpenbeeDataConfig

    processor = AutoProcessor.from_pretrained("./pretrained/Qwen3-VL-8B-Base-woDS-stage2")
    ds = build_dataset(
        OpenbeeConfig(
            data=OpenbeeDataConfig(
                train_path="/mnt/data-alpha-sg-01/team-camera/shared/mvp-engine/data/Open-Bee-Lance/stage3/meta.json",
                max_seq_len=16384,
                thinking_mode="non-empty",
                packing=True,
                packing_selection_strategy="best_fit",
                packing_open_pack_limit=8,
                packing_buffer_size=64,
                shuffle_on_packs=True,
                ref_columns=["images"],
            ),
            seed=42,
        ),
        processor=processor,
        process_fn=process_sample,
        resample=True,
        resolve_refs=True,
    )

    from tqdm import tqdm

    for data in tqdm(ds):
        print(data)
        break

    # sample = {
    #     "images": [
    #         "COYO-Recaption:Caption/COYO-Recaption/train-00062-of-00110.parquet:5949:images:0"
    #     ],
    #     "conversations": [
    #         {"from": "human", "value": "<image>\n"},
    #         {
    #             "from": "gpt",
    #             "value": "<think>\n\n</think>\n\nThe image shows a modern living room setup with a focus on a wall-mounted multi-panel artwork and a sofa arrangement. Here's a detailed description:\n\n### Wall Art:\n- The wall art is a black-and-white photograph of a bridge, likely the Manhattan Bridge in New York City, divided into four vertical panels.\n- The leftmost panel shows a cityscape with tall buildings and a river in the foreground.\n- The second panel from the left features the bridge's suspension cables and part of the bridge structure.\n- The third panel shows the central part of the bridge, including the towers and the roadway.\n- The rightmost panel captures the continuation of the bridge and the cityscape on the other side.\n- The panels are evenly spaced and aligned, creating a cohesive and visually striking composition.\n\n### Sofa and Decor:\n- Below the wall art is a gray fabric sofa with a clean, contemporary design.\n- The sofa is adorned with four throw pillows:\n  - A mustard-yellow pillow on the far left.\n  - A black-and-white striped pillow in the center.\n  - Two beige pillows on the right, one slightly larger than the other.\n- In front of the sofa is a low coffee table with a wicker basket design. Inside the basket are a few books and magazines, and a small potted plant with greenery is placed on the left side of the table.\n- A glass of water is also visible on the table, along with a small decorative item that appears to be a yellow flower or similar object.\n\n### Wall and Room:\n- The wall behind the sofa is painted a neutral gray color with a subtle vertical striped texture.\n- The overall ambiance of the room is modern and minimalist, with a focus on clean lines and a monochromatic color scheme accented by the yellow pillow and plant.\n\nThe image conveys a sense of sophistication and urban style, with the bridge artwork serving as the focal point.",
    #         },
    #     ],
    #     "id": "coyo_clean_sample_7m_0_37216",
    #     "img_phash": None,
    #     "img_size": [[600, 600]],
    #     "source": "coyo",
    #     "__source__": "COYO-Recaption",
    #     "__file__": "/mnt/data-alpha-sg-01/team-camera/shared/mvp-engine/data/Open-Bee-Lance/stage3/samples.lance",
    #     "__local_index__": 10804301,
    #     "__global_index__": 10804301,
    #     "__key__": "/mnt/data-alpha-sg-01/team-camera/shared/mvp-engine/data/Open-Bee-Lance/stage3/samples.lance:10804301",
    # }

    # sample = process_sample(
    #     sample,
    #     processor=processor,
    #     max_length=16384,
    #     thinking_mode="non-empty",
    # )

    # print(sample)
