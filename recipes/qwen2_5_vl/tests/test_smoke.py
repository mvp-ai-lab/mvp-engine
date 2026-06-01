"""Smoke tests for the qwen2_5_vl recipe."""

import traceback
from multiprocessing import get_context
from pathlib import Path
from queue import Empty

import pytest

from mvp_engine.testing.utils import (
    distributed_env,
    find_free_port,
    inject_engine_assert_hooks,
    load_recipe_skill_asserts,
    load_smoke_test_config,
)

CONFIG_NAME = "stage1"
PROCESS_TIMEOUT_SECONDS = 12000


def smoke_process(
    rank: int,
    world_size: int,
    master_port: int,
    output_dir: Path,
    config_overrides: tuple[str, ...],
    result_queue,
) -> None:
    """Run one distributed smoke worker and report its traceback to the parent."""
    try:
        with distributed_env(
            rank=rank,
            world_size=world_size,
            local_rank=rank,
            master_port=master_port,
            master_addr="127.0.0.1",
        ):
            from mvp_engine.engine import ENGINE_REGISTRY
            from mvp_engine.launch import _import_recipe_modules

            recipe_root = Path(__file__).resolve().parent.parent
            _import_recipe_modules(recipe_root)
            config = load_smoke_test_config(
                recipe_root,
                CONFIG_NAME,
                output_dir=output_dir,
                config_overrides=config_overrides,
            )

            engine = ENGINE_REGISTRY.get(config.engine)(config)
            inject_engine_assert_hooks(engine, load_recipe_skill_asserts(recipe_root))
            engine.train()

        result_queue.put((rank, True, ""))
    except BaseException:
        result_queue.put((rank, False, traceback.format_exc()))
        raise


def test_smoke(
    run_smoke: bool,
    world_size: int,
    config_overrides: tuple[str, ...],
    tmp_path: Path,
):
    """Run smoke validation only when explicitly requested."""
    if not run_smoke:
        pytest.skip("qwen2_5_vl smoke test requires --run-smoke and accelerator resources.")

    master_port = find_free_port()
    output_dir = tmp_path / "smoke_outputs"
    context = get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=smoke_process,
            args=(rank, world_size, master_port, output_dir, config_overrides, result_queue),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    for process in processes:
        process.join(PROCESS_TIMEOUT_SECONDS)

    errors = []
    for process in processes:
        if process.is_alive():
            process.terminate()
            errors.append(f"rank process {process.pid} timed out.")
        elif process.exitcode != 0:
            errors.append(f"rank process {process.pid} exited with code {process.exitcode}.")

    while True:
        try:
            rank, passed, message = result_queue.get_nowait()
        except Empty:
            break
        if not passed:
            errors.append(f"rank {rank} failed:\n{message}")

    assert not errors, "\n".join(errors)

