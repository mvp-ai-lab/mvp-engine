"""Recipe-level cumulative smoke tests for installed skills."""

from __future__ import annotations

import os
import runpy
import socket
import traceback
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from typing import Any

import torch

from mvp_engine.engine import TrainStepContext
from mvp_engine.test.recipe_probe import (
    build_engine,
    load_config,
    multi_rank_distributed_env,
    single_rank_distributed_env,
)
from mvp_engine.utils import skill_testing_util
from mvp_engine.utils.log import get_logger
from mvp_engine.utils.skill_testing_util import CURRENT_SKILL_ENV
from recipes.magic_transformer.configs.schema import MagicTransformerConfig

RECIPE_PATH = Path("recipes/magic_transformer")
MULTI_RANK_TIMEOUT_SECONDS = 120
SMOKE_CONFIG_OVERRIDE = {
    "parallel": {
        "mesh": {
            "replicate": 1,
            "shard": 1,
            "tensor": 1,
        },
    },
    "data": {
        "fake_train_size": 8,
        "fake_eval_size": 4,
        "batch_size": 2,
        "num_workers": 0,
        "seq_len": 8,
        "vocab_size": 128,
    },
    "model": {
        "vocab_size": 128,
        "max_seq_len": 8,
        "d_model": 32,
        "n_heads": 4,
        "n_kv_heads": 2,
        "n_layers": 2,
        "dropout": 0.0,
    },
    "optim": {
        "mixed_precision": "fp32",
    },
}


def test_recipe_smoke_matches_installed_skill_assertions(tmp_path: Path) -> None:
    skill_asserts = _load_skill_asserts()
    output_dir = tmp_path / "smoke_outputs"
    config = _load_smoke_config(output_dir=output_dir, run_id="smoke")
    world_size = _get_config_world_size(config)

    if world_size > 1:
        errors = _run_multi_gpu_test(output_dir=output_dir, run_id="smoke", world_size=world_size)
        assert not errors, "\n\n".join(errors)
        return

    _run_single_gpu_test(config, skill_asserts)


def _load_skill_asserts() -> tuple[tuple[str, dict[str, Any]], ...]:
    return tuple(
        (skill_id, runpy.run_path(str(asserts_path)))
        for skill_id, asserts_path in skill_testing_util.get_ordered_skill_asserts(
            RECIPE_PATH,
            current_skill_id=os.environ.get(CURRENT_SKILL_ENV),
        )
    )


def _run_single_gpu_test(config: Any, skill_asserts: tuple[tuple[str, dict[str, Any]], ...]) -> None:
    with single_rank_distributed_env():
        engine = build_engine(RECIPE_PATH, config, config_class=MagicTransformerConfig)
        try:
            _run_engine_smoke(engine, skill_asserts)
        finally:
            logger_instance = get_logger()
            if logger_instance is not None:
                logger_instance.destroy()


def _run_multi_gpu_test(*, output_dir: Path, run_id: str, world_size: int) -> list[str]:
    _require_visible_cuda_devices(world_size)
    master_port = _find_free_port()
    context = get_context("spawn")
    result_queue = context.Queue()

    processes = [
        context.Process(
            target=_multi_gpu_worker,
            args=(rank, world_size, master_port, output_dir, run_id, result_queue),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    for process in processes:
        process.join(MULTI_RANK_TIMEOUT_SECONDS)

    errors: list[str] = []
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

    return errors


def _run_engine_smoke(engine: Any, skill_asserts: tuple[tuple[str, dict[str, Any]], ...]) -> None:
    engine.before_train()
    batch = next(iter(engine.train_loader))

    ctx = TrainStepContext(
        data=batch,
        step=engine.step,
        epoch=engine.epoch,
        micro_step=engine.ga_state.micro_step,
    )
    prepared = engine.train_pre_step(ctx)
    if prepared is not None and prepared is not ctx:
        ctx.data = prepared

    engine.train_exec_step(ctx)
    engine.train_post_step(ctx)

    run_dir = Path(engine.project_dir)
    log_file = run_dir / f"log_{engine.run_id}.log"
    checkpoint_dir = run_dir / "checkpoints" / "iter_1"
    for skill_id, asserts in skill_asserts:
        assert_smoke = asserts.get("assert_smoke")
        if assert_smoke is None:
            raise AssertionError(f"{skill_id} asserts.py must define assert_smoke.")
        assert_smoke(engine=engine, ctx=ctx, batch=batch, log_file=log_file, checkpoint_dir=checkpoint_dir)
    engine.after_train()


def _multi_gpu_worker(
    rank: int,
    world_size: int,
    master_port: int,
    output_dir: Path,
    run_id: str,
    result_queue: Any,
) -> None:
    try:
        with multi_rank_distributed_env(rank=rank, world_size=world_size, master_port=master_port, local_rank=rank):
            config = _load_smoke_config(output_dir=output_dir, run_id=run_id)
            engine = build_engine(RECIPE_PATH, config, config_class=MagicTransformerConfig)
            try:
                _run_engine_smoke(engine, _load_skill_asserts())
            finally:
                logger_instance = get_logger()
                if logger_instance is not None:
                    logger_instance.destroy()

        result_queue.put((rank, True, ""))
    except BaseException:
        result_queue.put((rank, False, traceback.format_exc()))
        raise
    finally:
        logger_instance = get_logger()
        if logger_instance is not None:
            logger_instance.destroy()


def _load_smoke_config(*, output_dir: Path, run_id: str):
    return load_config(
        RECIPE_PATH,
        output_dir=output_dir,
        run_id=run_id,
        total_steps=1,
        extra_override=SMOKE_CONFIG_OVERRIDE,
    )


def _get_config_world_size(config: Any) -> int:
    mesh = config.parallel.mesh
    mesh_dims = (int(mesh.replicate), int(mesh.shard), int(mesh.tensor))
    if any(dim <= 0 for dim in mesh_dims):
        raise AssertionError(f"Smoke config mesh dimensions must be explicit positive values, got {mesh_dims}.")

    world_size = 1
    for dim in mesh_dims:
        world_size *= dim
    return world_size


def _require_visible_cuda_devices(world_size: int) -> None:
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        return

    skill_id = os.environ.get(CURRENT_SKILL_ENV, "new-recipe-template")
    command = f"python -m tests.test_skills --recipe magic_transformer --skill {skill_id} --layer smoke"
    skill_testing_util.raise_real_env_required(
        command=command,
        reason=f"Multi-GPU smoke needs {world_size} visible CUDA devices.",
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
