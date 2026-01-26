import sys

import torch
import torch.distributed as dist

from mvp_engine.utils.log import logger


def merge_partial_fc(ckpt_path, data_type, save=False):
    all_weights = []

    # find all ckpt_path / f"rgb/depth_head_rank{rank}.pt"
    for rank in range(sys.maxsize):
        partial_ckpt_path = ckpt_path / f"{data_type}_head_rank{rank}.pt"
        if partial_ckpt_path.is_file():
            state_dict = torch.load(partial_ckpt_path, map_location="cpu")
            weight_part = state_dict["weight"]
            all_weights.append(weight_part)
        else:
            break
    if not all_weights:
        raise FileNotFoundError(f"No partial FC checkpoint files found in directory: {ckpt_path}")

    full_weight = torch.cat(all_weights, dim=0)

    if save:
        merged_state_dict = {"weight": full_weight}
        torch.save(merged_state_dict, ckpt_path / f"{data_type}_head_full.pt")
    return full_weight


def repartition_fc(ckpt_path, world_size, rank, data_type):
    def get_split_info(r, total_n, ws):
        num_per_rank = total_n // ws
        remainder = total_n % ws
        l_n = num_per_rank + 1 if r < remainder else num_per_rank
        s_idx = r * l_n if r < remainder else r * num_per_rank + remainder
        return l_n, s_idx

    # if old_world_size == new world_size, skip merge
    count = 0
    for i in range(sys.maxsize):
        partial_ckpt_path = ckpt_path / f"{data_type}_head_rank{i}.pt"
        if partial_ckpt_path.is_file():
            count += 1
        else:
            break
    if count == world_size:
        partial_ckpt_path = ckpt_path / f"{data_type}_head_rank{rank}.pt"
        state_dict = torch.load(partial_ckpt_path, map_location="cpu")
        return state_dict

    local_weight = None
    if rank == 0:
        full_weight = merge_partial_fc(ckpt_path, data_type)
        if world_size == 1:
            merged_state_dict = {"weight": full_weight}
            return merged_state_dict

        total_classes, dims = full_weight.shape
        meta = torch.tensor([total_classes, dims], dtype=torch.long)
        cpu_group = dist.new_group(backend="gloo")
        dist.broadcast(meta, src=0, group=cpu_group)

        # send split weight
        for r in range(world_size):
            l_n, s_idx = get_split_info(r, total_classes, world_size)
            target_slice = full_weight[s_idx : s_idx + l_n].contiguous()

            if r == 0:
                local_weight = target_slice.clone()
            else:
                logger.info(f"Rank 0: send partial_fc [{s_idx}:{s_idx + l_n}] to Rank {r}...")
                dist.send(tensor=target_slice, dst=r, group=cpu_group)

        del full_weight
    else:
        meta = torch.zeros(2, dtype=torch.long)
        cpu_group = dist.new_group(backend="gloo")
        dist.broadcast(meta, src=0, group=cpu_group)
        total_classes, dims = meta.tolist()

        l_n, _ = get_split_info(rank, total_classes, world_size)
        local_weight = torch.empty((l_n, dims), dtype=torch.float32)
        dist.recv(tensor=local_weight, src=0, group=cpu_group)

    dist.destroy_process_group(cpu_group)
    return {"weight": local_weight}
