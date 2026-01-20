import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class iBOTLoss(nn.Module):
    def __init__(
        self,
        patch_out_dim,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_steps,
        nsteps,
        student_temp=0.1,
        center_momentum=0.9,
        lam=1.0,
        mim_start_step=0,
        warmup_steps=0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.lam = lam

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = (
            np.concatenate(
                (
                    np.linspace(
                        warmup_teacher_temp, teacher_temp, warmup_teacher_temp_steps
                    ),
                    np.ones(nsteps - warmup_teacher_temp_steps) * teacher_temp,
                )
            )
            if mim_start_step == 0
            else np.concatenate(
                (
                    np.ones(mim_start_step) * warmup_teacher_temp,
                    np.linspace(
                        warmup_teacher_temp, teacher_temp, warmup_teacher_temp_steps
                    ),
                    np.ones(nsteps - warmup_teacher_temp_steps - mim_start_step)
                    * teacher_temp,
                )
            )
        )
        self.lam_schedule = np.concatenate(
            (
                np.linspace(0, 1, warmup_steps),
                np.ones(nsteps - warmup_steps),
            )
        )

    def forward(self, student_patch, teacher_patch, student_mask, step):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        Only computes patch-level MIM loss on masked positions.

        Args:
            student_patch: [B, N, D] - student patch embeddings
            teacher_patch: [B, N, D] - teacher patch embeddings
            student_mask: [B, M] - indices of masked positions
            step: current training step for temperature scheduling

        Returns:
            dict with 'patch' and 'loss' keys
        """
        B, N, D = student_patch.shape

        # Gather masked positions: [B, M, D]
        mask_idx = student_mask.unsqueeze(-1).expand(-1, -1, D)  # [B, M, D]
        student_masked = torch.gather(student_patch, dim=1, index=mask_idx)  # [B, M, D]
        teacher_masked = torch.gather(teacher_patch, dim=1, index=mask_idx)  # [B, M, D]

        # Student: scale by temperature
        student_masked = student_masked / self.student_temp

        # Teacher: centering and sharpening
        temp = self.teacher_temp_schedule[step]
        teacher_masked = F.softmax((teacher_masked - self.center) / temp, dim=-1)
        teacher_masked = teacher_masked.detach()

        # Cross-entropy loss on masked positions
        loss = torch.sum(
            -teacher_masked * F.log_softmax(student_masked, dim=-1),
            dim=-1,
        )  # [B, M]
        loss = loss.mean()

        total_loss_ori = loss * self.lam
        total_loss = total_loss_ori * self.lam_schedule[step]
        self.update_center(teacher_patch)
        return total_loss, total_loss_ori

    @torch.no_grad()
    def update_center(self, teacher_patch):
        """
        Update center used for teacher output.
        """
        patch_center = teacher_patch.mean(dim=[0, 1], keepdim=True)  # [1, 1, D]
        dist.all_reduce(patch_center)
        patch_center = patch_center / dist.get_world_size()
        self.center = self.center * self.center_momentum + patch_center * (
            1 - self.center_momentum
        )
