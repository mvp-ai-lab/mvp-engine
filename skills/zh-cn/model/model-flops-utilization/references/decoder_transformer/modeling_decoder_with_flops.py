from __future__ import annotations

from transformers import GPT2LMHeadModel


class GPT2LMHeadModelWithFlops(GPT2LMHeadModel):
    """带有 calculate_model_flops 的 decoder-only Transformer 参考实现。"""

    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        seq_len: int | None = None,
        image_size: int | tuple[int, int] | None = None,
        patch_size: int | tuple[int, int] | None = None,
        is_training: bool = True,
    ) -> float:
        if seq_len is None:
            raise ValueError("Transformer FLOPs requires seq_len.")

        batch = int(batch_size)
        tokens = int(seq_len)
        if batch <= 0 or tokens <= 0:
            raise ValueError("batch_size and seq_len must be > 0")

        layers = int(self.config.n_layer)
        hidden = int(self.config.n_embd)
        intermediate = int(4 * hidden)
        vocab = int(self.config.vocab_size)

        per_layer = (
            8 * batch * tokens * hidden * hidden
            + 4 * batch * tokens * tokens * hidden
            + 4 * batch * tokens * hidden * intermediate
        )
        decoder_flops = layers * per_layer
        lm_head_flops = 2 * batch * tokens * hidden * vocab
        forward_flops = float(decoder_flops + lm_head_flops)
        return forward_flops * 3.0 if is_training else forward_flops
