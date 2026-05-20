Override `backward_step`, `optimizer_step`, or `train_post_step` only when the recipe needs special loss normalization, optimizer timing, or custom metrics.

Keep the engine thin: put model code in `model/`, data code in `dataset/`, and experiment configs in `configs/`.

![A tiny knight debugging a missing page](/04-fig-cutout.png)
