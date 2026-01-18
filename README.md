<p align="center">
  <picture>
    <img alt="MVP Engine" src="./assets/logo.png" width="600" style="max-width: 100%;">
  </picture>
</p>

<p align="center">
  <strong>Fully Open and Easy-to-Use Framework for Democratized Multimodal Model Training</strong>
</p>


## Getting Started

```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12355 -m mvp_engine.launch --config ./recipes/tomatovit/configs/stage1.yaml
```