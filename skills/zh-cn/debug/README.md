# Debug

**调试与性能分析** 类 skill（如显存 profiling、loss 调试、OOM 定位）。流程有固定模式但依赖具体故障现象与模型。

## Skills

- `recipe-merge-repair`：先检查最近合入的共享代码是否破坏当前 recipe，再完成 recipe-local 修复。
