# STADStream

统一的传感器时序动作检测实验库。当前目标是先把 WiFiTAD 数据集跑通，并把离线 DPWiT/WiFiTAD、SlimSTAD 风格 baseline 和在线 TAL baseline 接到同一套输入格式。

Important: `src/stadstream/models/*` 里的模型是 unified adapter / pipeline sanity 版本，不等同于论文严格复现。正式 baseline 数字应遵循 [Faithful Reproduction Policy](docs/FAITHFUL_REPRODUCTION.md)：有官方代码就保留官方训练、测试、loss、decoder、NMS 和 eval，只改传感器输入适配与必要的输出导出。

```text
input:  torch.Tensor [B, C, T]
target: list[Tensor[N, 3]], each row = [start_norm, end_norm, class_id]
output: {"loc": [B, A, 2], "conf": [B, A, K], "priors": [1, A, 1]}
```

## Unified Adapter Models

- `dpwit`: DPWiT/WiFiTAD 风格双金字塔 detector，文件在 `src/stadstream/models/dpwit.py`
- `slimstad`: SlimSTAD 风格轻量时序 detector，文件在 `src/stadstream/models/slimstad.py`
- `matr_signal`: 基于 MATR online TAL 思路的 memory transformer 传感器适配版，文件在 `src/stadstream/models/matr_signal.py`
- `moad_signal`: 基于 MOAD/Backtrace Mamba 思路的层次记忆压缩在线 baseline，文件在 `src/stadstream/models/moad_signal.py`

MATR 官方代码可放在 `../external/MATR_codebase`；本库里的 `matr_signal` 只是传感器输入接口适配版本，便于后续把官方模块迁入或包装。MOAD 当前没有找到官方公开代码，`moad_signal` 不能作为官方 MOAD 复现结果。

## Smoke Test

在 `E:\TAD\STADStream` 下运行：

```bash
python scripts/smoke_train.py --config configs/wifitad_mini.yaml --model dpwit --max-batches 1
python scripts/smoke_train.py --config configs/wifitad_mini.yaml --model slimstad --max-batches 1
python scripts/smoke_train.py --config configs/wifitad_mini.yaml --model matr_signal --max-batches 1
python scripts/smoke_train.py --config configs/wifitad_mini.yaml --model moad_signal --max-batches 1
```

这些命令只验证 WiFiTAD mini 数据、forward、loss、backward 都能跑通，不代表最终论文指标。

## Online Smoke Test

DPWiT 和 SlimSTAD 当前通过 past-only sliding-window wrapper 转成在线推理模式；MATR/MOAD 通过 chunk-native wrapper 每步只接收当前 chunk，并依赖模型内部 memory 保存历史：

```bash
python scripts/stream_smoke.py --config configs/wifitad_mini.yaml --model dpwit --max-chunks 4
python scripts/stream_smoke.py --config configs/wifitad_mini.yaml --model slimstad --max-chunks 4
python scripts/stream_smoke.py --config configs/wifitad_mini.yaml --model matr_signal --max-chunks 4
python scripts/stream_smoke.py --config configs/wifitad_mini.yaml --model moad_signal --max-chunks 4
```

在线 wrapper 位于 `src/stadstream/online.py`，核心接口是 `reset_stream(video)` 和 `stream_step(chunk)`。

## Full Train/Test/Eval

统一入口：

```bash
python scripts/run_experiment.py --config configs/wifitad_mini.yaml --model slimstad --mode train_test_eval
```

也可以分阶段跑：

```bash
python scripts/run_experiment.py --config configs/wifitad_mini.yaml --model slimstad --mode train --run-dir runs/slimstad-mini
python scripts/run_experiment.py --config configs/wifitad_mini.yaml --model slimstad --mode stream_test --checkpoint runs/slimstad-mini/checkpoint_last.pt --run-dir runs/slimstad-mini
python scripts/run_experiment.py --config configs/wifitad_mini.yaml --model slimstad --mode eval --pred runs/slimstad-mini/slimstad_stream_predictions.json --run-dir runs/slimstad-mini
```
