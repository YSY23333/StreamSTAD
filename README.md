# STADStream

统一的传感器时序动作检测实验库。当前目标是先把 WiFiTAD 数据集跑通，并把离线 DPWiT/WiFiTAD、SlimSTAD 风格 baseline 和在线 TAL baseline 接到同一套输入格式：

```text
input:  torch.Tensor [B, C, T]
target: list[Tensor[N, 3]], each row = [start_norm, end_norm, class_id]
output: {"loc": [B, A, 2], "conf": [B, A, K], "priors": [1, A, 1]}
```

## 已接入模型

- `dpwit`: DPWiT/WiFiTAD 风格双金字塔 detector，文件在 `src/stadstream/models/dpwit.py`
- `slimstad`: SlimSTAD 风格轻量时序 detector，文件在 `src/stadstream/models/slimstad.py`
- `matr_signal`: 基于 MATR online TAL 思路的 memory transformer 传感器适配版，文件在 `src/stadstream/models/matr_signal.py`
- `moad_signal`: 基于 MOAD/Backtrace Mamba 思路的层次记忆压缩在线 baseline，文件在 `src/stadstream/models/moad_signal.py`

MATR 官方代码已拉到 `../external/MATR_codebase`，本库里的 `matr_signal` 保持传感器输入接口，便于后续替换/搬运更多官方模块。

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
