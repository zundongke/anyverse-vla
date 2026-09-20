# Revo2 relative 推理服务

离线数据制作与在线推理共用 `lerobot.utils.dino_masking.DinoMasker`：Grounding DINO 检测后，按分数排序保留最多 4 个框，框内填黑，再用 PIL `ImageOps.pad` 等比例缩放并补黑边。默认 prompt 为 `robot arm . robot gripper .`，threshold 为 0.30，图像大小为 224，DINO 检测输入短边为 480。部署时这些参数必须与制作训练数据时使用的值一致。这里不使用 SAM 分割。

## 安装独立环境

从仓库根目录运行：

```bash
REVO2_BASE_PYTHON=/opt/python/3.11.9/bin/python3 bash src/core/wj_lerobot/setup_revo2_server_env.sh
```

默认环境位于 `.venv-revo2`。依赖包括 PyTorch 2.7.1、torchvision 0.22.1、Transformers 4.57.1、FastAPI 和 pytest。运行脚本优先导入仓库中的 LeRobot 和修改后的 Transformers。本次容器验证使用 CPU 版 PyTorch。可用 `REVO2_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu` 显式安装 CPU 版，或在 GPU 机器上设为 `https://download.pytorch.org/whl/cu126` 重新安装 CUDA 版。TorchCodec 固定为 0.3.0，与 PyTorch 2.7 对应（[官方兼容表](https://github.com/meta-pytorch/torchcodec/tree/v0.4.0#installing-torchcodec)）。安装 CUDA 版 PyTorch 不代表当前机器一定有可用 GPU；脚本会报告 `torch.cuda.is_available()`。

## 启动服务

```bash
export REVO2_MODEL_DIR=/path/to/revo2/checkpoint/pretrained_model
export REVO2_DATASET_ROOT=/path/to/matching/lerobot_dataset
export REVO2_DINO_MODEL=/path/to/grounding-dino-tiny
bash src/core/wj_lerobot/run_revo2_pi05_remote_server.sh
```

DINO 使用本地模型文件，不会在服务启动时自动联网下载。`REVO2_DEVICE` 默认 `cuda`，CPU 验证时可设为 `cpu` 并追加 `--dtype float32`。DINO 可通过 `--dino-device cpu` 单独运行在 CPU 上。首次启动必须使用对应的 Revo2 checkpoint 和数据集统计量。

服务默认监听 6007。`GET /health` 中的 `dino_mask.enabled` 应为 `true`。也可以直接启动 `pi05_remote_server.py` 并传 `--dino-model`；不传该参数时不会自动执行 DINO，适用于输入已离线处理的图像。

## 请求与返回

`POST /v1/predict` 使用 multipart/form-data：

- `cam_high`：原始头部 RGB 图片。多张图片按观测时间顺序提交，最后一张是当前帧，每张都执行 DINO mask。模型字段映射为 `observation.images.top_head`。启用 DINO 后不要重复提交已经 mask 的图像。
- `state`：JSON 数组，18 个数，排列为 `[左 xyz(3), 左 rotation_6d(6), 右 xyz(3), 右 rotation_6d(6)]`。HTTP 入口当前接受单帧 state；历史状态由模型内部配置管理。
- `task`：任务文本。

Revo2 单相机模型不需要 `wrist_left` 或 `wrist_right`；原有三相机配置仍要求相应图像。

state 在 relative 转换前保留副本，转换和归一化后送入模型。输出反归一化后，以本次请求的原始位姿为共同基准执行 `T_absolute[j] = T_base @ T_relative[j]`，不逐步积分。

`action` 和 `action_chunk` 的位姿为绝对值，`action_pose_frame` 为 `absolute`。30D 排列为 `[左位姿(9), 右位姿(9), 左手关节(6), 右手关节(6)]`。调试字段 `postprocessed` 保留反归一化后的 relative 值，并带有独立的 `action_pose_frame` 标记。输出末端位姿仍需要控制端 IK 转换为机械臂关节指令。

## 验证

```bash
export PYTHONPATH="$PWD/src/open_source/lerobot/src:$PWD/src/open_source/transformers_4.57.1"
.venv-revo2/bin/python -m pytest -q tests/unit/test_revo2_relative.py tests/unit/test_revo2_dino_server.py
```

DINO 测试注入固定检测结果，验证矩形 mask、无检测情况、与离线补边的逐像素一致性，以及单相机 HTTP 请求对全部历史帧的处理。它不替代真实 DINO 和完整 VLA checkpoint 的端到端验证。

## 选定的大脑预训练权重

Revo2 训练启动脚本默认使用：

```text
/wj-dataset/vla_pretrain_model/WJ-Pretrain/RynnBrain-Backbone-2B/20260529_100046_pi05_finetune/checkpoints/060000/pretrained_model
```

对应基础大脑目录是 `/wj-dataset/vla_pretrain_model/RynnBrain-2B`。可用 `PI05_BASE_DIR` 和 `RYNNBRAIN_PATH` 覆盖默认值。当前 CPU 环境不能运行这个 8 GPU 训练脚本；GPU 环境安装方法见上文。训练输出默认写入仓库的 `outputs/revo2-wj-pretrain-relative-8gpu-40k`。

原始预训练配置是 14D state、14D action 和三相机，不能直接作为 18D state、30D action、单相机 Revo2 relative 服务的 checkpoint。服务的 `REVO2_MODEL_DIR` 仍需指定适配并训练后的 Revo2 模型。服务脚本已默认指向本地找到的 Revo2 DINO 数据集和 DINO 权重，均可用环境变量覆盖。

## 本次 RRD 数据转换

任务文本固定为 `pick the cup into the box`。源目录为 `/zundong_ke/datasets/ego_pico_manus_2026-09-17_100`：从 `original_rrd` 读取头部 RGB 和左右腕位姿，从 `retargeted_csv` 读取左右手六个关节目标。

```bash
bash src/core/wj_lerobot/convert_ego_pico_manus_20260917_revo2.sh
```

输出为仓库内 `datasets/ego_pico_manus_20260917_revo2_dino`。100 段原始数据包含 4068 帧，下一帧动作对齐后预计保留 3968 个训练样本。缓存位于 `outputs/revo2_conversion/cache`，重复执行可复用已经完成的 DINO 结果。转换完成后生成 `meta/conversion_manifest.json`。

数据文件保存绝对位姿；`meta/stats.json` 的 state/action 统计量在相对位姿空间计算，匹配训练使用的六帧历史、间隔五帧和五十步动作窗口。原始绝对位姿统计另存为 `meta/absolute_stats.json`。

```bash
export PYTHONPATH="$PWD/src/open_source/lerobot/src:$PWD/src/open_source/transformers_4.57.1"
.venv-revo2/bin/python src/core/wj_lerobot/dataset_tools/validate_revo2_training_dataset.py --dataset "$PWD/datasets/ego_pico_manus_20260917_revo2_dino"
bash src/core/wj_lerobot/launch_revo2_wj_pretrain_relative_8gpu_40k.sh
```

训练脚本已默认使用这一新数据集。当前容器没有可见 GPU，转换使用 CPU，八卡训练需要在对应 GPU 环境启动。
