# Revo2 DINO + SAM2 数据转换

入口：`bash src/core/wj_lerobot/convert_ego_pico_manus_20260917_dinosam.sh`

从 `original_rrd` 读取原始 RGB 图像与双腕位姿，从 `retargeted_csv`
读取手指目标。复用 `outputs/revo2_conversion/cache` 中签名匹配的 DINO
检测框，将原图交给 SAM2，以检测框作为提示，对分割出的轮廓区域涂黑。
没有检测框的帧保留原图。不会使用已经整框涂黑的图像作为 SAM2 输入。

SAM2 使用已有 tiny 权重，输入按 RGB 处理。模型预测可能存在轮廓残留或误分割；
`outputs/revo2_dinosam/comparison.png` 为首条轨迹四帧的原图、整框遮罩、轮廓遮罩对照。

默认输出：`datasets/ego_pico_manus_20260917_revo2_dinosam`。
保持原有 18 维状态、30 维动作、15 FPS、224×224 图像以及相对位姿统计逻辑。
任务文本沿用 `pick the cup into the box`。
数据保存绝对位姿，训练预处理器转换为相对位姿。

SAM2 源码及依赖来自服务器已有 codebase，副本放在
`outputs/revo2_dinosam/vendor`。换机器时需提供对应依赖并设置 `SAM2_VENDOR`，
以及通过 `SAM2_CHECKPOINT` 指定权重。DINO 检测缓存不匹配时会报错，
需要先用原来的 DINO 转换入口生成匹配的缓存。

转换中断后可复用已经完成的 SAM2 episode 缓存。输出数据集目录如果已存在，
转换器会拒绝覆盖；应检查完成状态或选择新的 `DATASET_DIR`。

训练时将 `DATASET_DIR` / `REPO_ID` 指向新的数据集。
部署时需要匹配的 DINO+SAM2 图像预处理；当前默认 Revo2 在线 `DinoMasker`
仍是矩形遮罩，不能视为本数据集的等价预处理。

## Revo2 relative 训练

100 条轨迹、3968 帧已完成转换，训练预处理抽样校验通过。
在有 8 张可用 GPU 的训练环境，从仓库根目录启动：

```bash
bash src/core/wj_lerobot/launch_revo2_dinosam_relative_8gpu_40k.sh
```

入口复用 `launch_revo2_wj_pretrain_relative_8gpu_40k.sh` 的训练参数，
默认读取 `datasets/ego_pico_manus_20260917_revo2_dinosam`，输出到
`outputs/revo2-dinosam-relative-8gpu-40k/<时间戳>_WJVLA_finetune`。
图像直接使用离线轮廓遮罩结果，训练过程中无需运行 DINO/SAM2。

| 参数 | 配置 |
| --- | --- |
| 初始化 | WJ-Pretrain / RynnBrain-Backbone-2B，060000 checkpoint |
| 动作空间 / 目标 | `revo2_eef_pose` / `relative_pose` |
| 数据维度 | state 18；action 30；手指索引 18–29 |
| 图像 | `top_head`，224×224，15 FPS |
| 历史 | 6 帧，stride 5，包含 proprio |
| 动作预测 | 50 步 |
| 并行 / batch | 8 GPU；每卡 20；全局 160（单节点） |
| 训练 / 保存 | 40000 steps；每 10000 steps 保存 |
| 学习率 | 7.5e-5；warmup 500；衰减 40000 steps 至 2.5e-6 |
| weight decay / gradient clip | 0.001 / 1.0 |
| 精度 / 显存 | bfloat16；gradient checkpointing 开启 |
| 归一化 | state/action MIN_MAX；visual IDENTITY |
| 动作头 | split heads；位姿和手指 loss 权重均 1.0 |
| 辅助损失 | future action 开启，dim 128，权重 1.0；task/box/cross-center 关闭 |
| 训练范围 | vision encoder 不冻结；不只训练 action expert；LoRA 默认关闭 |
| 图像增强 | 开启，最多 3 项 |
| WandB | 关闭 |

默认 Python 是 `.venv-revo2/bin/python`。换机器时可通过环境变量指定路径：

```bash
LEROBOT_PYTHON=/path/to/env/bin/python \
DATASET_DIR=/path/to/ego_pico_manus_20260917_revo2_dinosam \
PI05_BASE_DIR=/path/to/checkpoints/060000/pretrained_model \
RYNNBRAIN_PATH=/path/to/RynnBrain-2B \
PALIGEMMA_TOKENIZER_PATH=/path/to/paligemma-tokenizer \
OUTPUT_BASE_DIR=/path/to/revo2-dinosam-relative-8gpu-40k \
bash src/core/wj_lerobot/launch_revo2_dinosam_relative_8gpu_40k.sh
```

底层 40k 基线脚本固定了 batch、学习率、步数等参数；需要临时调整时，
用末尾 CLI 参数，例如 `--batch_size=8`，不要通过 `BATCH_SIZE` 环境变量覆盖。
更改 history 或 action horizon 时，需要同时重算匹配的相对位姿统计。

恢复该输出目录下最近一次训练：

```bash
bash src/core/wj_lerobot/launch_revo2_dinosam_relative_8gpu_40k.sh --resume
```

如使用自定义输出目录，恢复时需指定相同 `OUTPUT_BASE_DIR`。
40k 配置沿用现有实验基线；尚未验证其在这 100 条数据上的最优训练步数。
