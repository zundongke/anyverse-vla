# Revo2 训练、预处理和后处理代码整理

本目录是 Anyverse-VLA 当前工作区的相关源码快照（包含未提交修改），便于阅读、交接和定位。源码保留仓库原始相对目录，未修改算法。它不是完整可独立运行的项目：完整 LeRobot、修改版 Transformers、SAM2、Python 环境、模型权重和数据集仍需原仓库及外部资源。以下运行命令均在原仓库根目录执行，不在这个快照目录执行。

## 1. 训练代码

| 文件 | 用途 |
| --- | --- |
| [DINO+SAM2 训练入口](src/core/wj_lerobot/launch_revo2_dinosam_relative_8gpu_40k.sh) | 选择轮廓遮罩数据集，动作窗口 50，开启短期记忆 |
| [Revo2 relative 配置](src/core/wj_lerobot/launch_revo2_wj_pretrain_relative_8gpu_40k.sh) | 8 GPU、40000 steps、每卡 batch 20、relative_pose、RynnBrain/WJ 预训练初始化 |
| [通用启动器](src/core/wj_lerobot/training_wjvla.sh) | 整理训练参数、分布式启动及恢复训练 |
| [Python 训练主程序](src/open_source/lerobot/src/lerobot/scripts/lerobot_train.py) | 数据加载、训练循环、优化及 checkpoint |
| [模型实现](src/open_source/lerobot/src/lerobot/policies/pi05/modeling_pi05.py) | PI05 模型、动作预测和训练损失 |
| [模型配置](src/open_source/lerobot/src/lerobot/policies/pi05/configuration_pi05.py) | 模型配置字段 |

调用顺序：DINO+SAM2 训练入口 → Revo2 relative 配置 → training_wjvla.sh → lerobot_train.py → PI05 模型。

## 2. 数据预处理：DINO + SAM2

| 文件 | 用途 |
| --- | --- |
| [DINO 转换入口](src/core/wj_lerobot/convert_ego_pico_manus_20260917_revo2.sh) | 生成基础数据和 DINO 检测缓存 |
| [DINO+SAM2 转换入口](src/core/wj_lerobot/convert_ego_pico_manus_20260917_dinosam.sh) | 复用检测缓存，选择 SAM2 后端 |
| [RRD/CSV 转 LeRobot](src/core/wj_lerobot/dataset_tools/convert_revo2_rrd_to_lerobot_dino.py) | 读取 RGB/腕位姿/手指目标、动作对齐、图像处理、写数据集和统计量 |
| [DinoMasker](src/open_source/lerobot/src/lerobot/utils/dino_masking.py) | Grounding DINO 检测和矩形遮罩 |
| [SamBoxMasker](src/open_source/lerobot/src/lerobot/utils/sam_box_masking.py) | 用检测框提示 SAM2，将分割区域涂黑 |
| [数据校验](src/core/wj_lerobot/dataset_tools/validate_revo2_training_dataset.py) | 校验转换后的训练数据 |

处理顺序：原始 RRD + retargeted CSV → DINO 框缓存 → SAM2 在原始 RGB 上分割 → 轮廓涂黑 → 等比例缩放并补边至 224×224 → LeRobot 数据集。

SAM2 分支依赖签名匹配的 DINO 缓存；首次使用需先执行 DINO 转换。训练直接读取离线处理后的图片，不在训练循环内运行 DINO/SAM2。

## 3. 数据预处理：relative

核心代码：[revo2_relative.py](src/open_source/lerobot/src/lerobot/policies/pi05/revo2_relative.py) 的 `convert_revo2_relative()`。

接入位置：[processor_pi05.py](src/open_source/lerobot/src/lerobot/policies/pi05/processor_pi05.py) 的 `Revo2RelativePoseProcessorStep` 和 `make_pi05_pre_post_processors()`。

数据文件保存绝对位姿。训练加载时，左右手分别以最后一个观测帧的位姿为基准，对历史 state 和整个 action chunk 执行 `T_relative = inverse(T_base) @ T_absolute`，然后做 MIN_MAX 归一化。手指目标保持不变。

state 是 18D（左右各 xyz + rotation_6d），action 是 30D（左右位姿 18D + 左右手指目标 12D）。转换器的 `add_relative_stats()` 在 relative 空间计算训练统计量，写入 `meta/stats.json`，绝对统计另存 `meta/absolute_stats.json`。历史长度、stride 或动作窗口改变时，需要重算相匹配的统计量。

## 4. 推理与数据后处理

| 文件 | 用途 |
| --- | --- |
| [processor_pi05.py](src/open_source/lerobot/src/lerobot/policies/pi05/processor_pi05.py) | 输出 pipeline 使用 UnnormalizerProcessorStep 做反归一化 |
| [normalize_processor.py](src/open_source/lerobot/src/lerobot/processor/normalize_processor.py) | 归一化/反归一化底层实现 |
| [revo2_relative.py](src/open_source/lerobot/src/lerobot/policies/pi05/revo2_relative.py) | restore_revo2_absolute() 将 relative 动作恢复为绝对位姿 |
| [pi05_remote_server.py](src/core/wj_lerobot/eval/pi05_remote_server.py) | 保存原始 state，并在预测后调用位姿还原，组织 action/action_chunk 返回值 |
| [action_postprocess.py](src/core/wj_lerobot/eval/action_postprocess.py) | 可选高斯平滑、chunk 截断和跨 chunk 插值；这是通用工具，当前 Revo2 默认服务未直接调用它 |
| [DINO+SAM 推理代理](src/core/wj_lerobot/eval/dinosam_pi05_proxy.py) | 在线图像遮罩代理实现，作为部署参考 |

后处理顺序：模型输出 → 反归一化 → 使用本次请求的原始 state 执行 `T_absolute[j] = T_base @ T_relative[j]` → 返回绝对 action/action_chunk。这里使用同一个观测基准，不对预测动作逐步积分。手指目标透传。末端位姿到机器人关节命令的 IK 属于后续控制端工作。

现有 [Revo2 默认服务启动器](src/core/wj_lerobot/run_revo2_pi05_remote_server.sh) 只开启 DINO 矩形遮罩，与 DINO+SAM2 数据集的轮廓遮罩不等价。代理代码不能仅凭文件存在就视为已经与当前数据预处理配置对齐；部署需核对颜色通道、检测参数和缩放补边等行为，避免重复遮罩。

## 5. 原仓库中的运行入口

```bash
cd /zundong_ke/code/code/Anyverse-VLA
# 先生成 DINO 缓存；已有匹配缓存时可跳过
bash src/core/wj_lerobot/convert_ego_pico_manus_20260917_revo2.sh
# 生成 SAM2 轮廓遮罩数据集
bash src/core/wj_lerobot/convert_ego_pico_manus_20260917_dinosam.sh
# 在具备 8 GPU、依赖和权重的环境训练
bash src/core/wj_lerobot/launch_revo2_dinosam_relative_8gpu_40k.sh
```

运行前按环境设置数据、模型和 Python 路径；转换器拒绝覆盖已有数据集目录。详细参数见 [DINO+SAM2 数据与训练说明](docs/revo2_dinosam_dataset.md) 和 [relative 服务说明](docs/revo2_relative_server.md)。这两份文档是原仓库历史说明，其中机器路径和历史运行结果不代表本次重新验证。

## 6. 测试与快照核对

随附 `tests/unit/test_revo2_relative.py`、`test_revo2_conversion.py`、`test_revo2_dino_server.py` 和 `test_sam_box_masking.py`。本次整理仅进行了逐文件内容一致性、Python AST 语法和 Shell 语法检查，没有运行训练或模型推理，也未重新执行这些单元测试。

`MANIFEST.json` 记录每个复制文件的来源相对路径、大小和 SHA-256；可用于核对快照。本目录不包含数据集、模型权重、虚拟环境和缓存。
