# 双头 LoRA 的使用时机

双头模型由 Qwen3-VL 基础模型、共享 LoRA 适配器和八分类头组成。生成文字时使用语言模型输出头；分类时读取最后一个提示 token 的隐藏状态，经独立分类头和 softmax 得到类别概率。分类前向不生成文字。

| 框架步骤 | 默认是否使用双头 LoRA | 实际行为 |
| --- | --- | --- |
| Agent 规划、工具选择及参数生成 | 否 | `adapter_mode=base`，临时禁用所有 LoRA |
| `llm.first_pass` 的视觉判断和描述 | 是 | `LLM_FIRST_PASS_ADAPTER=dualhead`；可显式改成 `base` |
| `llm.first_pass` 的分类概率 | 是，前提是双头模型可用 | 额外调用分类接口，用分类概率更新 `has_landslide` 和 `score`；文字 `assessment_label` 保留 |
| `cls.run` | 是，默认 `CLS_BACKEND=dualhead` | 复用缓存中的双头分类结果；默认还尝试 ConvNeXt 第二意见；双头不可用时回退 ConvNeXt |
| `vlm.describe` 的场景描述 | 是 | 默认 `LLM_DESCRIBE_ADAPTER=dualhead`；此处生成文字，不读取分类头 |
| `seg.llm_review` 的视觉复核 | 否 | 由 `LLM_VISUAL_EVIDENCE_ADAPTER` 控制，默认 `base` |
| 分割、地理查询、规则融合 | 否 | 使用分割模型、外部地理服务和规则代码 |
| 最终报告的语言生成 | 否 | 默认基础模型，依据融合后的证据组织文字 |

如果直接请求聊天接口，可以通过 `adapter_mode` 选择 `base`、`dualhead` 或 `sft`。`sft` 是 `LLM_LORA_PATH` 指定的另一个适配器，与 `LLM_DUAL_HEAD_PATH` 指定的双头适配器不同。只加载 SFT 权重不意味着默认规划和报告会使用它。

`LLM_DUAL_HEAD_ADAPTER_ALWAYS_ON=1` 是启动脚本的默认配置，用于保留常驻双头适配器状态，并省去分类时切换。它不覆盖生成请求的选择：`adapter_mode=base` 仍会在该请求内禁用 LoRA。适配器切换和推理由同一把锁串行保护，结束后恢复先前状态。部分旧注释把该开关描述为影响所有工具调用，应以当前请求作用域代码为准。

独立的 `infer_dual_head.py` 与 Agent 服务行为不同：独立脚本加载双头 LoRA 后同时用它执行分类和文字生成，不采用 Agent 的基础模型规划策略。

## 不提供权重时

本开源包仅包含推理脚本、标签和配置，不包含 `adapter_model.safetensors`、`classification_head.pt` 或基础模型。真实服务只有成功加载完整资源后才可使用双头功能；配置文件本身不能代替权重。可通过 `/health` 的 `dual_head_loaded` 检查加载状态。

双头不可用时，首轮分类概率不会覆盖文字判断，`cls.run` 尝试 ConvNeXt；默认请求双头的 `vlm.describe` 也无法正常完成。希望无双头权重运行真实模型时，可设置 `LLM_FIRST_PASS_ADAPTER=base`、`LLM_DESCRIBE_ADAPTER=base`、`LLM_VISUAL_EVIDENCE_ADAPTER=base`、`CLS_BACKEND=convnext`，并提供相应分类模型。`LLM_MOCK=1` 完全跳过真实模型加载。
