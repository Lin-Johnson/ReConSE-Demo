# Project Context

更新时间：2026-09-04（Asia/Shanghai）

此文件用于在不同聊天线程之间同步项目背景。新线程开始处理任务前，应先阅读本文件；完成重要修改、实验或环境配置后及时更新。

## 远程服务器

- 当前通过 cpolar SSH 隧道连接，不使用服务器内网直连：
  `ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p 8889 root@localhost`
- 服务器上的主要项目：`/yangliusha02/Project/`
- F5-TTS 项目：`/yangliusha02/F5-TTS`
- F5-TTS 主要 conda 环境：`f5-tts`

## F5-TTS 语音增强项目

总体结构：预训练 F5-TTS/DiT 主干 + Shared Recurrent ControlNet + Qwen Audio Encoder Cross-Attention Adapter。目标是从 noisy speech 生成 enhanced speech，不依赖推理时真实文本。

主要版本脉络：

- v66：无 Qwen feature loss 的基线，使用 Vocos；22 层 ControlNet/Adapter。
- v67：加入 Qwen feature consistency loss；Qwen hidden feature 经 Vocos 输出后与 clean Qwen feature 对齐，训练中加入 NaN/Inf 检查和失败记录。
- v68：基于 v66，将 Vocos 改为 BigVGAN 的实验版本。
- v70：曾尝试只对有文本样本计算 Whisper decoder cross-entropy ASR loss，后来显存和实现问题较多，不作为当前主线。
- v71：Qwen 多层 hidden-state 融合，曾使用 Qwen 层 12/18/24 的可学习 softmax 权重；之后曾改为 0--21 层 Adapter。
- v72：加入 Qwen-to-Text soft conditioning。Qwen hidden_states[18] 经过 QwenTextProjector 映射到 F5 TextEmbedding 空间，作为无文本时的 soft text condition；原 Qwen Cross-Attention Adapter 保持保留。
- v73：计划/实验加入 Temporal Conv 进行时间维度对齐；需要以服务器实际代码为准核对是否已合入。
- v74：计划将 Qwen 24 层分成 1--8、9--16、17--24 三组，组内先 LayerNorm 后求均值，再使用可学习 softmax 权重融合后输入 Adapter；需要以服务器实际代码为准。

已观察到的实验现象：

- 不提供文本时，感知质量较好但 WER 高于 noisy speech，主要瓶颈是 phonetic/semantic distortion。
- 提供正确文本是 oracle upper bound，可明显降低 WER。
- Qwen soft-text conditioning 曾使 WER 从约 0.211 降至约 0.187；lambda 0.3 比 0.1 更好，但仍需多次随机种子确认。
- Adapter 消融中跳过最后一层导致听感和 WER 明显恶化，说明最后层对声学恢复很重要。

## 数据集与评估

- 训练数据主要来自 LibriTTS、GLDNS/DNS Challenge 等混合劣化数据。
- LibriTTS 曾有数据不完整及官方失败样本过滤问题；后续补齐了 3 个 parquet，并合并旧 110,795 条与新增约 5,631 条后统一过滤官方失败样本列表。
- F5-TTS 数据通常先由 dataset/collate 读取音频，再计算 Mel；未添加 Qwen loss 时训练不需要经过 Vocos。Qwen loss 版本需要 enhanced Mel 经 Vocos 解码后再送入 Qwen。
- 常用评估脚本：`/yangliusha02/Evaluate/speech_evaluation/evaluate.py`
- URGENT 2025 数据：`/yangliusha02/datasets/URGENT2025/`
- 当前 URGENT non-blind metadata 发布包是 Kaldi 格式（`spk1.scp`、`text`、`utt2lang` 等），不包含原始 `meta.tsv`。官方 `meta.tsv` 是生成模拟数据时由 `generate_data_param.py` 写出的参数清单。
- URGENT 2025 官方来源包括 LibriVox/DNS5、LibriTTS、VCTK、WSJ、EARS、MLS、CommonVoice；噪声包括 DNS5 AudioSet/FreeSound、WHAM、FSD50K、FMA，另有 wind noise；RIR 主要来自 DNS5。

## FlowSE 环境（最近完成）

- 官方仓库：`Honee-W/FlowSE`
- 服务器代码目录：`/yangliusha02/Project/FlowSE`
- conda 环境：`flowse`，位置 `/opt/conda/envs/flowse`
- 核心版本：Python 3.10.21、PyTorch 2.2.0+cu121、Torchaudio 2.2.0+cu121、Vocos 0.1.0、librosa 0.10.1、x-transformers 1.43.2、accelerate 1.2.1。
- CUDA 检查已通过：`torch.cuda.is_available() == True`
- FlowSE 核心模块、MelSpec、DataReader、Vocos 可导入；`pip check` 无 broken requirements。
- 为兼容 librosa，setuptools 固定为 69.5.1，以保留 `pkg_resources`。
- 服务器 GitHub DNS 不稳定；代码通过本地官方 codeload/浅克隆后经 cpolar SSH 上传。服务器全局 Git 曾将 GitHub 重写到失效的 `kgithub.com`，操作时需注意。
- FlowSE 的 `config/train.yaml` 已将 Vocos `local_path` 配置为 `/yangliusha02/Model/charactr/vocos-mel-24khz`；该目录包含 `config.yaml` 和约 52 MB 的 `pytorch_model.bin`，已成功加载验证。
- FlowSE 官方 checkpoint 已下载到 `/yangliusha02/Model/flowse/wenetspeech4tts_Premium.pt.tar/wenetspeech4tts_Premium.pt.tar`，约 3.8 GB；安全加载验证通过，包含 `model_state_dict`（364 项），`epoch=10`。
- `config/train.yaml` 的推理配置已指向上述 checkpoint 目录和文件名：`checkpoint: /yangliusha02/Model/flowse/wenetspeech4tts_Premium.pt.tar`、`pt_name: wenetspeech4tts_Premium.pt.tar`。
- 2026-09-03 已使用 RTX 3090 通过官方 `infer.py` 完成 DNS Challenge 2020 synthetic 推理：`no_reverb` 和 `with_reverb` 各 150 个样本，输出分别为 `/yangliusha02/Project/FlowSE/output/no_reverb` 和 `/yangliusha02/Project/FlowSE/output/with_reverb`，两组均已生成 150 个 WAV 且无错误。临时 manifest/config/log 位于 `/tmp/flowse_dns_infer/`。
- 2026-09-03 随后使用官方无文本模式 `cond_type: wotext`（`drop_text=True`，不使用对应 `.txt` 标注）完成 DNS Challenge 2020 synthetic 推理：`no_reverb` 和 `with_reverb` 各 150 个样本，输出分别为 `/yangliusha02/Project/FlowSE/output/no_reverb_without_text` 和 `/yangliusha02/Project/FlowSE/output/with_reverb_without_text`，两组均已生成 150 个 WAV 且无错误。临时 config/log 位于 `/tmp/flowse_dns_infer/`。
- 2026-09-03 对 with_reverb 质量进行诊断：服务器上仅找到一个 FlowSE checkpoint `/yangliusha02/Model/flowse/wenetspeech4tts_Premium.pt.tar/wenetspeech4tts_Premium.pt.tar`，两组推理日志使用同一路径、同一 `epoch=10` 文件；输入 manifest、WAV 数量和 ID 均正确对应。论文明确称训练使用 OpenSLR-26/28 的 RIR 并报告 With Reverb 结果，但公开 GitHub 的 `train.yaml` 只配置 `p1:75`、`p2:25`，尽管 `dataloader.py` 含有 `p3`--`p6` 混响分支；因此不能仅凭公开配置断言 checkpoint 未训练混响，实际 checkpoint 训练配置仍需进一步复现/核验。with_reverb 的 clean 参考本身保留混响。另发现 with_reverb 的 90/150 条文本与 no_reverb 文本不一致，但无文本结果同样较差，文本差异不是主因。
- FlowSE 官方推理入口：`cd /yangliusha02/Project/FlowSE && python infer.py -conf config/train.yaml`

## PGUSE 环境（最近完成）

- 官方仓库：`hyyan2k/PGUSE`
- 服务器代码目录：`/yangliusha02/Project/pguse`
- conda 环境：`pguse`，位置 `/opt/conda/envs/pguse`
- 官方版本组合：Python 3.10.14、PyTorch 2.0.0+cu117、PyTorch Lightning 2.0.7。
- requirements.txt 依赖：`numpy`、`soundfile`、`torch_ema`、`pesq`、`pystoi`；因官方模型代码实际导入，另补装 `librosa 0.10.2.post1`。
- 为兼容 PyTorch 2.0.0 的 NumPy ABI，NumPy 固定为 1.24.4；为兼容旧版 Lightning，setuptools 固定为 80.10.2。
- 2026-09-04 使用清华 Anaconda/PyPI 镜像完成环境创建和依赖安装；`pip check`、官方 `model`/`dataloader` 导入、Model/DataModule 初始化及 CUDA 验证均已通过。GPU 为 RTX 3090。
- 2026-09-04 核对发现项目内已有 5 组、共 20 个 PGUSE Lightning checkpoint，位于 `/yangliusha02/Project/pguse/log/ckpts/`，单个约 97 MB；文件名中的验证 PESQ 分别约为 2.98、2.94、2.64、2.52、2.38。当前 `config/config.yaml` 的 `ckpt_path` 仍指向不存在的 `/yangliusha01/guoliang/...` 路径，推理前应改为 `/yangliusha02/Project/pguse/log/ckpts/...` 下的实际文件。
- 2026-09-04 使用 `/yangliusha02/Project/pguse/log/ckpts/20260428214151_dawz3w4u/epoch=64_step=2161184_val_pesq=2.64.ckpt` 完成 DNS Challenge 2020 synthetic 推理：`no_reverb/noisy` 和 `with_reverb/noisy` 各 150 条；常规最终结果读取 `/Trs=0.12_N=3_pgratio=0.4/pg/`，测试 PESQ 分别为 3.3752 和 1.3085。
- 2026-09-04 继续完成 VoiceFixer ALL_GSR 推理：输入 `/yangliusha02/datasets/voicefixer_testsets/TestSets/ALL_GSR/simulated_wav`，参考目录为同级 `target_wav`，共 503 对；常规最终结果读取 `/yangliusha02/Project/pguse/output/ALL_GSR/Trs=0.12_N=3_pgratio=0.4/pg/`，测试 PESQ 为 2.7542。此前运行产生的其他辅助输出目录不作为常规结果。
- 官方训练入口：`cd /yangliusha02/Project/pguse && python train.py --config ./config/config.yaml`
- 官方测试入口：先在 `config/config.yaml` 设置 `ckpt_path`，再运行 `python test.py --config ./config/config.yaml --save_enhanced <output_dir>`。

## FlowSE 与 PGUSE 推理方式（公共记忆）

### FlowSE

- 环境和目录：`source /opt/conda/etc/profile.d/conda.sh && conda activate flowse`，然后进入 `/yangliusha02/Project/FlowSE`。
- 官方入口：`python infer.py -conf <临时推理配置.yaml>`。不要直接复用含旧路径的配置；推理时使用临时 YAML，保留官方模型结构，只覆盖 `infer` 部分。
- checkpoint：`/yangliusha02/Model/flowse/wenetspeech4tts_Premium.pt.tar/wenetspeech4tts_Premium.pt.tar`；Vocos：`/yangliusha02/Model/charactr/vocos-mel-24khz`。
- 输入方式：FlowSE 的 `DataReader` 需要一个 JSON manifest 和音频目录。JSON 的 key 是不带 `.wav` 的音频 stem，value 是文本，例如 `{"sample_id": "transcript"}`；音频实际路径为 `<mix_dir>/<key>.wav`。
- 有文本推理：临时 YAML 中设置 `infer.test.cond_type: noisy`，manifest 中填写对应文本。
- 无文本推理：设置 `infer.test.cond_type: wotext`；代码会传入空文本并启用 `drop_text=True`，manifest 文本内容不会被使用。
- 同时设置 `infer.datareader.mix_json`、`mix_dir`、`mix_fs: 16000` 和 `infer.save.dir`，运行同一个入口即可。输出为 16 kHz WAV，文件名来自 manifest key。
- 已完成的 DNS 推理输出：有文本为 `/yangliusha02/Project/FlowSE/output/no_reverb`、`with_reverb`；无文本为 `/yangliusha02/Project/FlowSE/output/no_reverb_without_text`、`with_reverb_without_text`。每组 150 条。

### PGUSE

- 环境和目录：`source /opt/conda/etc/profile.d/conda.sh && conda activate pguse`，然后进入 `/yangliusha02/Project/pguse`。
- 官方入口：`python test.py --config <临时配置.yaml> --save_enhanced <输出根目录>`。
- 当前使用 checkpoint：`/yangliusha02/Project/pguse/log/ckpts/20260428214151_dawz3w4u/epoch=64_step=2161184_val_pesq=2.64.ckpt`。官方 `config/config.yaml` 里的旧 `/yangliusha01/guoliang/...` 路径不存在，推理时必须临时覆盖 `ckpt_path`。
- 临时配置至少覆盖：`ckpt_path`、`dataset_config.test_src_dir`、`dataset_config.test_tgt_dir`；采样率使用 16000。source 和 target 必须存在同名 WAV/FLAC，官方 DataModule 会按 stem 配对并计算 PESQ。
- 默认常规结果只读取 `<输出根目录>/Trs=0.12_N=3_pgratio=0.4/pg/`。`test.py` 会自动创建该参数目录；本项目不把其他辅助分支作为常规推理结果。
- DNS synthetic 示例：source 为 `/yangliusha02/datasets/DNSChallenge2020_test_set/synthetic/no_reverb/noisy` 或 `with_reverb/noisy`，target 为对应的 `clean`；输出根目录为 `/yangliusha02/Project/pguse/output/no_reverb` 或 `with_reverb`。
- ALL_GSR 示例：source 为 `/yangliusha02/datasets/voicefixer_testsets/TestSets/ALL_GSR/simulated_wav`，target 为 `/yangliusha02/datasets/voicefixer_testsets/TestSets/ALL_GSR/target_wav`，输出根目录为 `/yangliusha02/Project/pguse/output/ALL_GSR`。
- 如果只有 noisy、没有同名 target，官方 `test.py` 不能直接进行无参考推理，需要另写仅保存增强音频的推理入口；不能用空 target 目录替代。

## 工作约定

- 各模型的统一推理入口、checkpoint、输入输出目录和参数注意事项见项目文档 `INFERENCE_METHODS.md`。
- 评估脚本 `/yangliusha02/Evaluate/speech_evaluation/run.py` 的编辑规则也保存在 `INFERENCE_METHODS.md` 的“评估 run.py 的编辑方法”一节：多条 `python evaluate.py` 顺序执行；DNSMOS/NISQA 不需要 gt；路径由 `--enhanced_folder` 和 `--output_folder` 传入；保持使用当前镜像环境的普通 `python`，不要写死 sense 环境。

- 连接服务器优先使用 cpolar 隧道；除非用户明确改变要求。
- 用户要求“检查/分析”时先不修改；用户要求“修改/配置/运行”时才执行对应操作。
- 修改代码前先检查当前分支、工作区和实际文件，避免覆盖用户已有改动。
- 服务器项目的版本和状态以实际文件为准，本文件只作为跨线程背景摘要。
