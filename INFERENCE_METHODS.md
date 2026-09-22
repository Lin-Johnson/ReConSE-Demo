# 语音增强模型推理方式

更新时间：2026-09-04（Asia/Shanghai）

本文整理当前服务器上 8 种语音增强模型的实际推理入口、模型路径、输入输出约定和评估注意事项。默认通过 cpolar SSH 连接服务器：

```bash
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -p 8889 root@localhost
```

除非特别说明，批量推理都应保持输入文件 stem 不变，输出目录单独保存增强音频，不修改原始 noisy/clean 音频。

## 统一输入音频目录

目前实际存在并用于模型推理/评估的输入目录如下：

| 数据集/用途 | 输入目录 | 当前音频数 |
|---|---|---:|
| DNS Challenge synthetic no-reverb | `/yangliusha02/datasets/DNSChallenge2020_test_set/synthetic/no_reverb/noisy` | 150 |
| DNS Challenge synthetic with-reverb | `/yangliusha02/datasets/DNSChallenge2020_test_set/synthetic/with_reverb/noisy` | 150 |
| VoiceFixer ALL_GSR | `/yangliusha02/datasets/voicefixer_testsets/TestSets/ALL_GSR/simulated_wav` | 503 |
| DNS Challenge real recordings | `/yangliusha02/datasets/DNSChallenge2020_test_set/real_recordings` | 300 |
| Comprehensive noisy | `/yangliusha02/datasets/Comprehensive/noisy` | 582 |

对应的 clean/reference 目录为：

```text
/yangliusha02/datasets/DNSChallenge2020_test_set/synthetic/no_reverb/clean
/yangliusha02/datasets/DNSChallenge2020_test_set/synthetic/with_reverb/clean
/yangliusha02/datasets/voicefixer_testsets/TestSets/ALL_GSR/target_wav
/yangliusha02/datasets/Comprehensive/clean
```

其中 `/yangliusha02/datasets/Comprehensive/clean` 只是与 `Comprehensive/noisy` 配对的 clean reference，不作为增强模型的输入目录。

`real_recordings` 没有对应的 clean reference，适合只做无参考增强或后续 ASR 评估。其 Whisper 文本已保存到：

```text
/yangliusha02/datasets/DNSChallenge2020_test_set/real_recordings_text
```

两个 synthetic noisy 目录的 ASR 文本已分别保存到：

```text
/yangliusha02/datasets/DNSChallenge2020_test_set/synthetic/no_reverb/text_noisy
/yangliusha02/datasets/DNSChallenge2020_test_set/synthetic/with_reverb/text_noisy
```

## 各模型推荐输出根目录

后续新推理尽量将结果放在对应项目目录的 `output` 下：

| 模型 | 推荐输出根目录 | 说明 |
|---|---|---|
| FlowSE | `/yangliusha02/Project/FlowSE/output` | 已采用；按数据集和有/无文本继续分子目录 |
| PGUSE | `/yangliusha02/Project/pguse/output` | 已采用；官方还会在内部创建参数子目录 |
| MP-SENet | `/yangliusha02/Project/MP-SENet/output` | 旧结果在 `/yangliusha02/Evaluate/MP-SENet`，新结果建议改放这里 |
| TF-GridNet | `/yangliusha02/Project/TFGridNet/output` | 已采用 |
| AnyEnhance | `/yangliusha02/Project/Amphion_AnyEnhance/output` | 原始大模型建议使用这里 |
| UniPASE | `/yangliusha02/Project/unipase/output` | 新推理建议使用这里 |
| SenSE | `/yangliusha02/Project/SenSE-main/output` | 官方脚本默认写入 `results`，若不改代码则仍按官方目录保存 |

建议的子目录命名方式为：

```text
<model>/output/DNS_no_reverb
<model>/output/DNS_with_reverb
<model>/output/voicefixer_gsr
<model>/output/DNS_real_recordings
<model>/output/Comprehensive_clean
<model>/output/Comprehensive_noisy
```

## 1. FlowSE

服务器项目和环境：

```text
项目：/yangliusha02/Project/FlowSE
环境：flowse
入口：infer.py
checkpoint：/yangliusha02/Model/flowse/wenetspeech4tts_Premium.pt.tar/wenetspeech4tts_Premium.pt.tar
Vocos：/yangliusha02/Model/charactr/vocos-mel-24khz
```

FlowSE 不是直接读取一个输入目录，而是需要临时 YAML 和 JSON manifest：

```json
{
  "136": "corresponding transcript"
}
```

JSON key 必须是不带 `.wav` 的音频 stem，实际音频路径为：

```text
<mix_dir>/<key>.wav
```

有文本推理时设置：

```yaml
infer:
  test:
    cond_type: noisy
```

无文本推理时设置：

```yaml
infer:
  test:
    cond_type: wotext
```

`wotext` 会启用 `drop_text=True`，manifest 中的文本不会被使用。

运行形式：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate flowse
cd /yangliusha02/Project/FlowSE
python infer.py -conf /tmp/flowse_dns_infer/<temporary_config>.yaml
```

临时 YAML 至少需要覆盖 checkpoint、Vocos、本次输入 manifest、音频目录、采样率和输出目录。当前已完成的 DNS 输出为：

```text
/yangliusha02/Project/FlowSE/output/no_reverb
/yangliusha02/Project/FlowSE/output/with_reverb
/yangliusha02/Project/FlowSE/output/no_reverb_without_text
/yangliusha02/Project/FlowSE/output/with_reverb_without_text
```

FlowSE 输出通常为 16 kHz WAV。论文/方法背景见 [FlowSE](https://arxiv.org/abs/2505.19476)。

## 2. PGUSE

服务器项目和环境：

```text
项目：/yangliusha02/Project/pguse
环境：pguse
入口：test.py
checkpoint：/yangliusha02/Project/pguse/log/ckpts/20260428214151_dawz3w4u/epoch=64_step=2161184_val_pesq=2.64.ckpt
```

PGUSE 的官方测试入口需要 noisy/source 和 clean/target 同名配对，因为测试流程会计算 PESQ 等指标。不要直接修改官方 `config/config.yaml` 中已经失效的旧 checkpoint 路径，通常使用临时配置覆盖：

```yaml
ckpt_path: /yangliusha02/Project/pguse/log/ckpts/20260428214151_dawz3w4u/epoch=64_step=2161184_val_pesq=2.64.ckpt
dataset_config:
  test_src_dir: <noisy_dir>
  test_tgt_dir: <clean_dir>
```

运行形式：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate pguse
cd /yangliusha02/Project/pguse
python test.py --config /tmp/pguse_<dataset>.yaml --save_enhanced <output_root>
```

增强音频通常位于：

```text
<output_root>/Trs=0.12_N=3_pgratio=0.4/pg/
```

当前已使用的输入对应关系：

```text
DNS no-reverb：synthetic/no_reverb/noisy  ↔  synthetic/no_reverb/clean
DNS with-reverb：synthetic/with_reverb/noisy  ↔  synthetic/with_reverb/clean
VoiceFixer：ALL_GSR/simulated_wav  ↔  ALL_GSR/target_wav
```

如果只有 noisy、没有 clean target，官方 `test.py` 不能直接作为无参考批量增强入口，需要单独改写仅保存增强音频的推理入口。

## 3. MP-SENet

服务器项目和环境：

```text
项目：/yangliusha02/Project/MP-SENet
环境：f5-tts
入口：inference.py
checkpoint：best_ckpt/g_dns_1000h_MutiLoss
```

MP-SENet 的 `inference.py` 支持直接输入一个音频目录，不需要 manifest、文本或 clean reference：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate f5-tts
cd /yangliusha02/Project/MP-SENet

python inference.py \
  --input_noisy_wavs_dir <noisy_dir> \
  --output_dir <output_dir> \
  --checkpoint_file best_ckpt/g_dns_1000h_MutiLoss
```

当前批量脚本 `/yangliusha02/Project/MP-SENet/run.py` 使用的三个数据集是：

```text
/yangliusha02/datasets/DNSChallenge2020_test_set/synthetic/no_reverb/noisy
/yangliusha02/datasets/DNSChallenge2020_test_set/synthetic/with_reverb/noisy
/yangliusha02/datasets/voicefixer_testsets/TestSets/ALL_GSR/simulated_wav
```

历史脚本曾将结果写到 `/yangliusha02/Evaluate/MP-SENet`；后续建议改为项目目录下的 `output`：

```text
/yangliusha02/Project/MP-SENet/output/DNS_no_reverb
/yangliusha02/Project/MP-SENet/output/DNS_with_reverb
/yangliusha02/Project/MP-SENet/output/voicefixer_gsr
```

## 4. TF-GridNet

当前服务器上的实际项目目录是：

```text
/yangliusha02/Project/TFGridNet
```

批量自定义目录推理使用：

```text
/yangliusha02/Project/TFGridNet/enhance_test.py
```

该脚本优先加载本地模型：

```text
/yangliusha02/Project/TFGridNet/exp/tfgridnet_urgent25/exp/enh_train_enh_tfgridnet_dm_raw/config.yaml
/yangliusha02/Project/TFGridNet/exp/tfgridnet_urgent25/exp/enh_train_enh_tfgridnet_dm_raw/valid.loss.ave_5best.pth
```

如果本地文件不存在，则回退到 Hugging Face 模型：

```text
kohei0209/tfgridnet_urgent25
```

推理命令：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate urgent2025
cd /yangliusha02/Project/TFGridNet
python -P enhance_test.py \
  --noisy_dir <noisy_dir> \
  --output_dir <output_dir> \
  --device cuda
```

如服务器当前没有名为 `urgent2025` 的环境，应使用已经安装 ESPnet 的环境，并先确认：

```bash
python -P -c "import espnet2.bin.enh_inference; print('ESPnet OK')"
```

长音频可能需要分段：

```bash
python -P enhance_test.py \
  --noisy_dir <noisy_dir> \
  --output_dir <output_dir> \
  --device cuda \
  --segment_size 20 \
  --hop_size 10
```

当前已使用的输出目录为：

```text
/yangliusha02/Project/TFGridNet/output/no_reverb
/yangliusha02/Project/TFGridNet/output/with_reverb
/yangliusha02/Project/TFGridNet/output/voicefixer_gsr
```

如果使用完整 ESPnet recipe，而不是自定义目录脚本，则通过 `run.sh` 的 stage 7 进行推理；官方 `scoring.sh` 还会继续计算 DNSMOS、NISQA、UTMOS、侵入式指标、SpeechBERTScore、说话人相似度和 CER。

## 5. AnyEnhance 原始大模型

当前应区分原始大模型和轻量化 `AnyEnhance-v1`。之前评估使用的是原始 Amphion 实现：

```text
项目：/yangliusha02/Project/Amphion_AnyEnhance
入口：models/se/anyenhance/infer_anyenhance.py
环境：anyenhance
```

原始模型文件：

```text
/yangliusha02/Model/Amphion/anyenhance/epoch-1-step-300000-loss-4.3083/model.pt
/yangliusha02/Model/Amphion/anyenhance/dac/weights.pth
/yangliusha02/Model/Amphion/anyenhance/anyenhance-360M-selfcritic-v2.json
```

推理入口是单文件模式，批量处理需要由 shell 循环调用：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate anyenhance
cd /yangliusha02/Project/Amphion_AnyEnhance

python -m models.se.anyenhance.infer_anyenhance \
  --task_type enhancement \
  --input_file <noisy_wav> \
  --output_folder <output_dir> \
  --device cuda:0 \
  --timesteps 20 \
  --cond_scale 1
```

无 prompt 的 enhancement 不需要真实文本或 clean reference；有 prompt 的 enhancement/extraction 才额外使用 `--prompt_file`。

当前原始配置中：

```json
"self_critic": true,
"critic_v2": true,
"use_noisy_audio_embed": true
```

因此推理中会使用 DAC，并会加载冻结的 Wav2Vec-BERT 2.0 semantic encoder。论文中约 360M 指的是 AnyEnhance 核心网络；DAC 和外部语义编码器是独立冻结组件，未计入该模型规模。相关论文：[AnyEnhance](https://arxiv.org/abs/2501.15417)。

## 6. UniPASE

服务器项目和环境：

```text
项目：/yangliusha02/Project/unipase
环境：unipase
入口：python -m inference.inference
```

当前本地 checkpoint：

```text
/yangliusha02/Model/Xiaobin-Rong/unipase/DeWavLM-Omni.pt
/yangliusha02/Model/Xiaobin-Rong/unipase/Adapter.pt
/yangliusha02/Model/Xiaobin-Rong/unipase/Vocoder_DWO-L1.pt
/yangliusha02/Model/Xiaobin-Rong/unipase/PostNet.pt
```

常规短音频推理：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate unipase
cd /yangliusha02/Project/unipase

python -m inference.inference \
  -I <input_dir> \
  -O <output_dir> \
  -D cuda:0 \
  -E .wav \
  --dewavlm_ckpt /yangliusha02/Model/Xiaobin-Rong/unipase/DeWavLM-Omni.pt \
  --adapter_ckpt /yangliusha02/Model/Xiaobin-Rong/unipase/Adapter.pt \
  --vocoder_ckpt /yangliusha02/Model/Xiaobin-Rong/unipase/Vocoder_DWO-L1.pt \
  --postnet_ckpt /yangliusha02/Model/Xiaobin-Rong/unipase/PostNet.pt
```

长音频（例如超过约 20 秒）使用：

```bash
python -m inference.inference_long -I <input_dir> -O <output_dir> -D cuda:0
```

`--sr_out` 不指定时保持输入采样率；指定高于 16 kHz 的输出时会经过 PostNet 做带宽扩展。`Vocoder_WavLM-L24.pt` 不是最终 UniPASE 推理管线的一部分，而是用于重建 WavLM L24 表征、验证 DeWavLM 的辅助 checkpoint。论文：[UniPASE](https://arxiv.org/abs/2604.14606)。

## 7. SenSE Base

服务器项目和环境：

```text
项目：/yangliusha02/Project/SenSE-main
环境：sense
入口：src/sense/eval/eval_infer_batch.py
```

checkpoint：

```text
/yangliusha02/Model/ASLP-lab/SenSE/SenSE_LLM.safetensors
/yangliusha02/Model/ASLP-lab/SenSE/SenSE_CFM.safetensors
```

SenSE 的批量推理入口使用 `custom` 测试集，并通过 `--test_dir` 指定输入目录；它不是直接传入 `--output_dir`，输出目录由模型名、测试集名、seed、采样参数自动组织：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate sense
cd /yangliusha02/Project/SenSE-main

/opt/conda/envs/sense/bin/accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision no \
  src/sense/eval/eval_infer_batch.py \
  --seed 0 \
  --llm_model SenSE_LLM_Base \
  --llm_ckpt_file /yangliusha02/Model/ASLP-lab/SenSE/SenSE_LLM.safetensors \
  --fm_model SenSE_CFM_Base \
  --fm_ckpt_file /yangliusha02/Model/ASLP-lab/SenSE/SenSE_CFM.safetensors \
  --exp_name <experiment_name> \
  --save_sample_rate 24000 \
  --testset custom \
  --test_dir <input_dir> \
  --no_ref_audio \
  --nfestep 8 \
  --cfg_strength 0.5 \
  --swaysampling -1
```

无参考 enhancement 使用 `--no_ref_audio`；不需要真实 transcript。当前项目已有五 seed 循环脚本 `/Users/js/Documents/Controlnet-Enhanced/run_sense_5seeds.sh`，输出类似：

```text
/yangliusha02/Project/SenSE-main/results/SenSE_LLM_Base_SenSE_CFM_Base_<experiment_name>/custom/seed0_...
```

SenSE 的 LLM 内部使用冻结的 Whisper encoder 提取 noisy speech 的语义表示，然后由 speech LM 和 CFM/BigVGAN 生成增强语音；这不是外部 ASR 解码流程。论文：[SenSE](https://arxiv.org/abs/2509.24708)。

## 8. F5-TTS ControlNet（我们的模型）

服务器项目和环境：

```text
项目根目录：/yangliusha02/F5-TTS
环境：f5-tts
推理入口：src/f5_tts/infer/infer_dir.py
模型名：F5TTS_v1_Controlnet_A800_DNS
```

切换分支前必须先保存当前改动：

```bash
cd /yangliusha02/F5-TTS
git add .
git commit -m "update"
git switch <branch>
```

当前使用的 checkpoint：

```text
v66：/yangliusha02/F5-TTS/ckpts/F5TTS_Controlnet_GLDNS_LibriTTS_v66_c_vocos_custom_GLDNS_LibriTTS_controlnet/model_last.pt
v72：/yangliusha02/F5-TTS/ckpts/F5TTS_Controlnet_GLDNS_LibriTTS_v72_b_vocos_custom_GLDNS_LibriTTS_controlnet/model_last.pt
```

每次推理前必须核对三项结构：

1. YAML 中的 `control_layers`；
2. `src/f5_tts/model/control_f5.py` 中的 `adapter_layers`；
3. checkpoint 中的 ControlNet 和 Adapter 层编号。

不一致时必须中止推理。当前 checkpoint 结构为 ControlNet 21 层（编号 0–20）和 Adapter 22 层（编号 0–21）。v72 的 `adapter_layers` 必须为：

```python
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
```

v66 支持两种推理方式。无文本推理不添加 `--text_dir`：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate f5-tts
cd /yangliusha02/F5-TTS

python src/f5_tts/infer/infer_dir.py \
  --model F5TTS_v1_Controlnet_A800_DNS \
  --control_audio_dir <input_dir> \
  --ckpt_file <v66_checkpoint> \
  --output_dir <output_dir>
```

有文本推理额外添加 `--text_dir`，文本目录中的文件 stem 必须与输入音频 stem 对齐：

```bash
python src/f5_tts/infer/infer_dir.py \
  --model F5TTS_v1_Controlnet_A800_DNS \
  --control_audio_dir <input_dir> \
  --ckpt_file <v66_checkpoint> \
  --output_dir <output_dir> \
  --text_dir <text_dir>
```

DNS real recordings 当前使用的目录：

```text
输入：/yangliusha02/datasets/DNSChallenge2020_test_set/real_recordings
文本：/yangliusha02/datasets/DNSChallenge2020_test_set/real_recordings_text
```

已保存的 F5-TTS 输出：

```text
/yangliusha02/Evaluate/Model/Ours/output/DNS_real_recordings_without_text
/yangliusha02/Evaluate/Model/Ours/output/DNS_real_recordings_with_text_noisy
/yangliusha02/Evaluate/Model/Ours/output/DNS_real_recordings_v72_without_text
```

v72 默认无文本且不能提供文本，因此只能使用不带 `--text_dir` 的命令；当前 v72 已完成 300 个 real recordings 的无文本推理。F5-TTS 的 `output` 只保存增强语音，`result` 保存评估指标；本推理记录不负责运行评估。

## 9. 统一评估注意事项

- DNS synthetic 的 no-reverb、with-reverb 和 VoiceFixer GSR 应分别保存输出，不能混合命名。
- 需要计算 PESQ、STOI 等 intrusive 指标时，必须保留同名 clean reference。
- 只计算 WER/CER 时，可使用预先准备好的 clean text，避免重复对 clean 音频进行 ASR。
- Whisper large-v3 评估环境当前使用 `/opt/conda/envs/sense/bin/python` 和 `/yangliusha02/Evaluate/speech_evaluation/whisper/large-v3.pt`；如果只做推理，不要同时启动多个 large-v3 进程，以免显存重复占用。
- AnyEnhance、UniPASE、SenSE 的“模型参数量”与完整推理依赖的参数量口径不同；比较时应同时注明核心 checkpoint 和外部冻结 encoder/vocoder。

## 10. 评估 run.py 的编辑方法

统一评估调用脚本位于服务器：

```text
/yangliusha02/Evaluate/speech_evaluation/run.py
```

该文件通过 `full_command` 中的多条 `python evaluate.py` 命令依次评估多个增强结果。每个模型/数据集使用一条独立命令，并且必须使用不同的 `--output_folder`：

```python
full_command = """
cd /yangliusha02/Evaluate/speech_evaluation

python evaluate.py \\
    --enhanced_folder <enhanced_audio_dir> \\
    --output_folder <metrics_output_dir> \\
    --gt_folder "" \\
    --dnsmos --nisqa
"""
```

编辑时遵循以下规则：

1. `--enhanced_folder` 指向当前模型的增强音频目录。
2. `--output_folder` 指向 `/yangliusha02/Evaluate/Model/<ModelName>/...` 下的独立指标目录，不能让多个实验共用同一目录。
3. 只计算 DNSMOS/NISQA 时不需要 clean reference，可以省略 `--gt_folder`；当前脚本为避免误读默认路径，使用 `--gt_folder ""`。
4. 计算 WER、SpeechBERTScore、similarity、intrusive 或 common 时，必须加入真实的 `--gt_folder`，必要时再加入 `--gt_text_dir`。
5. 多条命令在同一个 `full_command` 中是顺序执行，不是并行执行；这样可以减少多个评估模型同时占用 GPU。
6. `run.py` 内部使用普通的 `python evaluate.py`，不要写死 `/opt/conda/envs/sense/bin/python`。运行者应先激活包含评估依赖的目标镜像环境，再执行：

```bash
python /yangliusha02/Evaluate/speech_evaluation/run.py
```

修改后可先检查：

```bash
sed -n '1,180p' /yangliusha02/Evaluate/speech_evaluation/run.py
```

确认每组输入、输出和指标选项正确后，再由用户手动运行评估。当前 real recording 的四组增强目录和指标输出目录已写入服务器 `run.py`，详见上文各模型的路径说明。
