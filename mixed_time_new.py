#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频退化仿真工具 - 完整版（集成 SC-Wind-Noise-Generator）
支持：
  1. 单一退化模式：bandwidth / clipping / reverb
  2. 组合退化模式：full
  3. 风噪生成：支持从原始风声录音生成风噪（调用 SC-Wind-Noise-Generator）
  4. 时长控制：可指定输出总时长

用法：
    # 使用预生成风噪文件
    python audio_degradation_tool.py full \
        --speech_scp speech.scp \
        --noise_scp noise.scp \
        --wind_noise_scp wind.scp \
        --rir_scp rir.scp \
        --dst_dir output/ \
        --target_hours 1000
    
    # 或使用 SC-Wind-Noise-Generator 实时生成风噪（无需预准备风噪文件）
    python audio_degradation_tool.py full \
        --speech_scp speech.scp \
        --noise_scp noise.scp \
        --raw_wind_scp raw_wind_recordings.scp \
        --use_wind_generator \
        --rir_scp rir.scp \
        --dst_dir output/
"""

import sys
import os
import io
import random
import argparse
import math
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy
import scipy.signal
import librosa
import soundfile as sf
from tqdm import tqdm

import torch
import torchaudio
from torchaudio.io import AudioEffector, CodecConfig

# ==================== 尝试导入 SC-Wind-Noise-Generator ====================
try:
    # 假设 SC-Wind-Noise-Generator 文件夹在同级目录
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SC-Wind-Noise-Generator'))
    from sc_wind_noise_generator import WindNoiseGenerator
    SC_WIND_GENERATOR_AVAILABLE = True
except ImportError:
    SC_WIND_GENERATOR_AVAILABLE = False
    WindNoiseGenerator = None

# ==================== 基础工具函数 ====================

def framing(
    x,
    frame_length: int = 512,
    frame_shift: int = 256,
    centered: bool = True,
    padded: bool = True,
):
    """将音频分帧"""
    if x.size == 0:
        raise ValueError("Input array size is zero")
    if frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if frame_length > x.shape[-1]:
        raise ValueError("frame_length is greater than input length")
    if 0 >= frame_shift:
        raise ValueError("frame_shift must be greater than 0")

    if centered:
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
            (frame_length // 2, frame_length // 2)
        ]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if padded:
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., None]
    else:
        shape = x.shape[:-1] + (
            (x.shape[-1] - frame_length) // frame_shift + 1,
            frame_length,
        )
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result


def detect_non_silence(
    x: np.ndarray,
    threshold: float = 0.01,
    frame_length: int = 1024,
    frame_shift: int = 512,
    window: str = "boxcar",
) -> np.ndarray:
    """基于功率的语音活动检测（与代码1一致）"""
    if x.shape[-1] < frame_length:
        return np.full(x.shape, fill_value=True, dtype=np.bool)

    if x.dtype.kind == "i":
        x = x.astype(np.float64)
    
    framed_w = framing(
        x,
        frame_length=frame_length,
        frame_shift=frame_shift,
        centered=False,
        padded=True,
    )
    framed_w *= scipy.signal.get_window(window, frame_length).astype(framed_w.dtype)
    power = (framed_w**2).mean(axis=-1)
    mean_power = np.mean(power, axis=-1, keepdims=True)
    
    if np.all(mean_power == 0):
        return np.full(x.shape, fill_value=True, dtype=bool)
    
    detect_frames = power / mean_power > threshold
    detects = np.broadcast_to(
        detect_frames[..., None], detect_frames.shape + (frame_shift,)
    )
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    
    return np.pad(
        detects,
        [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
        mode="edge",
    )


def read_audio(filename, force_1ch=False, fs=None):
    """读取音频文件"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Audio file {filename} does not exist.")
    
    audio_bytes = io.BytesIO(open(filename, "rb").read())
    audio, fs_ = sf.read(audio_bytes, always_2d=True)
    
    if force_1ch:
        audio = audio.mean(axis=-1, keepdims=True)
    audio = audio.T
    
    if fs is not None and fs != fs_:
        audio = librosa.resample(audio, orig_sr=fs_, target_sr=fs, res_type="soxr_hq")
        return audio, fs
    return audio, fs_


def save_audio(audio, filename, fs):
    """保存音频文件"""
    if audio.ndim != 1:
        audio = audio[0] if audio.shape[0] == 1 else audio.T
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sf.write(filename, audio, samplerate=fs)


def read_scp(path):
    """读取scp文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scp file not found: {path}")
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        # 解析 scp 格式：uid fs path 或 uid path
        result = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                result.append(parts[2])  # 取 path 部分
            elif len(parts) == 2:
                result.append(parts[1])
            else:
                result.append(parts[0])
        return result


def get_audio_duration(filename):
    """获取音频文件时长（秒）"""
    try:
        info = sf.info(filename)
        return info.duration
    except:
        return 0


# ==================== RIR 工具（替代 rir_utils.py）====================

def estimate_early_rir(rir_sample, fs, window_ms=10):
    """
    估计早期房间脉冲响应（与代码1一致）
    提取直达声+早期反射（通常前5-10ms）
    """
    delay_idx = np.argmax(np.abs(rir_sample[0]))
    delay_before = int(0.001 * fs)  # 1ms before peak
    delay_after = int(0.005 * fs)   # 5ms after peak
    
    idx_start = max(0, delay_idx - delay_before)
    idx_end = delay_idx + delay_after
    
    early_rir = np.zeros_like(rir_sample)
    early_rir[:, idx_start:idx_end] = rir_sample[:, idx_start:idx_end]
    return early_rir


# ==================== FFmpeg 风噪处理（与代码1完全一致）====================

def buildFFmpegCommand(params):
    """构建FFmpeg sidechaincompress命令（与代码1完全一致）"""
    filter_commands = ""
    filter_commands += "[1:a]asplit=2[sc][mix];"
    filter_commands += (
        "[0:a][sc]sidechaincompress="
        + f"threshold={params['threshold']}:"
        + f"ratio={params['ratio']}:"
        + f"level_sc={params['sc_gain']}"
        + f":release={params['release']}"
        + f":attack={params['attack']}"
        + "[compr];"
    )
    filter_commands += "[compr][mix]amix"

    commands_list = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "quiet",
        "-i",
        params["speech_path"],
        "-i",
        params["noise_path"],
        "-filter_complex",
        filter_commands,
        params["output_path"],
    ]

    return commands_list


def apply_wind_noise_ffmpeg(
    speech_sample,
    noise_sample,
    fs,
    uid,
    threshold,
    ratio,
    attack,
    release,
    sc_gain,
    clipping,
    clipping_threshold,
    snr,
    rng=None,
    on_the_fly=True
):
    """
    使用FFmpeg sidechaincompress的风噪处理（与代码1完全一致）
    """
    len_speech = speech_sample.shape[-1]
    len_noise = noise_sample.shape[-1]
    
    if rng is None:
        rng = np.random.default_rng()
    
    # 长度匹配（与代码1逻辑一致）
    if len_noise < len_speech:
        offset = rng.integers(0, len_speech - len_noise)
        noise_sample = np.pad(
            noise_sample,
            [(0, 0), (offset, len_speech - len_noise - offset)],
            mode="wrap",
        )
    elif len_noise > len_speech:
        offset = rng.integers(0, len_noise - len_speech)
        noise_sample = noise_sample[:, offset : offset + len_speech]

    # 计算SNR缩放（与代码1一致）
    power_speech = (speech_sample[detect_non_silence(speech_sample)] ** 2).mean()
    power_noise = (noise_sample[detect_non_silence(noise_sample)] ** 2).mean()
    scale = 10 ** (-snr / 20) * np.sqrt(power_speech) / np.sqrt(max(power_noise, 1e-10))
    noise = scale * noise_sample

    # 创建临时目录
    if on_the_fly:
        tmp_dir = Path(tempfile.gettempdir()) / "simulation_tmp"
    else:
        tmp_dir = Path("./simulation_tmp")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    
    speech_tmp_path = tmp_dir / f"speech_{uid}.wav"
    noise_tmp_path = tmp_dir / f"noise_{uid}.wav"
    mix_tmp_path = tmp_dir / f"mix_{uid}.wav"

    # 缩放防止削波（与代码1一致）
    norm_scale = 0.9 / max(
        np.max(np.abs(speech_sample)),
        np.max(np.abs(noise)),
        1e-8
    )
    speech_sample_scaled = speech_sample * norm_scale
    noise_scaled = noise * norm_scale

    # 保存临时文件
    sf.write(speech_tmp_path, speech_sample_scaled.T, fs)
    sf.write(noise_tmp_path, noise_scaled.T, fs)

    # 构建并执行FFmpeg命令
    commands = buildFFmpegCommand(
        {
            "speech_path": str(speech_tmp_path),
            "noise_path": str(noise_tmp_path),
            "output_path": str(mix_tmp_path),
            "threshold": threshold,
            "ratio": ratio,
            "attack": attack,
            "release": release,
            "sc_gain": sc_gain,
        }
    )

    try:
        result = subprocess.run(commands, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            # 如果失败，返回原语音+噪声的简单混合
            return speech_sample + noise
    except Exception as e:
        print(f"FFmpeg execution error: {e}")
        return speech_sample + noise

    # 读取混合结果
    mix, sr = sf.read(mix_tmp_path)
    if mix.ndim == 1:
        mix = mix[np.newaxis, :]
    else:
        mix = mix.T
        
    # 重新缩放回去
    mix = mix / norm_scale

    # 清理临时文件
    if on_the_fly:
        try:
            os.remove(speech_tmp_path)
            os.remove(noise_tmp_path)
            os.remove(mix_tmp_path)
        except:
            pass

    # 应用削波（与代码1一致）
    if clipping:
        mix = np.maximum(clipping_threshold * np.min(mix) * np.ones_like(mix), mix)
        mix = np.minimum(clipping_threshold * np.max(mix) * np.ones_like(mix), mix)

    return mix


# ==================== SC-Wind-Noise-Generator 集成 ====================

class WindNoiseGeneratorWrapper:
    """
    包装 SC-Wind-Noise-Generator，提供统一接口
    如果外部生成器不可用，使用备用简单生成
    """
    
    def __init__(self):
        self.generator = None
        if SC_WIND_GENERATOR_AVAILABLE and WindNoiseGenerator is not None:
            try:
                self.generator = WindNoiseGenerator()
                print("SC-Wind-Noise-Generator loaded successfully")
            except Exception as e:
                print(f"Failed to load SC-Wind-Noise-Generator: {e}")
                self.generator = None
    
    def generate(self, audio_path, fs=16000):
        """
        从原始风声录音生成处理后的风噪
        如果生成器不可用，返回原始音频（假设已是风噪）
        """
        if self.generator is not None:
            try:
                # 假设生成器有 process 或 generate 方法
                # 根据实际情况调整调用方式
                if hasattr(self.generator, 'process'):
                    return self.generator.process(audio_path, fs=fs)
                elif hasattr(self.generator, 'generate'):
                    return self.generator.generate(audio_path, fs=fs)
                else:
                    # 直接调用实例
                    return self.generator(audio_path, fs=fs)
            except Exception as e:
                print(f"Wind noise generation failed: {e}, using original audio")
                audio, _ = read_audio(audio_path, force_1ch=True, fs=fs)
                return audio
        else:
            # 备用：直接读取（假设输入已是风噪）
            audio, _ = read_audio(audio_path, force_1ch=True, fs=fs)
            return audio


# ==================== 退化算法函数 ====================

def add_noise(speech_sample, noise_sample, snr=5.0, rng=None):
    """按给定SNR混合噪声"""
    if rng is None:
        rng = np.random.default_rng()
    
    len_speech = speech_sample.shape[-1]
    len_noise = noise_sample.shape[-1]
    
    if len_noise < len_speech:
        offset = rng.integers(0, len_speech - len_noise)
        noise_sample = np.pad(
            noise_sample,
            [(0, 0), (offset, len_speech - len_noise - offset)],
            mode="wrap",
        )
    elif len_noise > len_speech:
        offset = rng.integers(0, len_noise - len_speech)
        noise_sample = noise_sample[:, offset:offset + len_speech]

    non_silence_indices_speech = detect_non_silence(speech_sample)
    non_silence_indices_noise = detect_non_silence(noise_sample)

    if len(non_silence_indices_noise) == 0:
        return speech_sample, speech_sample
    
    power_speech = (speech_sample[non_silence_indices_speech] ** 2).mean()
    power_noise = (noise_sample[non_silence_indices_noise] ** 2).mean()
    
    scale = 10 ** (-snr / 20) * np.sqrt(power_speech) / np.sqrt(max(power_noise, 1e-10))
    noise = scale * noise_sample
    noisy_speech = speech_sample + noise
    
    return noisy_speech, noise


def add_reverberation_v2(speech_sample, noisy_speech, rir_sample, fs):
    """
    添加混响（支持早期反射分离，与代码1一致）
    """
    wav_len = speech_sample.shape[1]
    
    # 计算延迟
    delay_idx = np.argmax(np.abs(rir_sample[0]))
    delay_before_num = int(0.001 * fs)
    delay_after_num = int(0.005 * fs)
    
    idx_start = max(0, delay_idx - delay_before_num)
    
    # 分离早期反射
    early_rir = estimate_early_rir(rir_sample, fs)
    
    # 卷积
    reverbant_speech_early = scipy.signal.fftconvolve(speech_sample, early_rir, mode="full")
    reverbant_speech = scipy.signal.fftconvolve(noisy_speech, rir_sample, mode="full")
    
    # 对齐长度
    reverbant_speech = reverbant_speech[:, idx_start:idx_start + wav_len]
    reverbant_speech_early = reverbant_speech_early[:, :wav_len]
    
    # 归一化
    scale = max(abs(reverbant_speech[0].max()), abs(reverbant_speech[0].min()))
    scale = 1.0 if scale == 0 else 0.5 / scale
    
    return reverbant_speech * scale, reverbant_speech_early * scale


def bandwidth_limitation(speech_sample, fs: int, fs_new: int, res_type="kaiser_best"):
    """应用带宽限制"""
    if fs == fs_new:
        return speech_sample
    
    ret = librosa.resample(speech_sample, orig_sr=fs, target_sr=fs_new, res_type=res_type)
    ret = librosa.resample(ret, orig_sr=fs_new, target_sr=fs, res_type=res_type)
    return ret[:, :speech_sample.shape[1]]


def clipping(speech_sample, min_quantile: float = 0.06, max_quantile: float = 0.9):
    """基础削波"""
    max_amp = np.max(np.abs(speech_sample))
    if max_amp < 1e-8:
        return speech_sample
    
    threshold = random.uniform(min_quantile, max_quantile) * max_amp
    return np.clip(speech_sample, -threshold, threshold)


def clipping_hard(speech_sample, min_quantile=0.1, max_quantile=0.5):
    """硬削波"""
    max_amp = np.max(np.abs(speech_sample))
    if max_amp < 1e-8:
        return speech_sample
    q = random.uniform(min_quantile, max_quantile)
    return np.clip(speech_sample, -q * max_amp, q * max_amp)


def clipping_soft(speech_sample, min_quantile=0.1, max_quantile=0.5):
    """软削波"""
    max_amp = np.max(np.abs(speech_sample))
    if max_amp < 1e-8:
        return speech_sample
    q = random.uniform(min_quantile, max_quantile)
    threshold = q * max_amp
    return np.tanh(speech_sample / threshold) * threshold


def clipping_with_gain(speech_sample, min_quantile=0.1, max_quantile=0.3, 
                       gain_range=(3, 8), normalize=True):
    """增益后削波"""
    gain = random.uniform(*gain_range)
    amplified = speech_sample * gain
    
    max_amp = np.max(np.abs(amplified))
    if max_amp < 1e-8:
        return speech_sample
    
    q = random.uniform(min_quantile, max_quantile)
    threshold = q * max_amp
    clipped = np.clip(amplified, -threshold, threshold)
    
    if normalize:
        scale = 1.0 / max(np.max(np.abs(clipped)), 1e-8)
        clipped *= scale * 0.9
    
    return clipped


# ==================== 丢包仿真 ====================

def packet_loss_simulation(speech, fs, packet_duration_ms=20, 
                          packet_loss_rate=(0.05, 0.15), 
                          max_continuous_packet_loss=3):
    """模拟网络传输丢包（与代码1一致）"""
    speech_duration_ms = speech.shape[-1] / fs * 1000
    num_packets = int(speech_duration_ms // packet_duration_ms)
    
    if num_packets == 0:
        return speech
    
    # 随机丢包率
    plr = np.random.uniform(*packet_loss_rate)
    num_packet_loss = int(round(plr * num_packets))
    
    if num_packet_loss == 0:
        return speech
    
    # 生成连续丢包段
    packet_loss_lengths = []
    remaining = num_packet_loss
    while remaining > 0:
        length = np.random.randint(1, min(max_continuous_packet_loss + 1, remaining + 1))
        packet_loss_lengths.append(length)
        remaining -= length
    
    # 随机选择起始位置
    start_indices = np.random.choice(
        range(num_packets), len(packet_loss_lengths), replace=False
    )
    
    # 应用丢包（置零）
    samples_per_packet = int(packet_duration_ms * fs / 1000)
    result = speech.copy()
    
    for idx, length in zip(start_indices, packet_loss_lengths):
        start_sample = idx * samples_per_packet
        end_sample = min((idx + length) * samples_per_packet, result.shape[-1])
        result[..., start_sample:end_sample] = 0
    
    return result


# ==================== 编解码器仿真 ====================

def apply_codec(speech, fs, codec_config=None):
    """应用音频编解码器压缩（使用 torchaudio AudioEffector）- 参考代码风格"""
    if codec_config is None:
        codec_config = {"format": "mp3", "encoder": None, "qscale": 2}
    
    format_type = codec_config.get("format", "mp3")
    encoder = codec_config.get("encoder", None)
    qscale = codec_config.get("qscale", 2)
    
    # 确保输入是 numpy 数组且形状正确
    if torch.is_tensor(speech):
        speech = speech.numpy()
    
    # 转换形状: (channel, sample) -> (sample, channel) 以匹配 torchaudio 期望
    if speech.ndim == 2 and speech.shape[0] == 1:
        speech_input = speech.T  # (1, T) -> (T, 1)
    else:
        speech_input = speech.T if speech.ndim == 2 else speech[:, None]
    
    try:
        # 创建 AudioEffector
        effector = AudioEffector(
            format=format_type,
            encoder=encoder if encoder != "None" else None,
            codec_config=CodecConfig(qscale=qscale) if qscale is not None else None,
            pad_end=True,
        )
        
        # 应用编解码器
        output = effector.apply(torch.from_numpy(speech_input), fs).numpy()
        
    except Exception as e:
        print(f"Codec error: format={format_type}, encoder={encoder}, qscale={qscale}", flush=True)
        print(f"Error: {e}", flush=True)
        # 错误时返回原始音频
        return speech
    
    # 转换回原始形状: (sample, channel) -> (channel, sample)
    if output.ndim == 2:
        output = output.T  # (T, C) -> (C, T)
    else:
        output = output[None, :]  # (T,) -> (1, T)
    
    # 长度对齐（参考代码风格）
    if output.shape[-1] < speech.shape[-1]:
        # 填充零
        pad_len = speech.shape[-1] - output.shape[-1]
        output = np.pad(output, [(0, 0), (0, pad_len)], mode="constant", constant_values=0)
    elif output.shape[-1] > speech.shape[-1]:
        # 截断
        output = output[..., :speech.shape[-1]]
    
    # 确保形状一致
    assert speech.shape == output.shape, f"Shape mismatch: {speech.shape} vs {output.shape}"
    
    return output

# ==================== 时长控制：生成采样列表 ====================

def generate_sampling_plan(speech_list, target_hours=None, target_samples=None, 
                         sr=16000, hours_tolerance=0.1):
    """
    生成采样计划，支持按目标时长或目标样本数循环采样
    """
    
    if target_hours is not None and target_samples is not None:
        raise ValueError("不能同时指定 target_hours 和 target_samples")
    
    if target_hours is None and target_samples is None:
        # 不指定时长：每个文件只用一次
        return [(path, 0) for path in speech_list], None
    
    # 计算输入总时长
    print("计算输入语音总时长...")
    input_durations = []
    total_input_duration = 0
    for path in tqdm(speech_list, desc="Scanning input files"):
        dur = get_audio_duration(path)
        input_durations.append(dur)
        total_input_duration += dur
    
    print(f"输入语音: {len(speech_list)} 个文件, 总时长 {total_input_duration/3600:.2f} 小时")
    
    if target_hours is not None:
        # 按目标时长计算
        total_target_duration = target_hours * 3600  # 转为秒
        avg_duration = total_input_duration / len(speech_list)
        target_samples = int(total_target_duration / avg_duration)
        
    else:
        # 按目标样本数
        total_target_duration = None
    
    # 生成采样计划（循环采样）
    sampling_plan = []
    num_input = len(speech_list)
    
    for i in range(target_samples):
        idx = i % num_input  # 循环索引
        repeat_id = i // num_input  # 第几轮重复
        sampling_plan.append((speech_list[idx], repeat_id))
    
    estimated_duration = total_input_duration * (target_samples / num_input)
    print(f"采样计划: {target_samples} 个样本, 约 {estimated_duration/3600:.2f} 小时")
    print(f"重复轮数: {target_samples // num_input} 轮 + {target_samples % num_input} 个文件")
    
    return sampling_plan, total_target_duration


# ==================== 核心处理函数（集成所有功能）====================

def process_from_audio_path(
    noise_path=None,
    wind_noise_path=None,
    raw_wind_path=None,  # 新增：原始风声录音路径
    use_wind_generator=False,  # 新增：是否使用生成器
    wind_generator=None,  # 新增：生成器实例
    vocal_path=None,
    rir_path=None,
    to_seperate_vocal_paths=None,
    fs=None,
    force_1ch=True,
    degradation_config=None,
    length=None,
    clean_audio=None,
    uid="temp",
    rng=None,
):
    """
    完整的音频处理流程（集成风噪生成）
    """
    if rng is None:
        rng = np.random.default_rng()
        
    if fs is None and vocal_path is not None:
        fs = sf.info(vocal_path).samplerate

    if clean_audio is None:
        vocal, _ = read_audio(vocal_path, force_1ch=force_1ch, fs=fs)
    else:
        vocal = clean_audio
    
    noisy_vocal = vocal.copy()
    
    # 1. 混合干扰语音（如果有）
    if to_seperate_vocal_paths is not None:
        for to_seperate_vocal_path in to_seperate_vocal_paths:
            to_seperate_vocal, _ = read_audio(to_seperate_vocal_path, force_1ch=force_1ch, fs=fs)
            snr = rng.uniform(degradation_config["voice_snr_min"], degradation_config["voice_snr_max"])
            noisy_vocal, _ = add_noise(noisy_vocal, to_seperate_vocal, snr=snr, rng=rng)

    # 2. 添加混响
    if rir_path is not None:
        rir_sample = read_audio(rir_path, force_1ch=force_1ch, fs=fs)[0]
        noisy_vocal, vocal = add_reverberation_v2(vocal, noisy_vocal, rir_sample, fs)

    # 3. 添加风噪声（支持预生成或实时生成）
    use_wind = False
    wind_sample = None
    
    # 优先级：预生成风噪 > 实时生成风噪 > 无风噪
    if wind_noise_path is not None:
        # 使用预生成的风噪文件
        wind_sample, _ = read_audio(wind_noise_path, force_1ch=force_1ch, fs=fs)
        use_wind = True
    elif use_wind_generator and raw_wind_path is not None:
        # 使用 SC-Wind-Noise-Generator 实时生成
        if wind_generator is not None:
            wind_sample = wind_generator.generate(raw_wind_path, fs=fs)
            if wind_sample is not None:
                use_wind = True
    
    if use_wind and wind_sample is not None:
        wn_conf = degradation_config["wind_noise_config"]
        snr = rng.uniform(*degradation_config["wind_noise_snr_range"])
        
        # 使用与代码1一致的 FFmpeg 风噪处理
        threshold = rng.uniform(*wn_conf["threshold"])
        ratio = rng.uniform(*wn_conf["ratio"])
        attack = rng.uniform(*wn_conf["attack"])
        release = rng.uniform(*wn_conf["release"])
        sc_gain = rng.uniform(*wn_conf["sc_gain"])
        clipping_threshold = rng.uniform(*wn_conf["clipping_threshold"])
        apply_clipping = rng.random() < wn_conf["clipping_chance"]
        
        noisy_vocal = apply_wind_noise_ffmpeg(
            speech_sample=noisy_vocal,
            noise_sample=wind_sample,
            fs=fs,
            uid=uid,
            threshold=threshold,
            ratio=ratio,
            attack=attack,
            release=release,
            sc_gain=sc_gain,
            clipping=apply_clipping,
            clipping_threshold=clipping_threshold,
            snr=snr,
            rng=rng,
            on_the_fly=True
        )
    
    # 4. 添加常规噪声（如果没有用风噪或概率允许叠加）
    elif noise_path is not None and rng.random() < degradation_config["p_noise"]:
        noise, _ = read_audio(noise_path, force_1ch=force_1ch, fs=fs)
        snr = rng.uniform(degradation_config["snr_min"], degradation_config["snr_max"])
        noisy_vocal, _ = add_noise(noisy_vocal, noise, snr=snr, rng=rng)

    # 5. 添加削波
    if rng.random() < degradation_config["p_clipping"]:
        clip_type = degradation_config.get("clip_type", "basic")
        if clip_type == "hard":
            noisy_vocal = clipping_hard(noisy_vocal, 0.1, 0.5)
        elif clip_type == "soft":
            noisy_vocal = clipping_soft(noisy_vocal, 0.1, 0.5)
        elif clip_type == "gain":
            noisy_vocal = clipping_with_gain(noisy_vocal, 0.1, 0.3, (3, 8))
        else:
            noisy_vocal = clipping(noisy_vocal, 0.06, 0.9)

    # 6. 添加带宽限制
    if rng.random() < degradation_config["p_bandwidth_limitation"]:
        fs_new = rng.choice(degradation_config["bandwidth_limitation_rates"])
        res_type = rng.choice(degradation_config["bandwidth_limitation_methods"])
        noisy_vocal = bandwidth_limitation(noisy_vocal, fs=fs, fs_new=fs_new, res_type=res_type)
    
    # 7. 添加丢包
    if rng.random() < degradation_config.get("p_packet_loss", 0.0):
        noisy_vocal = packet_loss_simulation(
            noisy_vocal, fs=fs,
            packet_duration_ms=degradation_config.get("packet_duration_ms", 20),
            packet_loss_rate=degradation_config.get("packet_loss_rate", (0.05, 0.15)),
            max_continuous_packet_loss=degradation_config.get("max_continuous_packet_loss", 3)
        )
    
    # 8. 添加编解码器（建议放最后，模拟传输链路末端）
    if rng.random() < degradation_config.get("p_codec", 0.0):
        codec_cfg = rng.choice(degradation_config["codec_configs"])
        noisy_vocal = apply_codec(noisy_vocal, fs=fs, codec_config=codec_cfg)

    # 最终归一化
    scale = 1 / max(np.max(np.abs(noisy_vocal)), np.max(np.abs(vocal)), 1e-8)
    vocal *= scale
    noisy_vocal *= scale

    return vocal, None, noisy_vocal, fs


# ==================== 处理单个样本的包装函数 ====================

def process_single_item_full(args_tuple):
    """组合退化处理（完整版）"""
    (speech_path, repeat_id, noise_list, wind_list, raw_wind_list, 
     use_wind_gen, wind_generator, rir_list, config, dst_dir, sr) = args_tuple
    
    try:
        # 生成唯一ID
        base_name = os.path.splitext(os.path.basename(speech_path))[0]
        uid = f"{base_name}_{repeat_id}_{random.randint(1000, 9999)}"
        
        # 创建随机数生成器
        rng = np.random.default_rng(int(repeat_id) + hash(base_name) % 10000)
        
        # 随机选择资源
        noise_path = rng.choice(noise_list) if noise_list else None
        rir_path = rng.choice(rir_list) if rir_list and rng.random() < config.get("p_reverb", 0.5) else None
        wind_path = rng.choice(wind_list) if wind_list and rng.random() < config.get("p_wind_noise", 0.1) else None
        raw_wind_path = rng.choice(raw_wind_list) if raw_wind_list and use_wind_gen else None
        
        # 如果用了风噪且没开常规噪声，可以跳过noise
        if (wind_path is not None or (use_wind_gen and raw_wind_path is not None)) and rng.random() < 0.5:
            noise_path = None

        audio, fs = read_audio(speech_path, force_1ch=True, fs=sr)

        clean_sample, _, noisy_speech, _ = process_from_audio_path(
            noise_path=noise_path,
            wind_noise_path=wind_path,
            raw_wind_path=raw_wind_path,
            use_wind_generator=use_wind_gen,
            wind_generator=wind_generator,
            rir_path=rir_path,
            fs=fs,
            force_1ch=True,
            degradation_config=config,
            clean_audio=audio,
            uid=uid,
            rng=rng,
        )

        filename = f"{base_name}_r{repeat_id}.wav" if repeat_id > 0 else f"{base_name}.wav"
        
        save_audio(clean_sample, os.path.join(dst_dir, "clean", filename), fs)
        save_audio(noisy_speech, os.path.join(dst_dir, "noisy", filename), fs)
        
        return f"OK: {filename}"
    except Exception as e:
        return f"Error: {speech_path}: {e}"


# ==================== 配置与默认参数 ====================

def get_default_full_config():
    """获取默认的完整组合退化配置"""
    return {
        "p_noise": 0,
        "snr_min": -5,
        "snr_max": 20,
        "voice_snr_min": 0,
        "voice_snr_max": 10,
        "p_reverb": 0,
        "p_clipping": 1,
        "clip_type": "basic",
        "clipping_min_quantile": 0.06,
        "clipping_max_quantile": 0.9,
        "p_bandwidth_limitation": 0,
        "bandwidth_limitation_rates": [4000, 8000],
        "bandwidth_limitation_methods": ["kaiser_best", "kaiser_fast", "scipy", "polyphase"],
        
        # 丢包参数
        "p_packet_loss": 0,
        "packet_duration_ms": 70,
        "packet_loss_rate": (0.05, 0.15),
        "max_continuous_packet_loss": 3,
        
        # 风噪声参数（与代码1一致）
        "p_wind_noise": 0,
        "wind_noise_snr_range": (-5, 15),
        "wind_noise_config": {
            "threshold": (-25, -15),      # dB
            "ratio": (3.0, 6.0),
            "attack": (3.0, 10.0),        # ms
            "release": (30.0, 100.0),     # ms
            "sc_gain": (1.2, 2.0),
            "clipping_threshold": (0.6, 0.9),
            "clipping_chance": 0.7
        },
        
        # 编解码器参数
        "p_codec": 0,
        "codec_configs": [
            {"format": "mp3", "encoder": None, "qscale": 2},      # MP3 中等质量
            {"format": "mp3", "encoder": None, "qscale": 4},      # MP3 较低质量
            {"format": "ogg", "encoder": "vorbis", "qscale": 3},  # Ogg Vorbis
            {"format": "ogg", "encoder": "opus", "qscale": 5},     # Opus（如果支持）
        ]
    }


# ==================== 子命令处理函数 ====================

def prepare_sampling_plan(args, speech_list):
    """准备采样计划"""
    if args.target_hours is not None:
        return generate_sampling_plan(
            speech_list, 
            target_hours=args.target_hours,
            sr=args.sr
        )
    elif args.target_samples is not None:
        return generate_sampling_plan(
            speech_list,
            target_samples=args.target_samples,
            sr=args.sr
        )
    else:
        return [(path, 0) for path in speech_list], None


def handle_bandwidth(args):
    """处理带宽限制子命令"""
    config = {
        "p_noise": 0.0,
        "p_reverb": 0.0,
        "p_clipping": 0.0,
        "p_bandwidth_limitation": 1.0,
        "bandwidth_limitation_rates": args.bw_rates,
        "bandwidth_limitation_methods": args.bw_methods,
    }
    
    os.makedirs(args.dst_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, "clean"), exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, "noisy"), exist_ok=True)
    
    speech_list = read_scp(args.speech_scp)
    sampling_plan, _ = prepare_sampling_plan(args, speech_list)
    
    print(f"Processing {len(sampling_plan)} samples...")
    
    def process_item(args_tuple):
        speech_path, repeat_id = args_tuple
        try:
            audio, fs = read_audio(speech_path, force_1ch=True, fs=args.sr)
            fs_new = random.choice(config["bandwidth_limitation_rates"])
            res_type = random.choice(config["bandwidth_limitation_methods"])
            
            limited_audio = bandwidth_limitation(audio, fs=fs, fs_new=fs_new, res_type=res_type)
            
            scale = 1.0 / max(np.max(np.abs(limited_audio)), 1e-8)
            limited_audio *= scale
            audio *= scale
            
            base_name = os.path.splitext(os.path.basename(speech_path))[0]
            filename = f"{base_name}_r{repeat_id}.wav" if repeat_id > 0 else f"{base_name}.wav"
            
            save_audio(audio, os.path.join(args.dst_dir, "clean", filename), fs)
            save_audio(limited_audio, os.path.join(args.dst_dir, "noisy", filename), fs)
            return f"OK: {filename}"
        except Exception as e:
            return f"Error: {speech_path}: {e}"
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_item, arg) for arg in sampling_plan]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Bandwidth"):
            result = future.result()
            if "Error" in result:
                tqdm.write(result)
    
    print(f"\nCompleted. Output: {args.dst_dir}")


def handle_clipping(args):
    """处理削波子命令"""
    config = {
        "p_noise": 0.0,
        "p_reverb": 0.0,
        "p_clipping": 1.0,
        "p_bandwidth_limitation": 0.0,
        "clip_type": args.clip_type,
        "clipping_min_quantile": args.min_quantile,
        "clipping_max_quantile": args.max_quantile,
    }
    
    if args.clip_type == "gain":
        config["gain_range"] = (args.gain_min, args.gain_max)
    
    os.makedirs(args.dst_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, "clean"), exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, "noisy"), exist_ok=True)
    
    speech_list = read_scp(args.speech_scp)
    sampling_plan, _ = prepare_sampling_plan(args, speech_list)
    
    print(f"Processing {len(sampling_plan)} samples...")
    
    def process_item(args_tuple):
        speech_path, repeat_id = args_tuple
        try:
            audio, fs = read_audio(speech_path, force_1ch=True, fs=args.sr)
            
            if config["clip_type"] == "hard":
                clipped = clipping_hard(audio, config["clipping_min_quantile"], config["clipping_max_quantile"])
            elif config["clip_type"] == "soft":
                clipped = clipping_soft(audio, config["clipping_min_quantile"], config["clipping_max_quantile"])
            elif config["clip_type"] == "gain":
                clipped = clipping_with_gain(audio, config["clipping_min_quantile"], 
                                           config["clipping_max_quantile"], config["gain_range"])
            else:
                clipped = clipping(audio, 0.06, 0.9)
            
            scale = 1.0 / max(np.max(np.abs(clipped)), 1e-8)
            clipped *= scale * 0.95
            audio *= scale * 0.95
            
            base_name = os.path.splitext(os.path.basename(speech_path))[0]
            filename = f"{base_name}_r{repeat_id}.wav" if repeat_id > 0 else f"{base_name}.wav"
            
            save_audio(audio, os.path.join(args.dst_dir, "clean", filename), fs)
            save_audio(clipped, os.path.join(args.dst_dir, "noisy", filename), fs)
            return f"OK: {filename}"
        except Exception as e:
            return f"Error: {speech_path}: {e}"
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_item, arg) for arg in sampling_plan]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Clipping"):
            result = future.result()
            if "Error" in result:
                tqdm.write(result)
    
    print(f"\nCompleted. Output: {args.dst_dir}")


def handle_reverb(args):
    """处理混响子命令"""
    config = {
        "p_noise": 0.0,
        "p_reverb": 1.0,
        "p_clipping": 0.0,
        "p_bandwidth_limitation": 0.0,
    }
    
    os.makedirs(args.dst_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, "clean"), exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, "noisy"), exist_ok=True)
    
    speech_list = read_scp(args.speech_scp)
    rir_list = read_scp(args.rir_scp)
    sampling_plan, _ = prepare_sampling_plan(args, speech_list)
    
    print(f"Processing {len(sampling_plan)} samples...")
    print(f"Found {len(rir_list)} RIR files.")
    
    def process_item(args_tuple):
        speech_path, repeat_id = args_tuple
        try:
            rir_path = random.choice(rir_list)
            audio, fs = read_audio(speech_path, force_1ch=True, fs=args.sr)
            rir_sample, _ = read_audio(rir_path, force_1ch=True, fs=fs)
            
            reverb_speech, early = add_reverberation_v2(audio, audio.copy(), rir_sample, fs)
            
            scale = 1.0 / max(np.max(np.abs(reverb_speech)), 1e-8)
            reverb_speech *= scale
            early *= scale
            
            base_name = os.path.splitext(os.path.basename(speech_path))[0]
            filename = f"{base_name}_r{repeat_id}.wav" if repeat_id > 0 else f"{base_name}.wav"
            
            save_audio(early, os.path.join(args.dst_dir, "clean", filename), fs)
            save_audio(reverb_speech, os.path.join(args.dst_dir, "noisy", filename), fs)
            return f"OK: {filename}"
        except Exception as e:
            return f"Error: {speech_path}: {e}"
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_item, arg) for arg in sampling_plan]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reverb"):
            result = future.result()
            if "Error" in result:
                tqdm.write(result)
    
    print(f"\nCompleted. Output: {args.dst_dir}")


def handle_full(args):
    """处理组合退化子命令（完整版，集成 SC-Wind-Noise-Generator）"""
    config = get_default_full_config()
    
    # 应用命令行覆盖
    if args.p_noise is not None:
        config["p_noise"] = args.p_noise
    if args.p_reverb is not None:
        config["p_reverb"] = args.p_reverb
    if args.p_clipping is not None:
        config["p_clipping"] = args.p_clipping
    if args.p_bandwidth is not None:
        config["p_bandwidth_limitation"] = args.p_bandwidth
    if args.snr_min is not None:
        config["snr_min"] = args.snr_min
    if args.snr_max is not None:
        config["snr_max"] = args.snr_max
    if args.p_packet_loss is not None:
        config["p_packet_loss"] = args.p_packet_loss
    if args.p_wind_noise is not None:
        config["p_wind_noise"] = args.p_wind_noise
    if args.p_codec is not None:
        config["p_codec"] = args.p_codec
    if hasattr(args, 'codec_format') and args.codec_format:
        config["codec_configs"] = [{
            "format": args.codec_format,
            "encoder": args.codec_encoder,
            "qscale": args.codec_qscale
        }]
    os.makedirs(args.dst_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, "clean"), exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, "noisy"), exist_ok=True)
    
    speech_list = read_scp(args.speech_scp)
    noise_list = read_scp(args.noise_scp) if args.noise_scp else []
    wind_list = read_scp(args.wind_noise_scp) if args.wind_noise_scp else []
    rir_list = read_scp(args.rir_scp) if args.rir_scp else []
    
    # 处理 SC-Wind-Noise-Generator
    raw_wind_list = []
    wind_generator = None
    use_wind_gen = False
    
    if args.use_wind_generator:
        if not SC_WIND_GENERATOR_AVAILABLE:
            print("Warning: SC-Wind-Noise-Generator not available, falling back to regular noise")
        else:
            wind_generator = WindNoiseGeneratorWrapper()
            use_wind_gen = True
            if args.raw_wind_scp:
                raw_wind_list = read_scp(args.raw_wind_scp)
                print(f"Using SC-Wind-Noise-Generator with {len(raw_wind_list)} raw wind recordings")
    
    sampling_plan, target_duration = prepare_sampling_plan(args, speech_list)
    
    print(f"Processing {len(sampling_plan)} samples...")
    print(f"Resources: {len(noise_list)} noise, {len(wind_list)} pre-gen wind, "
          f"{len(raw_wind_list)} raw wind, {len(rir_list)} RIR")
    print(f"Degradations: noise={config['p_noise']}, wind={config['p_wind_noise']}, "
          f"reverb={config['p_reverb']}, clip={config['p_clipping']}, "
          f"bandwidth={config['p_bandwidth_limitation']}, pkt_loss={config['p_packet_loss']}, "
          f"codec={config['p_codec']}")
    
    # 构建任务参数
    task_args = [
        (speech_path, repeat_id, noise_list, wind_list, raw_wind_list,
         use_wind_gen, wind_generator, rir_list, config, args.dst_dir, args.sr)
        for speech_path, repeat_id in sampling_plan
    ]
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single_item_full, arg) for arg in task_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Full degradation"):
            result = future.result()
            if "Error" in result:
                tqdm.write(result)
    
    print(f"\nCompleted. Output: {args.dst_dir}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="音频退化仿真工具 - 集成 SC-Wind-Noise-Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法（使用预生成风噪）
  %(prog)s full --speech_scp speech.scp --noise_scp noise.scp --wind_noise_scp wind.scp \\
                --rir_scp rir.scp --dst_dir output/ --target_hours 1000
  
  # 使用 SC-Wind-Noise-Generator 实时生成风噪
  %(prog)s full --speech_scp speech.scp --noise_scp noise.scp \\
                --raw_wind_scp raw_wind.scp --use_wind_generator \\
                --rir_scp rir.scp --dst_dir output/ --target_hours 100
  
  # 单一退化模式
  %(prog)s bandwidth --speech_scp speech.scp --dst_dir output/bw --target_hours 10
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 公共参数
    common_parser = argparse.ArgumentParser(add_help=False)
    # Incremental LibriTTS-R run: by default process only the three newly
    # extracted parquet shards.  The user can still override these paths.
    common_parser.add_argument(
        "--speech_scp",
        type=str,
        default="/yangliusha02/datasets/libritts_r/codex/new_samples/new_speech.scp",
        help="新增 LibriTTS-R 样本的 clean speech.scp",
    )
    common_parser.add_argument(
        "--dst_dir",
        type=str,
        default="/yangliusha02/datasets/libritts_r/codex/new_samples/mixed",
        help="新增样本的独立输出目录",
    )
    common_parser.add_argument("--num_workers", type=int, default=8, help="并行线程数")
    common_parser.add_argument("--sr", type=int, default=24000, help="采样率")
    
    duration_group = common_parser.add_mutually_exclusive_group()
    duration_group.add_argument("--target_hours", type=float, default=None,
                                help="目标输出时长（小时）")
    duration_group.add_argument("--target_samples", type=int, default=None,
                                help="目标输出样本数")
    
    # 带宽限制子命令
    bw_parser = subparsers.add_parser("bandwidth", parents=[common_parser], help="仅带宽限制")
    bw_parser.add_argument("--bw_rates", type=int, nargs="+",
                          default=[4000, 8000])
    bw_parser.add_argument("--bw_methods", type=str, nargs="+", default=["soxr_hq"])
    bw_parser.set_defaults(func=handle_bandwidth)
    
    # 削波子命令
    clip_parser = subparsers.add_parser("clipping", parents=[common_parser], help="仅削波")
    clip_parser.add_argument("--clip_type", type=str, default="hard",
                            choices=["basic", "hard", "soft", "gain"])
    clip_parser.add_argument("--min_quantile", type=float, default=0.06)
    clip_parser.add_argument("--max_quantile", type=float, default=0.9)
    clip_parser.add_argument("--gain_min", type=float, default=3)
    clip_parser.add_argument("--gain_max", type=float, default=8)
    clip_parser.set_defaults(func=handle_clipping)
    
    # 混响子命令
    reverb_parser = subparsers.add_parser("reverb", parents=[common_parser], help="仅混响")
    reverb_parser.add_argument("--rir_scp",default="/ds/yangliusha/ylsdataset/DNSChallenge/datasets_fullband/clean-dns-dataset/RIR.scp", type=str, required=True, help="RIR文件列表")
    reverb_parser.set_defaults(func=handle_reverb)
    
    # 组合退化子命令
    full_parser = subparsers.add_parser("full", parents=[common_parser], help="组合退化")
    full_parser.add_argument("--noise_scp",default="/ds/yangliusha/ylsdataset/DNSChallenge/datasets_fullband/clean-dns-dataset/noise-mixed.scp",type=str, help="噪声文件列表")
    full_parser.add_argument("--rir_scp", default="/ds/yangliusha/ylsdataset/DNSChallenge/datasets_fullband/clean-dns-dataset/RIR.scp",type=str, help="RIR文件列表")
    full_parser.add_argument("--wind_noise_scp", default="/ds/yangliusha/ylsdataset/DNSChallenge/datasets_fullband/clean-dns-dataset/wind.scp",type=str, help="预生成风噪文件列表")
    full_parser.add_argument("--raw_wind_scp", type=str, help="原始风声录音列表（用于生成风噪）")
    full_parser.add_argument("--use_wind_generator", action="store_true", 
                            help="使用 SC-Wind-Noise-Generator 生成风噪")
    full_parser.add_argument("--p_noise", type=float, default=0.9)
    full_parser.add_argument("--p_reverb", type=float, default=0.5)
    full_parser.add_argument("--p_clipping", type=float, default=0.25)
    full_parser.add_argument("--p_bandwidth", type=float, default=0.5)
    full_parser.add_argument("--snr_min", type=float, default=-5)
    full_parser.add_argument("--snr_max", type=float, default=20)
    full_parser.add_argument("--p_packet_loss", type=float, default=0.3)
    full_parser.add_argument("--p_wind_noise", type=float, default=0.1)
    full_parser.add_argument("--p_codec", type=float, default=0)
    full_parser.add_argument("--codec_format", type=str, default="wav", choices=["mp3", "ogg", "wav"])
    full_parser.add_argument("--codec_encoder", type=str, default=None,help="Encoder: None, vorbis, opus")
    full_parser.add_argument("--codec_qscale", type=int, default=10,help="Quality scale: 0-10 (lower is better quality)")
    full_parser.set_defaults(func=handle_full)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()



#python audio_degradation_tool.py full 
