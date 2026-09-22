import os

import pandas as pd


# ============================================================
# LibriTTS-R 配置
# ============================================================

LIBRITTS_AUDIO_BASE_PATH = (
    "/yangliusha02/datasets/libritts_r/train/clean"
)

LIBRITTS_CONTROL_BASE_PATH = (
    "/yangliusha02/datasets/libritts_r/train/noisy"
)

# LibriTTS-R 原始 metadata
LIBRITTS_METADATA_PATH = (
    "/yangliusha02/datasets/libritts_r/data/metadata.csv"
)

# 只输出 LibriTTS-R 的 metadata，不包含 DNS
OUTPUT_CSV = (
    "/yangliusha02/datasets/libritts_r/metadata_LibriTTS.csv"
)

RANDOM_SEED = 42


def generate_metadata():
    print("=" * 70)
    print("开始处理 LibriTTS-R 数据集")
    print("=" * 70)
    print(f"Metadata     : {LIBRITTS_METADATA_PATH}")
    print(f"Clean path   : {LIBRITTS_AUDIO_BASE_PATH}")
    print(f"Control path : {LIBRITTS_CONTROL_BASE_PATH}")
    print(f"Output CSV   : {OUTPUT_CSV}")

    if not os.path.isfile(LIBRITTS_METADATA_PATH):
        raise FileNotFoundError(
            f"找不到 LibriTTS-R metadata:\n{LIBRITTS_METADATA_PATH}"
        )

    if not os.path.isdir(LIBRITTS_AUDIO_BASE_PATH):
        raise FileNotFoundError(
            f"找不到 LibriTTS-R clean 文件夹:\n{LIBRITTS_AUDIO_BASE_PATH}"
        )

    if not os.path.isdir(LIBRITTS_CONTROL_BASE_PATH):
        raise FileNotFoundError(
            f"找不到 LibriTTS-R noisy 文件夹:\n{LIBRITTS_CONTROL_BASE_PATH}"
        )

    libritts_df = pd.read_csv(
        LIBRITTS_METADATA_PATH,
        encoding="utf-8",
    )

    print(f"LibriTTS-R metadata 总行数: {len(libritts_df)}")
    print(f"metadata columns: {list(libritts_df.columns)}")

    required_columns = ["path", "text_normalized"]
    missing_columns = [
        column for column in required_columns
        if column not in libritts_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"metadata.csv 缺少字段: {missing_columns}\n"
            f"当前字段为: {list(libritts_df.columns)}"
        )

    all_data = []
    missing_clean = 0
    missing_control = 0
    invalid_path = 0

    for _, row in libritts_df.iterrows():
        original_path = row["path"]

        if pd.isna(original_path):
            invalid_path += 1
            continue

        filename = os.path.basename(str(original_path).strip())
        if not filename:
            invalid_path += 1
            continue

        audio_path = os.path.join(
            LIBRITTS_AUDIO_BASE_PATH,
            filename,
        )
        control_path = os.path.join(
            LIBRITTS_CONTROL_BASE_PATH,
            filename,
        )

        if not os.path.isfile(audio_path):
            missing_clean += 1
            continue

        if not os.path.isfile(control_path):
            missing_control += 1
            continue

        text = row["text_normalized"]
        if pd.isna(text):
            text = ""
        else:
            text = str(text).replace("\n", " ").replace("\r", " ").strip()

        all_data.append({
            "audio_path": audio_path,
            "text": text,
            "control_path": control_path,
        })

    df = pd.DataFrame(
        all_data,
        columns=["audio_path", "text", "control_path"],
    )

    print("\nLibriTTS-R 处理完成")
    print(f"  有效样本        : {len(df)}")
    print(f"  缺少 clean      : {missing_clean}")
    print(f"  缺少 noisy      : {missing_control}")
    print(f"  path 无效       : {invalid_path}")

    df = (
        df.sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # prepare_csv_wavs.py 使用 | 作为分隔符
    df.to_csv(
        OUTPUT_CSV,
        index=False,
        sep="|",
        encoding="utf-8",
    )

    print("\nCSV 已保存至:")
    print(OUTPUT_CSV)
    print(f"最终样本数: {len(df)}")
    print("随机打乱后的前 5 条数据:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    generate_metadata()
