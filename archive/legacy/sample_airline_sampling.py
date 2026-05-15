# -*- coding: utf-8 -*-
"""
将 OpenML 航班延误百万数据集 抽样成 20 万条 的小数据集脚本
默认假设原始文件路径：
    data/airline/airlines_train_regression_1000000.arff
输出文件：
    data/airline/airlines_train_regression_200000.csv

说明：
1. 使用随机下采样（可选：按 label 分层抽样，见下方开关）
2. 抽样时设置 random_state 保证可复现
3. 若没有安装 scipy，请先：pip install scipy
"""

import os
import numpy as np
import pandas as pd
from scipy.io import arff  # 用于读取 .arff 文件

# ========== 参数区（需要改路径就改这里） ==========
INPUT_PATH = "data/airline/airlines_train_regression_1000000.arff"  # 原始 100w 数据
OUTPUT_PATH = "data/airline/airlines_train_regression_200000.csv"   # 输出 20w 数据

TARGET_SIZE = 200_000       # 目标样本数
RANDOM_STATE = 42           # 随机种子，保证可复现

# 是否按 label 分层抽样：
# - 若 True 且存在 'label' 列，则会尽量保持正负类比例一致
# - 若 False 或不存在 'label' 列，则普通随机抽样
USE_STRATIFIED_BY_LABEL = True


def main():
    # -------- 1. 读取 arff 文件 --------
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"找不到原始文件：{INPUT_PATH}")

    print(f"【INFO】开始加载原始数据：{INPUT_PATH}")
    data, meta = arff.loadarff(INPUT_PATH)
    df = pd.DataFrame(data)

    # 有些 arff 字段会是 bytes，需要统一转成 str
    # 这里只对 object/bytes 类型做简单转换，避免后续编码问题
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

    n_rows = len(df)
    print(f"【INFO】原始数据行数：{n_rows}，列数：{df.shape[1]}")

    if TARGET_SIZE > n_rows:
        raise ValueError(f"目标样本数 {TARGET_SIZE} 大于原始样本数 {n_rows}，无法抽样！")

    # -------- 2. 抽样 --------
    if USE_STRATIFIED_BY_LABEL and ("label" in df.columns):
        print("【INFO】检测到 'label' 列，启用按 label 分层抽样。")

        # 统计原始正负类比例
        print("【INFO】原始 label 分布：")
        print(df["label"].value_counts(normalize=True))

        # 使用 sklearn 的分层抽样比较方便，这里简单手写一个近似分层：
        # 根据原始比例，按类分别抽样，然后拼回去
        ratios = df["label"].value_counts(normalize=True)
        sampled_list = []

        # 为避免四舍五入误差，先按比例计算每类数量，最后一类用“剩余数”兜底
        labels = list(ratios.index)
        remaining = TARGET_SIZE

        for i, lb in enumerate(labels):
            if i < len(labels) - 1:
                # 按比例计算该类需要抽取的数量
                cnt = int(round(TARGET_SIZE * ratios[lb]))
                cnt = min(cnt, (df["label"] == lb).sum())  # 防止超过该类样本数
                remaining -= cnt
            else:
                # 最后一类用剩余的数量
                cnt = min(remaining, (df["label"] == lb).sum())

            if cnt <= 0:
                continue

            sub_df = df[df["label"] == lb]
            sampled_sub = sub_df.sample(
                n=cnt,
                random_state=RANDOM_STATE,
                replace=False
            )
            sampled_list.append(sampled_sub)

            print(f"【INFO】label={lb} 抽取 {cnt} 条样本（该类总数 {len(sub_df)}）")

        df_sampled = pd.concat(sampled_list, axis=0)
        # 打乱一下顺序更自然
        df_sampled = df_sampled.sample(
            frac=1.0,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

        print(f"【INFO】分层抽样后总行数：{len(df_sampled)}")
        print("【INFO】抽样后 label 分布：")
        print(df_sampled["label"].value_counts(normalize=True))

    else:
        if USE_STRATIFIED_BY_LABEL:
            print("【WARN】开启了分层抽样开关，但数据中没有 'label' 列，自动退化为普通随机抽样。")
        else:
            print("【INFO】不使用分层抽样，执行普通随机下采样。")

        df_sampled = df.sample(
            n=TARGET_SIZE,
            random_state=RANDOM_STATE,
            replace=False
        ).reset_index(drop=True)

        print(f"【INFO】随机抽样后总行数：{len(df_sampled)}")
        if "label" in df_sampled.columns:
            print("【INFO】抽样后 label 分布（若存在）：")
            print(df_sampled["label"].value_counts(normalize=True))

    # -------- 3. 保存为 CSV --------
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and (not os.path.exists(out_dir)):
        os.makedirs(out_dir, exist_ok=True)

    df_sampled.to_csv(OUTPUT_PATH, index=False)
    print(f"【INFO】抽样完成，已保存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
