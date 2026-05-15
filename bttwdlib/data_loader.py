import json
from pathlib import Path

import pandas as pd
from scipy.io import arff
from .synth_data import load_synth_strong_v1, load_synth_strong_v2
from .utils_logging import log_info


def _map_target_with_config(
    df: pd.DataFrame, target_col: str, data_cfg: dict, extra_mapping: dict | None = None
) -> pd.DataFrame:
    """按照配置将目标列安全映射为 0/1，并记录日志。"""

    dropna_target = data_cfg.get("dropna_target", False)
    positive_label = data_cfg.get("positive_label")
    negative_label = data_cfg.get("negative_label")

    # 如果目标列已经是 0/1，则直接返回，避免重复映射导致全 NaN
    uniq_raw = pd.unique(df[target_col].dropna())
    if uniq_raw.size:
        uniq_numeric = pd.to_numeric(uniq_raw, errors="coerce")
        if not pd.isna(uniq_numeric).any():
            if set(map(int, uniq_numeric)) <= {0, 1}:
                df[target_col] = df[target_col].astype(int)
                log_info(
                    f"【数据加载】目标列 {target_col} 已检测为 0/1 标签，跳过映射逻辑"
                )
                return df

    # 统一对标签进行字符串清洗
    y_raw = df[target_col].astype(str).str.strip().str.lower()

    mapping: dict[str, int] = {}
    if positive_label is not None:
        mapping[str(positive_label).strip().lower()] = 1
    if negative_label is not None:
        mapping[str(negative_label).strip().lower()] = 0

    if extra_mapping:
        mapping.update({str(k).strip().lower(): int(v) for k, v in extra_mapping.items()})

    df[target_col] = y_raw.map(mapping)

    before = len(df)
    missing_count = df[target_col].isna().sum()
    if dropna_target:
        df = df.dropna(subset=[target_col]).copy()
    elif negative_label is None and positive_label is not None:
        # 兼容旧行为：未指定负类标签且未开启 drop 时，将非正类视为 0
        if missing_count > 0:
            log_info(
                f"【数据加载】{missing_count} 条标签无法映射，未指定负类且未开启 dropna_target，已按 0 处理"
            )
        df[target_col] = df[target_col].fillna(0)
    elif missing_count > 0:
        # 当正负类都显式配置时，将无法映射的标签删除，避免 astype 触发 IntCastingNaNError
        log_info(
            f"【数据加载】{missing_count} 条标签无法映射，占比={missing_count / before:.2%}，正负类已指定且未开启 dropna_target，已自动删除这些样本"
        )
        df = df.dropna(subset=[target_col]).copy()
    after = len(df)

    df[target_col] = df[target_col].astype(int)

    log_info(
        f"【数据加载】标签列 {target_col} 已处理完成："
        f"dropna_target={dropna_target}, "
        f"丢弃样本={before - after}, "
        f"最终样本数={after}, "
        f"正类比例={df[target_col].mean():.2%}"
    )

    return df


def load_adult_raw(cfg: dict) -> pd.DataFrame:
    """
    从 cfg['DATA']['raw_path'] 读取 CSV，将 "?" 视为缺失值。
    返回 DataFrame，包含列名。
    """
    data_cfg = cfg.get("DATA", {})
    path = data_cfg.get("raw_path")
    col_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        data_cfg.get("target_col", "income"),
    ]
    df = pd.read_csv(
        path,
        header=None,
        names=col_names,
        na_values=["?"],
        skipinitialspace=True,
    )
    target_col = data_cfg.get("target_col", "income")
    pos_label = data_cfg.get("positive_label", ">50K")
    total = len(df)
    n_features = df.shape[1] - 1
    pos_rate = (df[target_col] == pos_label).mean()
    log_info(
        f"【数据加载完毕】样本数={total}，特征数={n_features}，正类比例={pos_rate:.2f}"
    )
    return df


def _load_csv_like(path: str, data_cfg: dict) -> pd.DataFrame:
    sep = data_cfg.get("sep", ",")
    encoding = data_cfg.get("encoding", "utf-8")
    header = data_cfg.get("header", "infer")
    names = data_cfg.get("col_names")
    skiprows = data_cfg.get("skiprows")
    na_values = data_cfg.get("na_values")
    df = pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        header=header,
        names=names,
        skiprows=skiprows,
        na_values=na_values,
    )
    log_info(f"【数据加载】文本表格 {path} 已读取，样本数={len(df)}，列数={df.shape[1]}")
    return df


def _load_arff(path: str) -> pd.DataFrame:
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    # 将 bytes 类型的类别值解码为字符串
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, (bytes, bytearray)) else x)
    log_info(f"【数据加载】ARFF 文件 {path} 已读取，含 {df.shape[0]} 条记录，{df.shape[1]} 列")
    return df


def _apply_target_transform(df: pd.DataFrame, data_cfg: dict) -> tuple[pd.DataFrame, str]:
    target_col = data_cfg.get("target_col")
    transform_cfg = data_cfg.get("target_transform") or {}
    if not transform_cfg:
        return df, target_col

    force_transform = transform_cfg.get("force", False)
    if not force_transform:
        unique_vals = set(df[target_col].dropna().unique())
        if unique_vals and unique_vals.issubset({0, 1}):
            log_info(
                f"【数据加载】目标列 {target_col} 已是二元标签 {sorted(unique_vals)}，跳过 target_transform"
            )
            return df, target_col

    transform_type = transform_cfg.get("type")
    if transform_type == "threshold_binary":
        threshold = transform_cfg.get("threshold", 0.0)
        greater_is_positive = transform_cfg.get("greater_is_positive", True)
        new_col = transform_cfg.get("new_col", f"{target_col}_bin")
        cmp = df[target_col] > threshold if greater_is_positive else df[target_col] < threshold
        df[new_col] = cmp.astype(int)
        log_info(
            f"【目标变换】已按阈值 {threshold} 生成二分类标签列 {new_col}，正类取 {'>' if greater_is_positive else '<'} {threshold}"
        )
        return df, new_col

    log_info(f"【目标变换】未识别的 target_transform.type={transform_type}，保持原目标列 {target_col}")
    return df, target_col


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, str]:
    """根据配置加载数据集，支持 adult CSV、ARFF 以及多种表格格式。"""

    data_cfg = cfg.get("DATA", {})
    train_csv = data_cfg.get("train_csv")
    test_csv = data_cfg.get("test_csv")
    raw_path = data_cfg.get("raw_path") or data_cfg.get("path") or data_cfg.get("data_path")
    file_type_cfg = data_cfg.get("file_type")
    file_type = str(file_type_cfg).lower() if file_type_cfg is not None else ""
    repo_root = Path(__file__).resolve().parent.parent
    target_mapped = False

    def _resolve_path(path_str: str | None):
        if not path_str:
            return None
        p = Path(path_str)
        if not p.is_absolute() and not p.exists():
            alt_path = repo_root / p
            if alt_path.exists():
                return str(alt_path)
        return str(p)

    train_csv = _resolve_path(train_csv)
    test_csv = _resolve_path(test_csv)

    df = None

    if raw_path:
        raw_path_path = Path(raw_path)
        if not raw_path_path.is_absolute() and not raw_path_path.exists():
            alt_path = repo_root / raw_path_path
            if alt_path.exists():
                raw_path_path = alt_path
        raw_path = str(raw_path_path)

    if train_csv and test_csv:
        train_df = _load_csv_like(train_csv, data_cfg)
        test_df = _load_csv_like(test_csv, data_cfg)
        train_df["split"] = "train"
        test_df["split"] = "test"
        df = pd.concat([train_df, test_df], ignore_index=True)
        file_type = "csv"
        raw_path = None
        log_info(
            "【数据加载】检测到显式 train/test 配置，" f"训练集 n={len(train_df)}，测试集 n={len(test_df)}"
        )

    if not file_type and raw_path:
        file_type = Path(raw_path).suffix.lower().lstrip(".") or "csv"
    dataset_name = data_cfg.get("dataset_name", "dataset")

    if raw_path is None and df is None:
        raise FileNotFoundError("配置中缺少 raw_path/path 字段，无法读取数据")

    if raw_path is not None or df is None:
        if file_type == "arff":
            df = _load_arff(raw_path)
        elif file_type in {"csv", "txt"}:
            dataset_name_lower = dataset_name.lower()
            if dataset_name_lower == "adult":
                df = load_adult_raw(cfg)
            elif dataset_name_lower == "bank_full":
                tmp_cfg = dict(data_cfg)
                tmp_cfg.setdefault("sep", ";")
                df = _load_csv_like(raw_path, tmp_cfg)
                target_col = data_cfg.get("target_col", "y")
                if target_col not in df.columns:
                    raise KeyError(f"银行数据集中未找到标签列 {target_col}")
                data_cfg.setdefault("positive_label", "yes")
                data_cfg.setdefault("negative_label", "no")
                df = _map_target_with_config(df, target_col, data_cfg)
                data_cfg["positive_label"] = 1
                data_cfg.setdefault("negative_label", 0)
                target_mapped = True
                data_cfg.setdefault(
                    "numeric_cols",
                    ["age", "balance", "day", "duration", "campaign", "pdays", "previous"],
                )
                data_cfg.setdefault(
                    "categorical_cols",
                    [
                        "job",
                        "marital",
                        "education",
                        "default",
                        "housing",
                        "loan",
                        "contact",
                        "month",
                        "poutcome",
                    ],
                )
                log_info(
                    "【数据加载】银行营销数据集已读取，标签已映射为0/1，"
                    f"样本数={len(df)}，正类比例={df[target_col].mean():.2%}"
                )
            elif dataset_name_lower == "hospital_readmissions":
                df = _load_csv_like(raw_path, data_cfg)
                target_col = data_cfg.get("target_col", "readmitted")
                if target_col not in df.columns:
                    raise KeyError(f"医院再入院数据集中未找到标签列 {target_col}")
                data_cfg.setdefault("positive_label", "yes")
                data_cfg.setdefault("negative_label", "no")
                df = _map_target_with_config(df, target_col, data_cfg)
                data_cfg["positive_label"] = 1
                data_cfg.setdefault("negative_label", 0)
                target_mapped = True



            elif dataset_name_lower == "diabetic":
                df = _load_csv_like(raw_path, data_cfg)
                target_col = data_cfg.get("target_col", "readmitted")
                if target_col not in df.columns:
                    raise KeyError(f"糖尿病再入院数据集中未找到标签列 {target_col}")
                data_cfg.setdefault("positive_label", "<30")
                data_cfg.setdefault("negative_label", ">30")
                df = _map_target_with_config(df, target_col, data_cfg, extra_mapping={"no": 0})
                data_cfg["positive_label"] = 1
                data_cfg.setdefault("negative_label", 0)
                target_mapped = True

            elif dataset_name_lower == "synth_strong_v1":
                default_path = raw_path or repo_root / "data" / "synth_strong_v1.csv"
                default_path = Path(default_path)
                if not default_path.exists():
                    raise FileNotFoundError(
                        f"未找到合成数据文件 {default_path}，请先运行 scripts/generate_synth_dataset.py 生成"
                    )
                df = load_synth_strong_v1(default_path)
                if not set(pd.unique(df.get("target", []))).issubset({0, 1}):
                    raise ValueError("合成数据 target 列必须为 0/1，请检查生成流程或手动修改是否破坏标签。")
                df["group"] = df["group"].astype(str)
                data_cfg.setdefault("target_col", "target")
                data_cfg["positive_label"] = 1
                data_cfg.setdefault("negative_label", 0)
                target_mapped = True

            elif dataset_name_lower == "synth_strong_v2":
                default_path = raw_path or repo_root / "data" / "synth_strong_v2.csv"
                default_path = Path(default_path)
                if not default_path.exists():
                    raise FileNotFoundError(
                        f"未找到合成数据文件 {default_path}，请先运行 scripts/generate_synth_dataset.py --version v2 生成"
                    )
                df = load_synth_strong_v2(default_path)
                if not set(pd.unique(df.get("target", []))).issubset({0, 1}):
                    raise ValueError("合成数据 target 列必须为 0/1，请检查生成流程或手动修改是否破坏标签。")
                df["group"] = df["group"].astype(str)
                data_cfg.setdefault("target_col", "target")
                data_cfg["positive_label"] = 1
                data_cfg.setdefault("negative_label", 0)
                meta_path_cfg = data_cfg.get("meta_path")
                meta_path = (
                    Path(meta_path_cfg)
                    if meta_path_cfg
                    else default_path.parent / f"{default_path.stem}_meta.json"
                )
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    costs_from_meta = meta.get("bucket_path_cost_profile")
                    if costs_from_meta:
                        thr = cfg.get("THRESHOLD")
                        thrs = cfg.get("THRESHOLDS")
                        if isinstance(thr, dict):
                            thr["costs_per_bucket"] = costs_from_meta
                        elif isinstance(thrs, dict):
                            thrs["costs_per_bucket"] = costs_from_meta
                        else:
                            cfg["THRESHOLD"] = {"costs_per_bucket": costs_from_meta}
                        log_info(
                            f"【数据加载】检测到 synth_strong_v2 元数据 {meta_path}，已注入 {len(costs_from_meta)} 条桶级 cost 配置"
                        )
                    else:
                        log_info(
                            f"【数据加载】元数据 {meta_path} 未包含 bucket_path_cost_profile，保持全局 cost"
                        )
                else:
                    log_info(f"【数据加载】未找到 synth_strong_v2 meta 文件 {meta_path}，跳过桶级 cost 注入")
                target_mapped = True

            else:
                df = _load_csv_like(raw_path, data_cfg)
        elif file_type == "tsv":
            tmp = dict(data_cfg)
            tmp.setdefault("sep", "\t")
            df = _load_csv_like(raw_path, tmp)
        elif file_type in {"dat", "data"}:
            tmp = dict(data_cfg)
            tmp.setdefault("sep", None)
            df = _load_csv_like(raw_path, tmp)
        elif file_type == "parquet":
            df = pd.read_parquet(raw_path)
        elif file_type in {"feather"}:
            df = pd.read_feather(raw_path)
        elif file_type in {"excel", "xlsx", "xls"}:
            sheet = data_cfg.get("sheet_name", 0)
            excel_cfg = data_cfg.get("excel") or {}
            header = excel_cfg.get("header", 0)
            try:
                engine = "xlrd" if file_type == "xls" else None
                df = pd.read_excel(raw_path, sheet_name=sheet, header=header, engine=engine)
            except ImportError as e:  # pragma: no cover - 依赖问题
                raise ImportError("读取Excel失败，可能缺少xlrd>=2.0.1，请安装后重试") from e
        elif file_type in {"json", "jsonl"}:
            df = pd.read_json(raw_path, lines=True)
        else:
            raise ValueError(f"未知的 file_type={file_type}")

    if dataset_name == "telco_churn" and "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df.columns = [str(c).replace("\ufeff", "").replace("\xa0", " ").strip() for c in df.columns]

    df, target_col = _apply_target_transform(df, data_cfg)

    feature_cols = data_cfg.get("feature_cols")
    drop_cols = set(data_cfg.get("drop_cols", []) or [])
    split_col = "split" if "split" in df.columns else None
    cols_to_keep = set(df.columns)
    if feature_cols:
        cols_to_keep = set(feature_cols + [target_col])
        if split_col:
            cols_to_keep.add(split_col)
    df = df[[c for c in df.columns if c in cols_to_keep and c not in drop_cols]].copy()

    positive_label = data_cfg.get("positive_label")
    negative_label = data_cfg.get("negative_label")
    if not target_mapped and positive_label is not None and target_col in df.columns:
        df = _map_target_with_config(df, target_col, data_cfg)
        target_mapped = True

    if positive_label is not None and target_col in df.columns:
        train_mask = df["split"].str.lower() == "train" if "split" in df.columns else None
        test_mask = df["split"].str.lower() == "test" if "split" in df.columns else None
        pos_rate = (df[target_col] == 1).mean()
        msg = f"【数据集信息】名称={dataset_name}，样本数={len(df)}，目标列={target_col}，正类比例={pos_rate:.2%}"
        if train_mask is not None:
            msg += f"；训练集正类比例={(df.loc[train_mask, target_col] == 1).mean():.2%}"
        if test_mask is not None:
            msg += f"；测试集正类比例={(df.loc[test_mask, target_col] == 1).mean():.2%}"
        log_info(msg)
    else:
        log_info(f"【数据集信息】名称={dataset_name}，样本数={len(df)}，目标列={target_col}")

    return df, target_col
