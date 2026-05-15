import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from .utils_logging import log_info


def _infer_columns(df: pd.DataFrame, target_col: str):
    continuous_cols = df.drop(columns=[target_col]).select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.drop(columns=[target_col]).select_dtypes(exclude=["number"]).columns.tolist()
    return continuous_cols, categorical_cols


def _apply_credit_default_feature_engineering(df: pd.DataFrame, fe_cfg: dict) -> pd.DataFrame:
    """按配置生成信用卡违约相关的派生特征。"""

    df = df.copy()
    pay_cols = fe_cfg.get("pay_status_cols") or []
    missing_pay_cols = [col for col in pay_cols if col not in df.columns]
    if missing_pay_cols:
        raise KeyError(f"缺少信用卡逾期状态列：{', '.join(missing_pay_cols)}")

    ever_delay_col = fe_cfg.get("derive_ever_delay_col", "ever_delay")
    max_delay_col = fe_cfg.get("derive_max_delay_col", "max_delay")
    max_delay_bin_col = fe_cfg.get("derive_max_delay_bin_col", "max_delay_bin")
    max_delay_bins = fe_cfg.get("max_delay_bins") or []
    max_delay_labels = fe_cfg.get("max_delay_labels")

    pay_status_df = df[pay_cols]
    df[ever_delay_col] = (pay_status_df > 0).any(axis=1).astype(int)

    df[max_delay_col] = pay_status_df.clip(lower=0).max(axis=1).astype(int)
    df[max_delay_bin_col] = (
        pd.cut(df[max_delay_col], bins=max_delay_bins, labels=max_delay_labels, include_lowest=True)
        .astype(object)
        .fillna("unknown")
    )

    log_info("已生成 credit_default 派生特征：ever_delay / max_delay / max_delay_bin")
    log_info(f"ever_delay 分布：\n{df[ever_delay_col].value_counts(dropna=False).to_string()}")
    log_info(f"max_delay_bin 分布：\n{df[max_delay_bin_col].value_counts(dropna=False).to_string()}")
    log_info(f"max_delay_bins={max_delay_bins}, labels={max_delay_labels}")
    return df


def apply_feature_engineering_with_config(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """根据配置执行可选的特征工程，返回拷贝后的 DataFrame。"""

    prep_cfg = cfg.get("PREPROCESS", {})
    fe_cfg = prep_cfg.get("feature_engineering") or {}
    if fe_cfg.get("enable_credit_default_derived"):
        return _apply_credit_default_feature_engineering(df, fe_cfg)
    return df.copy()


def prepare_features_and_labels(df: pd.DataFrame, cfg: dict):
    """
    返回 X, y, meta：
    - X: 编码后的特征矩阵
    - y: 0/1 标签
    - meta: 特征名等辅助信息
    """
    prep_cfg = cfg.get("PREPROCESS", {})
    data_cfg = cfg.get("DATA", {})
    target_col = data_cfg.get("target_col", "income")
    target_transform = data_cfg.get("target_transform") or {}
    target_col_for_model = target_transform.get("new_col", target_col)
    positive_label = data_cfg.get("positive_label", ">50K")
    negative_label = data_cfg.get("negative_label")

    df = df.copy()
    # 缺失值统一处理
    if prep_cfg.get("handle_missing") == "question_mark":
        df.replace("?", np.nan, inplace=True)
    elif prep_cfg.get("handle_missing") == "simple":
        strategy = prep_cfg.get("fillna_strategy", "most_frequent")
        if strategy == "most_frequent":
            fill_values = df.mode().iloc[0]
            df.fillna(fill_values, inplace=True)
        elif strategy == "median":
            numeric_median = df.median(numeric_only=True)
            df.fillna(numeric_median, inplace=True)
        elif strategy == "mean":
            numeric_mean = df.mean(numeric_only=True)
            df.fillna(numeric_mean, inplace=True)
        elif strategy == "zero":
            df.fillna(0, inplace=True)
        elif strategy == "drop":
            df.dropna(inplace=True)
        log_info(f"【预处理】缺失值填充策略={strategy}")

    # 特征工程：信用卡违约派生特征
    df = apply_feature_engineering_with_config(df, cfg)

    # 推断列
    continuous_cols = (
        prep_cfg.get("continuous_cols")
        or prep_cfg.get("numeric")
        or prep_cfg.get("numeric_cols")
        or []
    )
    categorical_cols = (
        prep_cfg.get("categorical_cols")
        or prep_cfg.get("categorical")
        or []
    )
    if not continuous_cols and not categorical_cols:
        continuous_cols, categorical_cols = _infer_columns(df, target_col_for_model)
    log_info(f"【预处理】连续特征={len(continuous_cols)}个，类别特征={len(categorical_cols)}个")

    target_series = df[target_col_for_model]
    # 若目标列已经是 0/1 数值标签，直接复用，避免二次转换导致全为 0
    if set(pd.unique(target_series.dropna())) <= {0, 1}:
        y = target_series.astype(int).values
    else:
        y = (target_series == positive_label).astype(int).values
        if negative_label is not None:
            y = np.where(target_series == positive_label, 1, 0)
    drop_cols = set(prep_cfg.get("drop_cols", []))
    drop_cols.add(target_col_for_model)
    source_target_col = data_cfg.get("target_col")
    if source_target_col and source_target_col != target_col_for_model:
        drop_cols.add(source_target_col)
    X_raw = df.drop(columns=list(drop_cols), errors="ignore")

    transformers = []
    impute_cfg = prep_cfg.get("impute_strategy", {}) or {}
    cat_strategy = impute_cfg.get("categorical")
    num_strategy = impute_cfg.get("continuous")

    if categorical_cols:
        cat_steps = []
        if cat_strategy:
            cat_steps.append(("imputer", SimpleImputer(strategy=cat_strategy)))
        encoder = OneHotEncoder(
            drop="first" if prep_cfg.get("drop_first") else None, handle_unknown="ignore"
        )
        cat_steps.append(("encoder", encoder))
        transformers.append(("cat", Pipeline(cat_steps), categorical_cols))
    if continuous_cols:
        num_steps = []
        if num_strategy:
            num_steps.append(("imputer", SimpleImputer(strategy=num_strategy)))
        scaler_type = prep_cfg.get("scaler_type", "standard")
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        num_steps.append(("scaler", scaler))
        transformers.append(("num", Pipeline(num_steps), continuous_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    pipeline = Pipeline(steps=[("preprocess", preprocessor)])
    X = pipeline.fit_transform(X_raw)

    # 确保输出的特征矩阵 X 为可写的 numpy 数组（避免后续模型报 buffer read-only 错误）
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    if not X.flags.writeable:
        X = np.array(X, copy=True)

    # 生成特征名
    feature_names = []
    if categorical_cols:
        cat_encoder: OneHotEncoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
        feature_names.extend(cat_encoder.get_feature_names_out(categorical_cols).tolist())
    if continuous_cols:
        feature_names.extend(continuous_cols)

    log_info(f"【预处理】编码后维度={X.shape[1]}")

    meta = {
        "feature_names": feature_names,
        "continuous_cols": continuous_cols,
        "categorical_cols": categorical_cols,
        "preprocess_pipeline": pipeline,
        "df_processed": df,
    }
    return X, y, meta
