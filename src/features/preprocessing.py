import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

def preprocess_node_features(
    X,
    nan_threshold: float = 0.5,
    variance_threshold: float = 0.01,
    scale: bool = True,
    verbose: bool = False
) -> np.ndarray:
    """
    Принимает pd.DataFrame или scipy.sparse.csr_matrix.
    Возвращает плотный np.ndarray float32.
    """
    if sp.issparse(X):                     # Twitch, LastFM (one-hot sparse)
        if verbose:
            print("  Processing sparse features...")
        # 1. VarianceThreshold (без NaN)
        if variance_threshold > 0:
            selector = VarianceThreshold(threshold=variance_threshold)
            try:
                X = selector.fit_transform(X)
            except ValueError:
                # Если все фичи константны, возвращаем исходные без фильтрации
                if verbose:
                    print("  All features constant, keeping original")
        # 2. Масштабирование без центрирования
        if scale:
            scaler = StandardScaler(with_mean=False)
            X = scaler.fit_transform(X)
        else:
            X = X.copy()
        return X.toarray().astype(np.float32)

    # DataFrame (Cora, PubMed, ogbn-arxiv)
    df = X.copy()
    original_cols = df.shape[1]

    # 1. Удаление колонок с высоким процентом NaN
    nan_frac = df.isna().mean()
    cols_to_drop = nan_frac[nan_frac > nan_threshold].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        if verbose:
            print(f"  Dropped {len(cols_to_drop)} columns with >{nan_threshold*100}% NaNs")

    # 2. Заполнение оставшихся NaN
    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "missing")
        else:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)

    # 3. Кодирование категориальных признаков
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Приводим все колонки к числовому типу
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].astype(float)

    # 4. Удаление константных признаков
    if variance_threshold > 0:
        selector = VarianceThreshold(threshold=variance_threshold)
        try:
            X_temp = selector.fit_transform(df.values)
            kept = selector.get_support()
            removed = np.where(~kept)[0]
            if verbose and len(removed):
                print(f"  Removed {len(removed)} constant/near-constant features")
            df = pd.DataFrame(X_temp, index=df.index, columns=np.array(df.columns)[kept])
        except ValueError:
            if verbose:
                print("  VarianceThreshold failed (possibly all constant), skipping")

    # 5. Масштабирование
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.values)
    else:
        X_scaled = df.values

    if verbose:
        print(f"  Preprocessing finished: {original_cols} columns -> {X_scaled.shape[1]} features")

    return X_scaled.astype(np.float32)