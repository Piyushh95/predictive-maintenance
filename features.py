# import pandas as pd

# def create_rolling_features(df, window=20):
#     features = df.copy()

#     for col in ['vibration', 'temperature', 'pressure']:
#         features[f'{col}_mean'] = features[col].rolling(window).mean()
#         features[f'{col}_std'] = features[col].rolling(window).std()

#     features = features.dropna()
#     return features
import pandas as pd

def create_rolling_features(df, sensor_cols=None, window=20, drop_na=True):
    """
    Create rolling mean and standard deviation features for sensor columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a 'timestamp' column and sensor columns.
    sensor_cols : list of str, optional
        Names of sensor columns to compute rolling statistics for.
        If None, defaults to ['vibration', 'temperature', 'pressure'].
    window : int, default 20
        Size of the rolling window (number of rows).
    drop_na : bool, default True
        Whether to drop rows that contain NaN values after computing
        rolling statistics (e.g., the first `window - 1` rows).

    Returns
    -------
    pandas.DataFrame
        DataFrame sorted by 'timestamp' with original columns plus rolling
        mean and standard deviation features for each sensor column.

    Raises
    ------
    ValueError
        If any required columns (sensor columns or 'timestamp') are missing.
    """
    if sensor_cols is None:
        sensor_cols = ['vibration', 'temperature', 'pressure']

    required_cols = set(sensor_cols) | {'timestamp'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Input DataFrame is missing required columns: {sorted(missing_cols)}"
        )

    # Ensure DataFrame is sorted by timestamp
    if not df['timestamp'].is_monotonic_increasing:
        df = df.sort_values('timestamp')

    features = df.copy()

    for col in sensor_cols:
        rolling = df[col].rolling(window=window, min_periods=window)
        features[f'{col}_mean'] = rolling.mean()
        features[f'{col}_std'] = rolling.std()

    if drop_na:
        features = features.dropna()

    return features