import pandas as pd
import numpy as np

np.random.seed(42)

# Generate synthetic sensor data with late-stage accelerated degradation
dates = pd.date_range(start='2024-01-01 00:00:00', periods=1000, freq='1min')
n = len(dates)
t = np.arange(n)

degradation = np.zeros(n)
degradation[int(0.7*n):] = np.linspace(0, 1, n - int(0.7*n))

vibration = (
    0.1
    + 0.02 * np.sin(t/100)
    + 0.01 * np.random.randn(n)
    + 0.0001 * t
    + 0.05 * degradation
)

temperature = (
    45
    + 0.5 * np.sin(t/200)
    + 0.3 * np.random.randn(n)
    + 0.01 * t
    + 5 * degradation
)

pressure = (
    101.3
    + 0.2 * np.sin(t/150)
    + 0.1 * np.random.randn(n)
    - 0.0005 * t
    - 2 * degradation
)

df = pd.DataFrame({
    'timestamp': dates,
    'vibration': vibration,
    'temperature': temperature,
    'pressure': pressure
})

df.to_csv('data/sensor_data.csv', index=False)
print(f'Generated sensor_data.csv with {len(df)} rows')
