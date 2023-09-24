import pandas as pd

train = pd.read_csv(
    r'future-1\scenario17_dev_series_test.csv')
mmWave = pd.read_csv(r'unit1\mmWave_data\mmWave_power_1.txt')

print(mmWave)
