import pandas as pd
import os
from helpers import read_vitalsign_table, remove_outliers, convert_temp_to_celcius
from sklearn.impute import SimpleImputer

mimic_iv_ed_path = '/mnt/mimic/mimic_data/mimic-iv-full/ed'
output_path = '/mnt/mimic/mgrzeszczyk/data/ed'
df_vitalsign = read_vitalsign_table(os.path.join(mimic_iv_ed_path, 'vitalsign.csv.gz'))

vitals_valid_range = {
    'temperature': {'outlier_low': 14.2, 'valid_low': 26, 'valid_high': 45, 'outlier_high':47},
    'heartrate': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 350, 'outlier_high':390},
    'resprate': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 300, 'outlier_high':330},
    'o2sat': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 100, 'outlier_high':150},
    'sbp': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 375, 'outlier_high':375},
    'dbp': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 375, 'outlier_high':375},
    'pain': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 10, 'outlier_high':10},
    'acuity': {'outlier_low': 1, 'valid_low': 1, 'valid_high': 5, 'outlier_high':5},
}

df_vitalsign['charttime'] = pd.to_datetime(df_vitalsign['charttime'])
df_vitalsign.sort_values('charttime', inplace=True)
df_vitalsign.drop('ed_rhythm', axis=1, inplace=True)
df_vitalsign.head()

grouped = df_vitalsign.groupby('stay_id')
resample_freq = '1H' # 1 hour

df_list = []
counter = 0
N = len(grouped)
for stay_id, stay_df in grouped:
    counter += 1
    stay_df.set_index('charttime', inplace=True)
    stay_df = stay_df.resample('1T', origin='start').interpolate(method='linear')\
                     .resample(resample_freq, origin='start').asfreq().ffill().bfill()
    if counter % 10000 == 0 or counter == N:
        print('%d/%d' % (counter, N), end='\r')
    df_list.append(stay_df)

df_vitalsign_resampled = pd.concat(df_list)
df_vitalsign_resampled = convert_temp_to_celcius(df_vitalsign_resampled)
df_vitalsign_resampled = remove_outliers(df_vitalsign_resampled, vitals_valid_range)
vitals_cols = [col for col in df_vitalsign_resampled.columns if len(col.split('_')) > 1 and
                                                                col.split('_')[1] in vitals_valid_range]
imputer = SimpleImputer(strategy='median')
df_vitalsign_resampled[vitals_cols] = imputer.fit_transform(df_vitalsign_resampled[vitals_cols])
df_vitalsign_resampled.to_csv(os.path.join(output_path, 'ed_vitalsign_'+resample_freq+'_resampled.csv'))