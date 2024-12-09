import argparse
import pandas as pd
import os
from helpers import *
from sklearn.impute import SimpleImputer
from loguru import logger

parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-IV CSV files.')

parser.add_argument('--master_dataset_path', type=str, help='Directory containing "master_dataset.csv"',required=True)
parser.add_argument('--output_path', type=str, help='Output directory for filtered and processed "train.csv" and "test.csv"',required=True)

args, _ = parser.parse_known_args()

master_dataset_path = args.master_dataset_path
split_path = "data/subject_splits.parquet"
output_path = args.output_path

# from mimic-extract
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


# Reading master_dataset.csv
logger.info("Opening Master dataset")
df_master = pd.read_csv(os.path.join(master_dataset_path, 'master_dataset.csv'))
logger.info("Opening Split file")
df_split = pd.read_parquet(split_path)

# Filtering for non-null triage acuity
logger.info(f'Before filtering for non-null "triage_acuity" : master dataset size = {len(df_master)}')
df_master = df_master[df_master['triage_acuity'].notnull()]
logger.info(f'After filtering for non-null "triage_acuity" : master dataset size = {len(df_master)}')

# Outlier detection and removal
logger.info("Converting temperature to celsius")
df_master = convert_temp_to_celcius(df_master)
logger.info("Removing outliers")
df_master = remove_outliers(df_master, vitals_valid_range)

merged_df = pd.merge(df_master, df_split, on='subject_id', how='inner')
df_train = merged_df[merged_df['split'] == 'train'].drop(columns=["split"])
df_test = merged_df[merged_df['split'] == 'test'].drop(columns=["split"])

#print(df_train.head())

# Missing value imputation
logger.info("Imputing missing values")
df_missing_stats = df_train.isnull().sum().to_frame().T
df_missing_stats.loc[1] = df_missing_stats.loc[0] / len(df_master)
df_missing_stats.index = ['no. of missing values', 'percentage of missing values']

vitals_cols = [col for col in df_master.columns if len(col.split('_')) > 1 and 
                                                   col.split('_')[1] in vitals_valid_range]

imputer = SimpleImputer(strategy='median')
df_train[vitals_cols] = imputer.fit_transform(df_train[vitals_cols])
df_test[vitals_cols] = imputer.transform(df_test[vitals_cols])

logger.info("Adding score values for scoring systems")
# Adding Score values for train and test
add_triage_MAP(df_test) # add an extra variable MAP
add_score_CCI(df_test)
add_score_CART(df_test)
add_score_REMS(df_test)
add_score_NEWS(df_test)
add_score_NEWS2(df_test)
add_score_MEWS(df_test)

add_triage_MAP(df_train) # add an extra variable MAP
add_score_CCI(df_train)
add_score_CART(df_train)
add_score_REMS(df_train)
add_score_NEWS(df_train)
add_score_NEWS2(df_train)
add_score_MEWS(df_train)

# Output train and test csv
logger.info("Saving train and test")
df_train.to_csv(os.path.join(output_path, 'train.csv'), index=False)
df_test.to_csv(os.path.join(output_path, 'test.csv'), index=False)