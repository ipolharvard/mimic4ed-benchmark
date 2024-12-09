from scripts.helpers import run_benchmark_taskMIMIC-IV-ED Benchmark
=========================

This is an updated version of MIMIC-IV-ED benchmark based
on [mimic4ed-benchmark](https://github.com/nliulab/mimic4ed-benchmark) repository and
the [Scientific Data](https://www.nature.com/articles/s41597-022-01782-9) article.

## Structure

`scripts/...` contains the scripts for benchmark dataset generation (master_data.csv), data split into training and
testing sets and the code for building the various task-specific benchmark models.

## Requirements and Setup

MIMIC-IV-ED and MIMIC-IV databases are not provided with this repository and are **required** for this workflow.
MIMIC-IV-ED can be downloaded
from [https://physionet.org/content/mimic-iv-ed/2.2/](https://physionet.org/content/mimic-iv-ed/2.2/) and MIMIC-IV can
be downloaded from [https://physionet.org/content/mimiciv/2.2/](https://physionet.org/content/mimiciv/2.2/).

***NOTE** Upon downloading and extracting the MIMIC databases from their compressed files, the directory
`/mimic-iv-ed-2.2/ed` should be moved/copied to the directory containing MIMIC-IV data `/mimic-iv-2.2`.

Set up the python environment and run:

```shell
pip install -r requirements.txt
```

## Workflow

### 1. Benchmark Data Generation

~~~
python extract_master_dataset.py --mimic4_path {mimic4_path} --output_path {output_path}
~~~

**Arguments**:

- `mimic4_path` : Path to directory containing MIMIC-IV data. Refer to [Requirements and Setup](#requirements-and-setup)
  for details.
- `output_path ` : Path to output directory.
- `icu_transfer_timerange` : Timerange in hours for ICU transfer outcome. Default set to 12.
- `next_ed_visit_timerange` : Timerange in days days for next ED visit outcome. Default set to 3.

**Output**:

`master_dataset.csv` output to `output_path`

**Details**:

The input `edstays.csv` from the MIMIC-IV-ED database is taken to be the root table, with `subject_id` as the unique
identifier for each patient and `stay_id` as the unique identifier for each ED visit. This root table is then merged
with other tables from the main MIMIC-IV database to capture an informative array of clinical variables for each
patient.

A total of **81** variables are included in `master_dataset.csv` (Refer to Table 3 in
the [Scientific Data](https://www.nature.com/articles/s41597-022-01782-9) article for full variable list).

### 2. Cohort Filtering and Data Processing

~~~
python data_general_processing.py --master_dataset_path {master_dataset_path} --output_path {output_path}
~~~

**Arguments**:

- `master_dataset_path` : Path to directory containing "master_dataset.csv".
- `output_path` : Path to output directory.

**Output**:

`train.csv` and `test.csv` output to `output_path`

**Details**:

Outlier values in vital sign and lab test variables are detected using an identical method
to [Wang et al.](https://github.com/MLforHealth/MIMIC_Extract), with outlier thresholds defined previously
by [Harutyunyan et al.](https://github.com/YerevaNN/mimic3-benchmarks) Outliers are then imputed with the neareset valid
value.

The data is then split into `train.csv` and `test.csv`  based on a `subject_splits.parquet` and clinical scores for each
patient are then added as additional variables.

### 3. Prediction Task Selection and Model evaluation

To run benchmarks for one of the tasks (_hospitalization_, _critical_outcome_ or _ed_reattendance_) execute the
following
command:

```shell
python run_benchmark.py --input_path /path/to/inputs --output_path /path/to/outputs --task hospitalization
```

**Arguments**:

- `input_path` : Path to directory containing `train.csv` and `test.csv` and all files generated in steps 1. and 2.
- `output_path` : Path to directory for saving results.
- `task` : Name of the task (one of: _hospitalization_, _critical_outcome_ or _ed_reattendance_)

**Output**:

`results_{task}.csv` : Results of all the methods trained for the specific task.

Note: The AutoScore method is implemented in R. We provide an R script (`scripts/AutoScore.R`) than can be used for
generating results of the AutoScore method on benchmark tasks.

## Based on:

[MIMIC-IV-ED GitHub repository](https://github.com/nliulab/mimic4ed-benchmark)

Xie F, Zhou J, Lee JW, Tan M, Li SQ, Rajnthern L, Chee ML, Chakraborty B, Wong AKI, Dagan A, Ong MEH, Gao F, Liu N.
Benchmarking emergency department prediction models with machine learning and public electronic health records.
Scientific Data 2022 Oct; 9: 658. <https://doi.org/10.1038/s41597-022-01782-9>


