# Real-World Smartphone PPG Dataset and Machine Learning code

This repository contains the official implementation of the paper:  
**"Large-Scale Real-World Smartphone Photoplethysmography Datasets for Vascular Assessment"**  
(doi: https://doi.org/10.3390/electronics1010000)

It includes:
- Two smartphone PPG datasets (classification + regression)
- Preprocessing pipelines for signal normalization, second derivatives, and Gaussian fitting (2G, 3G)
- Benchmark scripts for deep learning and ensemble models
- Utility functions for signal processing, evaluation, and visualization

## Datasets

### Download & Extraction
The datasets are available in the `data/` folder as archived zip files.  
**Important:** After cloning the repository, you **must extract** the archives before running any script:

```bash
cd data
unzip rws_ppg_classification_dataset.zip
unzip rws_ppg_regression_dataset.zip
```

**Expected structure after extraction:**
```bash
data/
├── rws_ppg_classification_dataset.csv
├── rws_ppg_regression_dataset.csv
├── rws_ppg_regression_dataset.zip
└── rws_ppg_classification_dataset.zip
```

Script **code/classification_prepare_trainig_data.py** will create classification dataset **data/train_rws_ppg_classification_dataset_gauss.csv** with following columns: 
*index,cnt,id,data_id,app_id,age,gender,hr,rmssd,class,template_ppg,ppg,class1,class2,class3,class4,template_ppg_norm,sd_template_ppg_norm,class_label,amp1_2g,mean1_2g,sigma1_2g,amp2_2g,mean2_2g,sigma2_2g,amp1_3g,mean1_3g,sigma1_3g,amp2_3g,mean2_3g,sigma2_3g,amp3_3g,mean3_3g,sigma3_3g*

Script **code/regression_prepare_trainig_data.py** will create regression dataset **data/train_rws_ppg_classification_dataset_gauss.csv** with following columns: 
*id,age,sex,hr,rmssd,data_id,app_id,ppg_signal,template_ppg,beat_corr_ratio,interbeat_corr_ratio,total_area_3g,peak_distance_12,peak_distance_13,peak_distance_23,sd_template_ppg,amp1_2g,mean1_2g,sigma1_2g,amp2_2g,mean2_2g,sigma2_2g,gauss_fit,od,do,a_index,a_value,b_index,b_value,a/b_ratio,amp1_3g,mean1_3g,sigma1_3g,amp2_3g,mean2_3g,sigma2_3g,amp3_3g,mean3_3g,sigma3_3g,gauss3_fit,ratio_3g_12,ratio_3g_13,ratio_3g_23,total_area_3g,peak_distance_12,peak_distance_13,peak_distance_23,percentile_25,percentile_75,triangular_index,mean1_2gsd,median,skewness,kurtosis,signal_length*

**Expected structure after extraction:**
```bash
data/
├── train_rws_ppg_regression_dataset.csv
├── train_rws_ppg_classification_dataset.csv
├── rws_ppg_classification_dataset.csv
├── rws_ppg_regression_dataset.csv
├── rws_ppg_regression_dataset.zip
└── rws_ppg_classification_dataset.zip
```

All CSV files are standard comma-separated plain text files and can be opened with any text editor, spreadsheet software (Excel, LibreOffice), or loaded via pandas.read_csv().
If you encounter issues opening the files, please verify that the zip archives are fully extracted and that the file paths match the script defaults.

## Code
After preparing the datasets, you can run the benchmark scripts:

**Classification Benchmarks**
The following scripts evaluate morphological classification performance using deep learning and ensemble methods on raw PPG, second derivatives, and Gaussian‑based features.  
| Script | Description |
|--------|-------------|
| `classification_dl_ml_benchmark.py` | Deep learning (CNN, ResNet, MLP) on raw PPG and second derivatives |
| `classification_ensemble_ml_benchmark.py` | Random Forest, Extra Trees, XGBoost on Gaussian features |
| `classification_xgboost_grid_search_eval.py` | Hyperparameter tuning and evaluation |
| `classification_2g_vs_3g.py` | Comparison of 2‑Gaussian vs. 3‑Gaussian representations |
| `classification_dataset_stats.py` | Descriptive statistics of the classification dataset |

**Regression Benchmark**  
The following scripts assess chronological age prediction performance using deep learning and ensemble regressors on waveform representations and extended feature sets.
| Script | Description |
|--------|-------------|
| `regression_dl_ml_benchmark.py` | Deep learning models for age regression |
| `regression_ensemble_ml_benchmark.py` | Ensemble models on Gaussian + statistical features |
| `regression_2g_vs_3g.py` | Impact of Gaussian model order on regression performance |
| `regression_dataset_stat.py` | Demographic and signal quality statistics |
| `regression_dataset_collect.py` | Collect and balance samples for regression |


---
If you use these datasets, code, or any part of this repository in your research, please cite the original paper with doi: https://doi.org/10.3390/electronics1010000

### Real-World Smartphone PPG Dataset and Machine Learning Code

[![DOI](https://img.shields.io/badge/DOI-10.3390/electronics1010000-blue)](https://doi.org/10.3390/electronics1010000)
[![License](https://img.shields.io/badge/License-Research%20Only-red)]()

This repository contains the official implementation and datasets accompanying the paper:

> **Jokić, S., Gligorić, N., Kartali, A., & Machidon, O. (2026).**  
> *Large-Scale Real-World Smartphone Photoplethysmography Datasets for Vascular Assessment.*  
> **Electronics**, , . https://doi.org/10.3390/electronics1010000

**If you use this code, datasets, or any derivative work in your research, you must cite the above paper.**

## Contact & Correspondence  
Stevan Jokić  
Faculty of Information Technology, Alfa BK University, Belgrade, Serbia  
stevan.jokic@alfa.edu.rs
