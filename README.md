# Real-World Smartphone Dataset and Machine Learning code

## Datasets
Datasets are available under data folder, extract archived zip files under data folder.

Script **code/classification_prepare_trainig_data.py** will create classification dataset **data/train_rws_ppg_classification_dataset_gauss.csv** with following columns: 
*index,cnt,id,data_id,app_id,age,gender,hr,rmssd,class,template_ppg,ppg,class1,class2,class3,class4,template_ppg_norm,sd_template_ppg_norm,class_label,amp1_2g,mean1_2g,sigma1_2g,amp2_2g,mean2_2g,sigma2_2g,amp1_3g,mean1_3g,sigma1_3g,amp2_3g,mean2_3g,sigma2_3g,amp3_3g,mean3_3g,sigma3_3g*

Script **code/regression_prepare_trainig_data.py** will create regression dataset **data/train_rws_ppg_classification_dataset_gauss.csv** with following columns: 
*id,age,sex,hr,rmssd,data_id,app_id,ppg_signal,template_ppg,beat_corr_ratio,interbeat_corr_ratio,total_area_3g,peak_distance_12,peak_distance_13,peak_distance_23,sd_template_ppg,amp1_2g,mean1_2g,sigma1_2g,amp2_2g,mean2_2g,sigma2_2g,gauss_fit,od,do,a_index,a_value,b_index,b_value,a/b_ratio,amp1_3g,mean1_3g,sigma1_3g,amp2_3g,mean2_3g,sigma2_3g,amp3_3g,mean3_3g,sigma3_3g,gauss3_fit,ratio_3g_12,ratio_3g_13,ratio_3g_23,total_area_3g,peak_distance_12,peak_distance_13,peak_distance_23,percentile_25,percentile_75,triangular_index,mean1_2gsd,median,skewness,kurtosis,signal_length*

