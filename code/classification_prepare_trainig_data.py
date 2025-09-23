"""
The script adds columns for normalized PPG templates, second derivative of PPG signals, and Gaussian curve fitting parameters with both 2- and 3-Gaussian models
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
import util

inputFn = "data/rws_ppg_classification_dataset.csv"
outputFn = "data/train_rws_ppg_classification_dataset.csv"

startInd = 13
endInd = 83

fs = 100  # Sampling frequency
fc = 10   # Cutoff frequency for low-pass filter
ord = 5   # Filter order

initial_guess_2g = [1, 15, 7, 0.6, 35, 8]
initial_guess_3g = [1, 15, 7, 0.6, 35, 8, 0.4, 50, 5]
x_data = np.array(range(endInd - startInd))

def processrow(row):
    # Process and normalize PPG signal
    processed_ppg = util.process_ppg(row["template_ppg"])
    y_data = util.normalize_0_1(processed_ppg[startInd:endInd])
    
    # Extract normalized PPG template for the entire segment
    template_ppg_norm = util.normalize_0_1(processed_ppg)
    sd_ppg = util.normalize_m1_1(util.sd5(processed_ppg))[startInd-3:endInd]

    # Fit with 2 Gaussians
    try:
        params_2g, _ = curve_fit(util.sum_of_2_gaussians, x_data, y_data,
                                 p0=initial_guess_2g, jac=util.jacobian_2g, maxfev=5000)
    except:
        params_2g = [np.nan] * 6

    # Fit with 3 Gaussians
    try:
        params_3g, _ = curve_fit(util.sum_of_3_gaussians, x_data, y_data,
                                 p0=initial_guess_3g, jac=util.jacobian_3g, maxfev=5000)
    except:
        params_3g = [np.nan] * 9

    # Map class 1–4 to label 0–3
    class_label = int(row["class"]) - 1
    
    # Return original row data + normalized PPG template + class label + Gaussian parameters
    return list(row) + [util.array2str(template_ppg_norm)] + [util.array2str(sd_ppg)] + [class_label] + list(params_2g) + list(params_3g)

# Load dataset
df = pd.read_csv(inputFn)
results = []

# Progress bar setup
total_rows = len(df)
print_interval = max(1, total_rows // 20)

# Iterate through rows
for index, row in df.iterrows():
    results.append(processrow(row))

    if index % print_interval == 0 or index == total_rows - 1:
        progress = (index + 1) / total_rows * 100
        print(f"Processed: {index + 1}/{total_rows} rows ({progress:.1f}%)")

# Create new DataFrame with results
columns = list(df.columns) + ["template_ppg_norm"] + ["sd_template_ppg_norm"] + ["class_label"] + [
    "amp1_2g", "mean1_2g", "sigma1_2g",
    "amp2_2g", "mean2_2g", "sigma2_2g",
    "amp1_3g", "mean1_3g", "sigma1_3g",
    "amp2_3g", "mean2_3g", "sigma2_3g",
    "amp3_3g", "mean3_3g", "sigma3_3g"
]
df_results = pd.DataFrame(results, columns=columns)

# Save results to CSV
df_results.to_csv(outputFn, index=True, index_label='index')

print("Fitting completed! Results saved to:", outputFn) 