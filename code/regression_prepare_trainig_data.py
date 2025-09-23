# Processing PPG data for SD, 2G and 3G Gaussian fitting
# Combined script for feature extraction from PPG waveforms

import traceback
import pandas as pd
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import util
import ast

# File paths
input_fn = "data/rws_ppg_regression_dataset.csv"
out_fn = "data/train_rws_ppg_regression_dataset.csv"

sist_peak_ind = 45  # Index of systolic peak in template

print("Loading data...")
df = pd.read_csv(input_fn)

g3errorsCnt = 0
g3ProcessedCnt = 0

def triangular_index(signal):
    signal = signal - np.min(signal)
    area = np.sum(signal)
    peak = np.max(signal)
    return (area / len(signal)) / (peak + 1e-8)

def process_ppg_complete(row):
    """
    Complete PPG processing: SD filtering, 2G and 3G Gaussian fitting in one function.
    """
    global g3errorsCnt, g3ProcessedCnt
    
    # Initialize results dictionary with NaN values
    results = {
        # SD results
        'sd_template_ppg': '',
        
        # 2G results
        'amp1_2g': np.nan, 'mean1_2g': np.nan, 'sigma1_2g': np.nan,
        'amp2_2g': np.nan, 'mean2_2g': np.nan, 'sigma2_2g': np.nan,
        'gauss_fit': '', 'od': np.nan, 'do': np.nan,
        'a_index': np.nan, 'a_value': np.nan, 'b_index': np.nan, 'b_value': np.nan,
        'a/b_ratio': np.nan,
        
        # 3G results
        'amp1_3g': np.nan, 'mean1_3g': np.nan, 'sigma1_3g': np.nan,
        'amp2_3g': np.nan, 'mean2_3g': np.nan, 'sigma2_3g': np.nan,
        'amp3_3g': np.nan, 'mean3_3g': np.nan, 'sigma3_3g': np.nan,
        'gauss3_fit': '', 'ratio_3g_12': np.nan, 'ratio_3g_13': np.nan, 'ratio_3g_23': np.nan,

        'total_area_3g' : np.nan, 'peak_distance_12' : np.nan, 'peak_distance_13' : np.nan, 'peak_distance_23' : np.nan 
    }
    
    try:
        # Step 1: Process original PPG signal and apply SD filter
        ppg_array_full = util.str2arr(row['template_ppg'])
        ppg_array_full = util.nfFilt(15, 100, 5, ppg_array_full)
        
        # Apply SD5 
        sd_ppg = util.normalize_m1_1(util.sd5(ppg_array_full))
        results['sd_template_ppg'] = util.array2str(sd_ppg)
        
        # Step 2: Determine OD/DO range
        od, do = util.ppg_template_range(ppg_array_full, peakInd=sist_peak_ind, hr=row['hr'])
        results['od'] = od
        results['do'] = do
        
        # Step 3: Detect A and B points
        a_index, a_value, b_index, b_value = util.detect_ab_sdppg(ppg_array_full, sist_peak_ind, od)
        results['a_index'] = a_index
        results['a_value'] = a_value
        results['b_index'] = b_index
        results['b_value'] = b_value
        results['a/b_ratio'] = a_value / b_value if b_value != 0 else np.nan

        
        
        # Extract and normalize the segment for Gaussian fitting
        ppg_array = ppg_array_full[od:do]
        ppg_array = util.normalize_0_1(ppg_array)
        x_data = np.arange(len(ppg_array))
        
        results['percentile_25'] = np.percentile(ppg_array, 25)
        results['percentile_75'] = np.percentile(ppg_array, 75)
        results['triangular_index'] = triangular_index(ppg_array)
        results['mean1_2gsd'] = np.mean(ppg_array) / np.std(ppg_array) if np.std(ppg_array) != 0 else 0
        results['median'] = np.median(ppg_array)
        results['skewness'] = pd.Series(ppg_array).skew()
        results['kurtosis'] = pd.Series(ppg_array).kurtosis()    
        results['signal_length'] = len(ppg_array)
        
        # Step 4: Fit 2 Gaussians
        initial_guess_2g = [
            1,                      # amp1
            sist_peak_ind - od,     # mean1
            5,                      # sigma1
            0.5,                    # amp2
            sist_peak_ind - od + 24,# mean2
            8                       # sigma2
        ]
        
        params_2g, _ = curve_fit(
            util.sum_of_2_gaussians,
            x_data,
            ppg_array,
            p0=initial_guess_2g,
            jac=util.jacobian_2g,
            maxfev=5000
        )
        
        # Store 2G results
        results['amp1_2g'] = params_2g[0]
        results['mean1_2g'] = params_2g[1]
        results['sigma1_2g'] = params_2g[2]
        results['amp2_2g'] = params_2g[3]
        results['mean2_2g'] = params_2g[4]
        results['sigma2_2g'] = params_2g[5]
        results['gauss_fit'] = ', '.join(map(str, util.sum_of_2_gaussians(x_data, *params_2g)))
        
        # Step 5: Fit 3 Gaussians
        g3ProcessedCnt += 1
        
        initial_guess_3g = [
            0.8,                    # amp1
            sist_peak_ind - od,     # mean1
            5,                      # sigma1
            0.6,                    # amp2
            sist_peak_ind - od + 12,# mean2
            6,                      # sigma2
            0.4,                    # amp3
            sist_peak_ind - od + 24,# mean3
            8                       # sigma3
        ]
        
        params_3g, _ = curve_fit(
            util.sum_of_3_gaussians,
            x_data,
            ppg_array,
            p0=initial_guess_3g,
            jac=util.jacobian_3g,
            maxfev=7000
        )
        
        # Store 3G results
        results['amp1_3g'] = params_3g[0]
        results['mean1_3g'] = params_3g[1]
        results['sigma1_3g'] = params_3g[2]
        results['amp2_3g'] = params_3g[3]
        results['mean2_3g'] = params_3g[4]
        results['sigma2_3g'] = params_3g[5]
        results['amp3_3g'] = params_3g[6]
        results['mean3_3g'] = params_3g[7]
        results['sigma3_3g'] = params_3g[8]
        results['gauss3_fit'] = ', '.join(map(str, util.sum_of_3_gaussians(x_data, *params_3g)))
        results['ratio_3g_12'] = params_3g[0] / params_3g[3]  # amp1/amp2
        results['ratio_3g_13'] = params_3g[0] / params_3g[6]  # amp1/amp3
        results['ratio_3g_23'] = params_3g[3] / params_3g[6]  # amp2/amp3

        # Calculate additional composite features
        # print("Calculating composite features...")
        df['total_area_3g'] = (results['amp1_3g'] * results['sigma1_3g'] + 
                            results['amp2_3g'] * results['sigma2_3g'] + 
                            results['amp3_3g'] * results['sigma3_3g'])

        df['peak_distance_12'] = results['mean2_3g'] - results['mean1_3g']
        df['peak_distance_13'] = results['mean3_3g'] - results['mean1_3g']
        df['peak_distance_23'] = results['mean3_3g'] - results['mean2_3g']
        
    except Exception as e:
        g3errorsCnt += 1
        print(f"Processing failed for ID {row.get('id', 'unknown')}: {e}")
        if 'od' in locals() and 'do' in locals():
            print(f"od:do={od}:{do}")
        print(f"Errors: {g3errorsCnt} of {g3ProcessedCnt}\n")
    
    return pd.Series(results)

print("Processing PPG data (SD filtering + 2G + 3G fitting)...")
all_results = df.apply(process_ppg_complete, axis=1)
df = pd.concat([df, all_results], axis=1)

# Save final results
df.to_csv(out_fn, index=False)
print(f"Processing complete! Final dataset saved to {out_fn}")
print(f"Dataset shape: {df.shape}")
print(f"3G fitting errors: {g3errorsCnt} of {g3ProcessedCnt}")