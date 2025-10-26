# This script implements a machine learning pipeline for evaluating
# regression models on the RWS PPG regression dataset for age prediction.
# It performs the following tasks:

# - Loads and preprocesses the dataset, using AB features, 3G features, and statistical features
# - Conducts exploratory data analysis (EDA), generating summary statistics, target distribution, and feature correlation plots
# - Evaluates RandomForest, XGBoost, ExtraTrees, and MLP regressors with hyperparameter tuning using GridSearchCV
# - Uses GroupKFold cross-validation (5 folds) to prevent data leakage across groups (app_id)
# - Computes comprehensive regression metrics (MAE, RMSE, R2, MAPE) with bootstrap confidence intervals
# - Performs permutation importance analysis to assess feature contributions
# - Tracks training time (including hyperparameter tuning) and prediction time per model and fold
# - Generates visualizations: predicted vs actual plots, residual plots, feature importance, and summary boxplots
# - Conducts statistical comparisons using Wilcoxon tests to evaluate model performance differences
# - Aggregates metrics across folds, reporting mean and standard deviation (including for training and prediction times)
# - Saves results as CSV, LaTeX tables, and high-resolution PNG plots for reproducibility and publication
# - Provides detailed analysis of correct and incorrect predictions with error breakdown by age ranges and PPG signal visualization
# - Includes Gaussian fit visualization and comprehensive error analysis for publication-quality results

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from scipy.stats import wilcoxon
import matplotlib.gridspec as gridspec
import ast
import warnings
warnings.filterwarnings('ignore')

# Feature definitions
FEATURE_SETS = {
    "AB_features": [
        'a_index', 'a_value', 'a_index_2start', 'a_index_2peak',
        'b_index', 'b_value', 'b_index_2start', 'b_index_2peak', 'a/b_ratio'
    ],
    "3G_features": [
        'amp1_3g', 'mean1_3g', 'sigma1_3g',
        'amp2_3g', 'mean2_3g', 'sigma2_3g',
        'amp3_3g', 'mean3_3g', 'sigma3_3g',
        'ratio_3g_12', 'ratio_3g_13', 'ratio_3g_23',
        'total_area_3g',
        'peak_distance_12', 'peak_distance_13', 'peak_distance_23'
    ],
    "statistical_features": [
        'percentile_25', 'percentile_75',
        'triangular_index', 'mean1_2gsd',
        'median', 'skewness', 'kurtosis',
        'signal_length'
    ]
}

# Combine all features
ALL_FEATURES = FEATURE_SETS["AB_features"] + FEATURE_SETS["3G_features"] + FEATURE_SETS["statistical_features"]

# Model definitions with hyperparameter tuning (including MLP)
MODEL_DEFS = {
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'param_grid': {
            'n_estimators': [50, 100],
            'max_depth': [10, 15],
            'min_samples_split': [5, 10]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42, n_jobs=-1),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [5, 7],
            'learning_rate': [0.1, 0.2],
        }
    },
    'ExtraTrees': {
        'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
        'param_grid': {
            'n_estimators': [50, 100],
            'max_depth': [10, 15],
        }
    },
    'MLP': {
        'model': MLPRegressor(random_state=42, max_iter=1000, early_stopping=True, validation_fraction=0.1),
        'param_grid': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'learning_rate_init': [0.001, 0.01],
            'alpha': [0.001, 0.01]
        }
    }
}

# Define line styles, markers, hatches, and colors for differentiation
MODEL_STYLES = {
    'RandomForest': {'linestyle': '-', 'marker': 'o', 'hatch': '/', 'color': '#66c2a5'},  # Green
    'XGBoost': {'linestyle': '--', 'marker': '^', 'hatch': '\\', 'color': '#fc8d62'},     # Orange
    'ExtraTrees': {'linestyle': ':', 'marker': 's', 'hatch': '|', 'color': '#8da0cb'},    # Blue
    'MLP': {'linestyle': '-.', 'marker': '*', 'hatch': '-', 'color': '#e78ac3'}           # Purple
}

# Age range definitions for error analysis
AGE_RANGES = {
    'under_30': (0, 30),
    '30_40': (30, 40),
    '40_50': (40, 50),
    '50_60': (50, 60),
    'over_60': (60, 150)
}

def get_age_range(age):
    """Get the age range category for a given age"""
    if age < 30:
        return 'under_30'
    elif age < 40:
        return '30_40'
    elif age < 50:
        return '40_50'
    elif age < 60:
        return '50_60'
    else:
        return 'over_60'

# Load data with error handling and data validation
def load_data(path):
    try:
        df = pd.read_csv(path)
        target_col = 'age'
        group_col = 'app_id'
        
        if not all(col in df.columns for col in [target_col, group_col] + ALL_FEATURES):
            missing = [col for col in [target_col, group_col] + ALL_FEATURES if col not in df.columns]
            raise ValueError(f"Required columns missing in dataset: {missing}")

        # Create a copy to avoid SettingWithCopyWarning
        df_clean = df[[target_col, group_col] + ALL_FEATURES].copy()
        df_clean = df_clean.dropna(subset=[target_col, group_col] + ALL_FEATURES)

        # Validate and convert feature columns to numeric
        for col in ALL_FEATURES:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='raise')
            except ValueError as e:
                print(f"Error: Column {col} contains non-numeric values: {e}")
                sys.exit(1)

        # Normalize features
        scaler = StandardScaler()
        df_clean.loc[:, ALL_FEATURES] = scaler.fit_transform(df_clean[ALL_FEATURES])

        return df_clean[ALL_FEATURES], df_clean[target_col], df_clean[group_col], df
    
    except FileNotFoundError:
        print(f"Error: File {path} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def parse_signal(signal_string):
    """Parse signal string to numpy array"""
    try:
        if isinstance(signal_string, str):
            # Remove quotes and convert to numpy array
            signal_string = signal_string.strip('"')
            signal_array = np.fromstring(signal_string, sep=',')
            return signal_array
        else:
            return np.array([])
    except Exception as e:
        print(f"Error parsing signal: {e}")
        return np.array([])

# Exploratory Data Analysis
def exploratory_data_analysis(df, results_dir):
    try:
        os.makedirs(results_dir, exist_ok=True)
        
        # Summary statistics
        summary_stats = df.describe()
        summary_stats.to_csv(f"{results_dir}/dataset_summary_stats.csv")
        
        # Target distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7, hatch='/')
        plt.title('Age Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Age (years)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{results_dir}/age_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature correlations with target
        numeric_cols = df[ALL_FEATURES + ['age']].select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        target_corr = corr_matrix['age'].drop('age').sort_values(ascending=False)
        
        plt.figure(figsize=(12, 10))
        target_corr.head(15).plot(kind='barh', color='lightcoral', edgecolor='black', hatch='/')
        plt.title('Top 15 Features Correlated with Age', fontsize=16, fontweight='bold')
        plt.xlabel('Pearson Correlation Coefficient', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/target_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in EDA: {e}")

# Bootstrap confidence intervals for regression metrics
def bootstrap_ci(y_true, y_pred, n_boot=500, alpha=0.95):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae_scores, rmse_scores, r2_scores = [], [], []
    
    for _ in range(n_boot):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        mae = mean_absolute_error(y_true[idx], y_pred[idx])
        rmse = np.sqrt(mean_squared_error(y_true[idx], y_pred[idx]))
        r2 = r2_score(y_true[idx], y_pred[idx])
            
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
    
    mae_ci = np.percentile(mae_scores, [(1-alpha)/2*100, (1+alpha)/2*100])
    rmse_ci = np.percentile(rmse_scores, [(1-alpha)/2*100, (1+alpha)/2*100])
    r2_ci = np.percentile(r2_scores, [(1-alpha)/2*100, (1+alpha)/2*100])
    
    return mae_ci, rmse_ci, r2_ci

# Feature importance analysis
def feature_importance_analysis(model, X, y, feature_cols, results_dir, model_name, fold, prefix):
    try:
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42, 
                                      scoring='r2', n_jobs=-1)
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': result.importances_mean,
            'std': result.importances_std
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(f"{results_dir}/{prefix}{model_name}_feature_importance_fold{fold}.csv", index=False)
        
        # Plot top 10 features
        top_features = importance_df.head(10)
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=MODEL_STYLES[model_name]['color'], edgecolor='black', 
                       hatch=MODEL_STYLES[model_name]['hatch'])
        plt.yticks(range(len(top_features)), top_features['feature'], fontsize=12)
        plt.title(f'Feature Importance - {model_name} (Fold {fold})', fontsize=16, fontweight='bold')
        plt.xlabel('Permutation Importance', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{prefix}{model_name}_feature_importance_fold{fold}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in feature importance for {model_name}, fold {fold}: {e}")

# Comprehensive error analysis with detailed signal visualization
def analyze_prediction_errors(y_true, y_pred, ppg_signals, gauss_signals, indices, model_name, fold, results_dir, prefix):
    """Analyze prediction errors by age range and error direction with detailed signal visualization"""
    try:
        errors = y_pred - y_true
        absolute_errors = np.abs(errors)
        
        # Create error analysis DataFrame
        error_df = pd.DataFrame({
            'actual_age': y_true,
            'predicted_age': y_pred,
            'error': errors,
            'absolute_error': absolute_errors,
            'age_range': [get_age_range(age) for age in y_true],
            'error_direction': ['overestimation' if err > 0 else 'underestimation' for err in errors],
            'index': indices
        })
        
        # Save detailed error analysis
        error_df.to_csv(f"{results_dir}/{prefix}{model_name}_error_analysis_fold{fold}.csv", index=False)
        
        return error_df
        
    except Exception as e:
        print(f"Error in prediction error analysis for {model_name}, fold {fold}: {e}")
        return pd.DataFrame()

# Create comprehensive signal visualization for publication
def create_comprehensive_signal_analysis(error_df, ppg_signals, gauss_signals, model_name, fold, results_dir, prefix):
    """Create comprehensive signal visualization for publication with 5 examples per age range and error direction"""
    try:
        # Create separate visualizations for overestimation and underestimation
        for error_direction in ['overestimation', 'underestimation']:
            direction_df = error_df[error_df['error_direction'] == error_direction]
            
            if direction_df.empty:
                continue
                
            # Create figure for this error direction
            fig = plt.figure(figsize=(20, 25))
            gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.3)
            
            age_ranges = ['under_30', '30_40', '40_50', '50_60', 'over_60']
            
            for row, age_range in enumerate(age_ranges):
                # Get top 5 worst predictions for this age range and error direction
                range_df = direction_df[direction_df['age_range'] == age_range]
                if len(range_df) > 0:
                    worst_cases = range_df.nlargest(5, 'absolute_error')
                    
                    for col in range(2):  # Two columns: PPG signal and Gaussian fit
                        if col == 0:
                            signal_type = 'PPG'
                            signals = ppg_signals
                            title_suffix = 'PPG Signal'
                            color = 'blue'
                        else:
                            signal_type = 'Gaussian'
                            signals = gauss_signals
                            title_suffix = 'Gaussian Fit'
                            color = 'red'
                        
                        ax = fig.add_subplot(gs[row, col])
                        
                        # Plot up to 5 cases
                        for idx, (_, case) in enumerate(worst_cases.iterrows()):
                            if idx >= 5:
                                break
                                
                            case_index = case['index']
                            if case_index < len(signals):
                                signal = signals[case_index]
                                if len(signal) > 0:
                                    time_axis = np.linspace(0, 1, len(signal))
                                    alpha = 0.7 - (idx * 0.1)  # Decreasing alpha for better visibility
                                    linewidth = 2.0 - (idx * 0.2)
                                    
                                    ax.plot(time_axis, signal, color=color, alpha=alpha, 
                                           linewidth=linewidth, label=f'Err: {case["error"]:.1f} yrs')
                        
                        ax.set_title(f'{age_range.replace("_", "-")} - {title_suffix}\n{error_direction.title()}', 
                                   fontsize=14, fontweight='bold')
                        ax.set_xlabel('Normalized Time', fontsize=12)
                        ax.set_ylabel('Amplitude (a.u.)', fontsize=12)
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize=10)
            
            plt.suptitle(f'Signal Analysis - {model_name} (Fold {fold})\n{error_direction.title()} Cases by Age Range', 
                        fontsize=20, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(f"{results_dir}/{prefix}{model_name}_{error_direction}_signals_fold{fold}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f"Error creating comprehensive signal analysis for {model_name}, fold {fold}: {e}")

# Create detailed case studies with both PPG and Gaussian signals
def create_detailed_case_studies(error_df, ppg_signals, gauss_signals, model_name, fold, results_dir, prefix):
    """Create detailed case studies with both original PPG and Gaussian fitted signals"""
    try:
        # Select top 4 worst cases from each error direction
        worst_over = error_df[error_df['error_direction'] == 'overestimation'].nlargest(4, 'absolute_error')
        worst_under = error_df[error_df['error_direction'] == 'underestimation'].nlargest(4, 'absolute_error')
        
        cases = pd.concat([worst_over, worst_under])
        
        if cases.empty:
            return
        
        # Create detailed case study visualization
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        
        for idx, (_, case) in enumerate(cases.iterrows()):
            if idx >= 8:
                break
                
            row = idx // 2
            col = idx % 2
            
            actual = case['actual_age']
            predicted = case['predicted_age']
            error = case['error']
            abs_error = case['absolute_error']
            direction = case['error_direction']
            age_range = case['age_range']
            case_index = case['index']
            
            # Get signals for this case
            ppg_signal = ppg_signals[case_index] if case_index < len(ppg_signals) else np.array([])
            gauss_signal = gauss_signals[case_index] if case_index < len(gauss_signals) else np.array([])
            
            if len(ppg_signal) > 0 and len(gauss_signal) > 0:
                # Plot both signals
                time_axis = np.linspace(0, 1, len(ppg_signal))
                
                axes[row, col].plot(time_axis, ppg_signal, 'b-', linewidth=2, alpha=0.7, label='PPG Signal')
                axes[row, col].plot(time_axis, gauss_signal, 'r--', linewidth=2, alpha=0.8, label='Gaussian Fit')
                
                # Customize based on error direction
                if direction == 'overestimation':
                    bg_color = '#FFE6E6'
                    title_color = 'darkred'
                else:
                    bg_color = '#E6F3FF'
                    title_color = 'darkblue'
                
                axes[row, col].set_facecolor(bg_color)
                axes[row, col].set_title(f'Age: {actual}→{predicted} (Error: {error:+.1f} yrs)\n'
                                       f'{direction.title()}, {age_range.replace("_", "-")}', 
                                       fontweight='bold', fontsize=12, color=title_color)
                axes[row, col].set_xlabel('Normalized Time', fontsize=11)
                axes[row, col].set_ylabel('Amplitude (a.u.)', fontsize=11)
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].legend(fontsize=10)
        
        plt.suptitle(f'Detailed Case Studies - {model_name} (Fold {fold})\n'
                    f'Comparison of Original PPG Signals and Gaussian Fits', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{prefix}{model_name}_detailed_case_studies_fold{fold}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save case studies to CSV
        cases.to_csv(f"{results_dir}/{prefix}{model_name}_case_studies_fold{fold}.csv", index=False)
        
    except Exception as e:
        print(f"Error creating detailed case studies for {model_name}, fold {fold}: {e}")

# Evaluation and export function
def evaluate_and_export(models, X_test, y_test, test_indices, ppg_signals, gauss_signals, fold, results_dir, prefix):
    summary = []
    all_error_dfs = []
    
    # Create subplots for visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create DataFrame to store plot data
    plot_data = []
    
    for name, model in models.items():
        try:
            # Measure prediction time
            start_pred = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - start_pred
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Bootstrap confidence intervals
            mae_ci, rmse_ci, r2_ci = bootstrap_ci(y_test, y_pred)
            
            metrics = {
                'fold': fold,
                'model': name,
                'mae': mae,
                'mae_ci_lower': mae_ci[0],
                'mae_ci_upper': mae_ci[1],
                'rmse': rmse,
                'rmse_ci_lower': rmse_ci[0],
                'rmse_ci_upper': rmse_ci[1],
                'r2': r2,
                'r2_ci_lower': r2_ci[0],
                'r2_ci_upper': r2_ci[1],
                'training_time': model.__dict__.get('training_time', np.nan),
                'prediction_time': prediction_time
            }
            
            summary.append(metrics)
            
            # Store plot data
            for i in range(len(y_test)):
                plot_data.append({
                    'fold': fold,
                    'model': name,
                    'actual_age': y_test[i],
                    'predicted_age': y_pred[i],
                    'residual': y_test[i] - y_pred[i]
                })
            
            # Feature importance (only for first two folds to save time)
            if fold < 2:
                feature_importance_analysis(model, X_test, y_test, X_test.columns, results_dir, name, fold, prefix)
            
            # Error analysis with signals
            error_df = analyze_prediction_errors(y_test, y_pred, ppg_signals, gauss_signals, test_indices, name, fold, results_dir, prefix)
            all_error_dfs.append(error_df)
            
            # Create comprehensive signal analysis (only for best model or first fold to save time)
            if name == 'XGBoost' or fold == 0:
                create_comprehensive_signal_analysis(error_df, ppg_signals, gauss_signals, name, fold, results_dir, prefix)
                create_detailed_case_studies(error_df, ppg_signals, gauss_signals, name, fold, results_dir, prefix)
            
            # Predicted vs Actual plot
            axes[0].scatter(y_test, y_pred, alpha=0.7, 
                          color=MODEL_STYLES[name]['color'],
                          marker=MODEL_STYLES[name]['marker'],
                          label=f'{name} (R² = {r2:.3f})',
                          s=60)
            
            # Residual plot
            residuals = y_test - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.7,
                          color=MODEL_STYLES[name]['color'],
                          marker=MODEL_STYLES[name]['marker'],
                          label=name,
                          s=60)
            
            print(f"Prediction time for {name} (Fold {fold}): {prediction_time:.2f}s")
            
        except Exception as e:
            print(f"Error evaluating model {name}: {e}")
            # Add NaN entry for failed models
            summary.append({
                'fold': fold,
                'model': name,
                'mae': np.nan,
                'mae_ci_lower': np.nan,
                'mae_ci_upper': np.nan,
                'rmse': np.nan,
                'rmse_ci_lower': np.nan,
                'rmse_ci_upper': np.nan,
                'r2': np.nan,
                'r2_ci_lower': np.nan,
                'r2_ci_upper': np.nan,
                'training_time': np.nan,
                'prediction_time': np.nan
            })
    
    # Finalize Predicted vs Actual plot with fixed axis limits
    axes[0].plot([15, 75], [15, 75], 'k--', alpha=0.8, linewidth=2)
    axes[0].set_xlabel('Actual Age (years)', fontsize=14)
    axes[0].set_ylabel('Predicted Age (years)', fontsize=14)
    axes[0].set_title('Predicted vs Actual Age', fontsize=16, fontweight='bold')
    axes[0].set_xlim(15, 75)
    axes[0].set_ylim(15, 75)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Finalize Residual plot with fixed x-axis limits
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.8, linewidth=2)
    axes[1].set_xlabel('Predicted Age (years)', fontsize=14)
    axes[1].set_ylabel('Residuals (years)', fontsize=14)
    axes[1].set_title('Residual Plot', fontsize=16, fontweight='bold')
    axes[1].set_xlim(15, 75)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Save the combined plot
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{prefix}regression_plots_fold{fold}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save plot data to CSV
    plot_df = pd.DataFrame(plot_data)
    plot_df.to_csv(f"{results_dir}/{prefix}regression_plots_data_fold{fold}.csv", index=False)
    
    # Save all error analyses
    if all_error_dfs:
        combined_errors = pd.concat(all_error_dfs, ignore_index=True)
        combined_errors.to_csv(f"{results_dir}/{prefix}combined_error_analysis_fold{fold}.csv", index=False)
    
    return summary

# Box plots for summary metrics
def plot_summary_boxplots(summary_df, results_dir, prefix):
    try:
        if summary_df.empty:
            print("Warning: summary_df is empty, skipping summary boxplots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        metrics = ['mae', 'rmse', 'r2', 'training_time']
        titles = ['MAE Across Folds', 'RMSE Across Folds', 'R² Across Folds', 'Training Time Across Folds']
        ylabels = ['MAE (years)', 'RMSE (years)', 'R²', 'Training Time (seconds)']
        
        for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            row, col = i // 2, i % 2
            
            box_data = []
            labels = []
            
            for model in summary_df['model'].unique():
                subset = summary_df[summary_df['model'] == model]
                method_data = subset[metric].dropna()
                if len(method_data) > 0:
                    box_data.append(method_data)
                    labels.append(model)
            
            if box_data:
                box_plots = axes[row, col].boxplot(box_data, labels=labels, patch_artist=True, widths=0.6)
                
                # Color the boxes
                for patch, model in zip(box_plots['boxes'], labels):
                    patch.set_facecolor(MODEL_STYLES[model]['color'])
                    patch.set_hatch(MODEL_STYLES[model]['hatch'])
                    patch.set_edgecolor('black')
                    patch.set_alpha(0.7)
            
            axes[row, col].set_title(title, fontsize=16, fontweight='bold')
            axes[row, col].set_ylabel(ylabel, fontsize=14)
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{prefix}summary_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in summary boxplots: {e}")

# Aggregate metrics across folds
def aggregate_metrics(summary_df, results_dir, prefix):
    try:
        if summary_df.empty:
            print("Warning: summary_df is empty, skipping metric aggregation")
            return
        
        metrics = ['mae', 'rmse', 'r2', 'training_time', 'prediction_time']
        agg_results = []
        
        for model in summary_df['model'].unique():
            subset = summary_df[summary_df['model'] == model]
            
            agg_metrics = {
                'model': model,
                'n_folds': len(subset)
            }
            
            for metric in metrics:
                mean_val = subset[metric].mean()
                std_val = subset[metric].std()
                agg_metrics[f'{metric}_mean'] = mean_val
                agg_metrics[f'{metric}_std'] = std_val
                agg_metrics[f'{metric}_cv'] = std_val / mean_val if mean_val != 0 else np.nan
            
            agg_results.append(agg_metrics)
        
        agg_df = pd.DataFrame(agg_results)
        agg_df.to_csv(f"{results_dir}/{prefix}aggregate_metrics.csv", index=False)
        
        # Generate LaTeX table for aggregated metrics
        with open(f"{results_dir}/{prefix}aggregate_metrics.tex", 'w') as f:
            f.write(agg_df.to_latex(float_format="%.3f", index=False))
            
        # Generate comprehensive performance report
        with open(f"{results_dir}/{prefix}performance_report.txt", 'w') as f:
            f.write("COMPREHENSIVE PERFORMANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write("Dataset: RWS PPG Regression Dataset\n")
            f.write(f"Total models evaluated: {len(agg_df)}\n")
            f.write(f"Total folds: {len(summary_df)}\n\n")
            
            # Best model by MAE
            best_mae = agg_df.loc[agg_df['mae_mean'].idxmin()]
            f.write(f"Best Model by MAE: {best_mae['model']} (MAE = {best_mae['mae_mean']:.3f} ± {best_mae['mae_std']:.3f} years)\n")
            
            # Best model by R²
            best_r2 = agg_df.loc[agg_df['r2_mean'].idxmax()]
            f.write(f"Best Model by R²: {best_r2['model']} (R² = {best_r2['r2_mean']:.3f} ± {best_r2['r2_std']:.3f})\n\n")
            
            f.write("Detailed Metrics:\n")
            f.write("-" * 30 + "\n")
            for _, row in agg_df.iterrows():
                f.write(f"\n{row['model']}:\n")
                f.write(f"  MAE: {row['mae_mean']:.3f} ± {row['mae_std']:.3f} years\n")
                f.write(f"  RMSE: {row['rmse_mean']:.3f} ± {row['rmse_std']:.3f} years\n")
                f.write(f"  R²: {row['r2_mean']:.3f} ± {row['r2_std']:.3f}\n")
                f.write(f"  Training Time: {row['training_time_mean']:.2f} ± {row['training_time_std']:.2f} s\n")
                f.write(f"  Prediction Time: {row['prediction_time_mean']:.4f} ± {row['prediction_time_std']:.4f} s\n")
            
    except Exception as e:
        print(f"Error in aggregating metrics: {e}")

# Main pipeline with resume capability
def run_regression_benchmark(data_path, results_dir="results", prefix="regression_ensemble_"):
    try:
        os.makedirs(results_dir, exist_ok=True)
        
        # Check if results already exist
        summary_file = f"{results_dir}/{prefix}summary_all.csv"
        if os.path.exists(summary_file):
            print(f"Found existing results at {summary_file}. Loading...")
            existing_summary = pd.read_csv(summary_file)
            completed_folds = existing_summary['fold'].unique()
            print(f"Found completed folds: {completed_folds}")
        else:
            existing_summary = pd.DataFrame()
            completed_folds = []
        
        X, y, groups, original_df = load_data(data_path)
        
        # Parse PPG and Gaussian signals
        print("Parsing PPG and Gaussian signals...")
        ppg_signals = []
        gauss_signals = []
        
        for idx, row in original_df.iterrows():
            ppg_signal = parse_signal(row['ppg_hb'])
            gauss_signal = parse_signal(row['gauss3_fit'])
            ppg_signals.append(ppg_signal)
            gauss_signals.append(gauss_signal)
        
        print(f"Parsed {len(ppg_signals)} PPG signals and {len(gauss_signals)} Gaussian signals")
        
        # EDA (only if not done before)
        if not os.path.exists(f"{results_dir}/age_distribution.png"):
            print("Performing exploratory data analysis...")
            exploratory_data_analysis(original_df, results_dir)
        
        gkf = GroupKFold(n_splits=5)  # 5 folds as requested
        all_records = []
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            if fold in completed_folds:
                print(f"Skipping completed fold {fold}")
                continue
                
            print(f"\n{'='*50}")
            print(f"Starting Fold {fold+1}/5")
            print(f"{'='*50}")
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx].values
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx].values
            test_indices = test_idx  # Store original indices for signal lookup
            
            models = {}
            
            for name, model_info in MODEL_DEFS.items():
                try:
                    print(f"Training {name}...")
                    start_train = time.time()
                    
                    grid = GridSearchCV(
                        model_info['model'],
                        model_info['param_grid'],
                        cv=3,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1,
                        error_score=np.nan
                    )
                    
                    grid.fit(X_train, y_train)
                    training_time = time.time() - start_train
                    
                    # Store training time in the best estimator
                    grid.best_estimator_.training_time = training_time
                    models[name] = grid.best_estimator_
                    
                    print(f"✓ {name} trained in {training_time:.2f}s")
                    print(f"  Best parameters: {grid.best_params_}")
                    
                except Exception as e:
                    print(f"✗ Error training {name} in fold {fold}: {e}")
                    continue
            
            if models:
                print(f"Evaluating {len(models)} models...")
                fold_summary = evaluate_and_export(models, X_test, y_test, test_indices, 
                                                 ppg_signals, gauss_signals, fold, results_dir, prefix)
                all_records.extend(fold_summary)
                
                # Save incremental results
                current_summary = pd.DataFrame(all_records)
                if not existing_summary.empty:
                    current_summary = pd.concat([existing_summary, current_summary], ignore_index=True)
                current_summary.to_csv(summary_file, index=False)
                print(f"✓ Saved results for fold {fold}")
        
        # Final aggregation and visualization
        if all_records or not existing_summary.empty:
            if all_records:
                summary_df = pd.DataFrame(all_records)
                if not existing_summary.empty:
                    summary_df = pd.concat([existing_summary, summary_df], ignore_index=True)
            else:
                summary_df = existing_summary
                
            summary_df.to_csv(summary_file, index=False)
            
            print("\nPerforming final aggregation and visualization...")
            aggregate_metrics(summary_df, results_dir, prefix)
            plot_summary_boxplots(summary_df, results_dir, prefix)
        
        # Save environment info
        with open(f"{results_dir}/environment.txt", 'w') as f:
            f.write(f"Python: {sys.version}\n")
            f.write(f"scikit-learn: {sys.modules['sklearn'].__version__}\n")
            f.write(f"xgboost: {sys.modules['xgboost'].__version__}\n")
            f.write(f"pandas: {pd.__version__}\n")
            f.write(f"numpy: {np.__version__}\n")
            
        print(f"\n{'='*50}")
        print("BENCHMARK COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {results_dir}")
        print(f"{'='*50}")
            
    except Exception as e:
        print(f"Error in benchmark: {e}")
        import traceback
        traceback.print_exc()

# Statistical comparison using Wilcoxon test
def statistical_comparison(summary_csv, output_txt):
    try:
        df = pd.read_csv(summary_csv)
        
        if df.empty:
            print(f"Warning: {summary_csv} is empty, skipping statistical comparison")
            return
        
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("STATISTICAL COMPARISON OF REGRESSION MODELS\n")
            f.write("=" * 50 + "\n\n")
            f.write("Dataset: RWS PPG Regression Dataset\n")
            f.write("Metric: MAE (Mean Absolute Error)\n")
            f.write("Test: Wilcoxon Signed-Rank Test\n\n")
            
            models = df['model'].unique()
            
            f.write("Pairwise Model Comparisons:\n")
            f.write("-" * 40 + "\n")
            
            for i, a in enumerate(models):
                for b in models[i+1:]:
                    mae_a = df[df['model'] == a]['mae'].values
                    mae_b = df[df['model'] == b]['mae'].values
                    
                    if len(mae_a) == len(mae_b) and len(mae_a) > 1:
                        w_stat, w_pval = wilcoxon(mae_a, mae_b)
                        f.write(f"\n{a} vs {b}:\n")
                        f.write(f"  Wilcoxon W = {w_stat:.3f}, p = {w_pval:.5f}\n")
                        if w_pval < 0.001:
                            f.write("  *** p < 0.001 (Highly Significant)\n")
                        elif w_pval < 0.01:
                            f.write("  ** p < 0.01 (Very Significant)\n")
                        elif w_pval < 0.05:
                            f.write("  * p < 0.05 (Significant)\n")
                        else:
                            f.write("  p ≥ 0.05 (Not Significant)\n")
                        
    except Exception as e:
        print(f"Error in statistical comparison: {e}")

if __name__ == "__main__":
    DATA_PATH = "data/train_rws_ppg_regression_dataset.csv"
    RESULTS_DIR = "results"
    PREFIX = "regression_ensemble_"
    
    run_regression_benchmark(DATA_PATH, RESULTS_DIR, PREFIX)
    statistical_comparison(f"{RESULTS_DIR}/{PREFIX}summary_all.csv",
                         f"{RESULTS_DIR}/{PREFIX}statistical_significance.txt")