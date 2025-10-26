import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import ttest_rel, wilcoxon, shapiro, bootstrap
import warnings
warnings.filterwarnings('ignore')

class FeatureSetStatisticalComparison:
    """
    Rigorous statistical comparison of feature sets for publication
    """
    
    def __init__(self, alpha=0.05, n_repeats=100, n_splits=5):
        self.alpha = alpha
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        
    def compare_feature_sets(self, X_2g, X_3g, y, groups):
        """
        Comprehensive comparison of 2G vs 3G features using repeated nested CV
        """
        maes_2g, maes_3g = [], []
        r2s_2g, r2s_3g = [], []
        
        for repeat in range(self.n_repeats):
            # Reset seed for each repeat
            np.random.seed(42 + repeat)
            
            # Group K-Fold to prevent data leakage
            gkf = GroupKFold(n_splits=self.n_splits)
            
            for train_idx, test_idx in gkf.split(X_2g, y, groups):
                # Split data
                X_2g_train, X_2g_test = X_2g.iloc[train_idx], X_2g.iloc[test_idx]
                X_3g_train, X_3g_test = X_3g.iloc[train_idx], X_3g.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train models with identical hyperparameters
                from xgboost import XGBRegressor
                model_2g = XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ).fit(X_2g_train, y_train)
                
                model_3g = XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ).fit(X_3g_train, y_train)
                
                # Predictions and metrics
                y_pred_2g = model_2g.predict(X_2g_test)
                y_pred_3g = model_3g.predict(X_3g_test)
                
                maes_2g.append(mean_absolute_error(y_test, y_pred_2g))
                maes_3g.append(mean_absolute_error(y_test, y_pred_3g))
                r2s_2g.append(r2_score(y_test, y_pred_2g))
                r2s_3g.append(r2_score(y_test, y_pred_3g))
        
        return {
            'mae_2g': np.array(maes_2g), 'mae_3g': np.array(maes_3g),
            'r2_2g': np.array(r2s_2g), 'r2_3g': np.array(r2s_3g)
        }
    
    def cohens_d(self, x, y):
        """Calculate Cohen's d effect size"""
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
    
    def statistical_analysis(self, results):
        """
        Comprehensive statistical analysis for publication
        """
        analysis = {}
        
        # 1. Descriptive statistics
        analysis['descriptive'] = {
            'mae_2g': {
                'mean': np.mean(results['mae_2g']),
                'std': np.std(results['mae_2g']),
                'median': np.median(results['mae_2g'])
            },
            'mae_3g': {
                'mean': np.mean(results['mae_3g']),
                'std': np.std(results['mae_3g']),
                'median': np.median(results['mae_3g'])
            },
            'r2_2g': {
                'mean': np.mean(results['r2_2g']),
                'std': np.std(results['r2_2g']),
                'median': np.median(results['r2_2g'])
            },
            'r2_3g': {
                'mean': np.mean(results['r2_3g']),
                'std': np.std(results['r2_3g']),
                'median': np.median(results['r2_3g'])
            }
        }
        
        # 2. Normality testing
        analysis['normality'] = {
            'mae_2g_shapiro': shapiro(results['mae_2g']),
            'mae_3g_shapiro': shapiro(results['mae_3g']),
            'r2_2g_shapiro': shapiro(results['r2_2g']),
            'r2_3g_shapiro': shapiro(results['r2_3g'])
        }
        
        # 3. Statistical hypothesis testing
        # For MAE (3G better if lower MAE)
        mae_2g_norm = analysis['normality']['mae_2g_shapiro'][1] > self.alpha
        mae_3g_norm = analysis['normality']['mae_3g_shapiro'][1] > self.alpha
        
        if mae_2g_norm and mae_3g_norm:
            # Parametric test
            mae_stat, mae_p = ttest_rel(results['mae_2g'], results['mae_3g'])
            mae_test = 'paired_ttest'
        else:
            # Non-parametric test
            mae_stat, mae_p = wilcoxon(results['mae_2g'], results['mae_3g'])
            mae_test = 'wilcoxon'
        
        # For R² (3G better if higher R²)
        r2_2g_norm = analysis['normality']['r2_2g_shapiro'][1] > self.alpha
        r2_3g_norm = analysis['normality']['r2_3g_shapiro'][1] > self.alpha
        
        if r2_2g_norm and r2_3g_norm:
            r2_stat, r2_p = ttest_rel(results['r2_2g'], results['r2_3g'])
            r2_test = 'paired_ttest'
        else:
            r2_stat, r2_p = wilcoxon(results['r2_2g'], results['r2_3g'])
            r2_test = 'wilcoxon'
        
        analysis['hypothesis_tests'] = {
            'mae': {
                'test': mae_test,
                'statistic': mae_stat,
                'p_value': mae_p,
                'significant': mae_p < self.alpha,
                'interpretation': '3G_features better' if mae_p < self.alpha and np.mean(results['mae_3g']) < np.mean(results['mae_2g']) else 'No significant difference'
            },
            'r2': {
                'test': r2_test,
                'statistic': r2_stat,
                'p_value': r2_p,
                'significant': r2_p < self.alpha,
                'interpretation': '3G_features better' if r2_p < self.alpha and np.mean(results['r2_3g']) > np.mean(results['r2_2g']) else 'No significant difference'
            }
        }
        
        # 4. Effect sizes
        analysis['effect_sizes'] = {
            'mae_cohen_d': self.cohens_d(results['mae_2g'], results['mae_3g']),
            'r2_cohen_d': self.cohens_d(results['r2_3g'], results['r2_2g'])
        }
        
        # 5. Confidence intervals for differences
        mae_diff = results['mae_2g'] - results['mae_3g']  # Positive if 3G better
        r2_diff = results['r2_3g'] - results['r2_2g']     # Positive if 3G better
        
        mae_ci = bootstrap((mae_diff,), np.mean, confidence_level=0.95, random_state=42)
        r2_ci = bootstrap((r2_diff,), np.mean, confidence_level=0.95, random_state=42)
        
        analysis['confidence_intervals'] = {
            'mae_diff_95_ci': mae_ci.confidence_interval,
            'r2_diff_95_ci': r2_ci.confidence_interval
        }
        
        # 6. Improvement percentages
        mae_improvement = ((np.mean(results['mae_2g']) - np.mean(results['mae_3g'])) / 
                          np.mean(results['mae_2g'])) * 100
        r2_improvement = ((np.mean(results['r2_3g']) - np.mean(results['r2_2g'])) / 
                         np.abs(np.mean(results['r2_2g']))) * 100
        
        analysis['improvement_analysis'] = {
            'mae_improvement_percent': mae_improvement,
            'r2_improvement_percent': r2_improvement,
            'mae_absolute_improvement': np.mean(results['mae_2g']) - np.mean(results['mae_3g']),
            'r2_absolute_improvement': np.mean(results['r2_3g']) - np.mean(results['r2_2g'])
        }
        
        return analysis
    
    def generate_publication_report(self, analysis):
        """
        Generatestatistical report
        """
        report = []
        report.append("="*80)
        report.append("STATISTICAL COMPARISON: 2G_features vs 3G_features")
        report.append("="*80)
        
        # Descriptive statistics
        report.append("\n1. DESCRIPTIVE STATISTICS")
        report.append("-" * 40)
        report.append(f"MAE - 2G_features: {analysis['descriptive']['mae_2g']['mean']:.4f} ± {analysis['descriptive']['mae_2g']['std']:.4f}")
        report.append(f"MAE - 3G_features: {analysis['descriptive']['mae_3g']['mean']:.4f} ± {analysis['descriptive']['mae_3g']['std']:.4f}")
        report.append(f"R²  - 2G_features: {analysis['descriptive']['r2_2g']['mean']:.4f} ± {analysis['descriptive']['r2_2g']['std']:.4f}")
        report.append(f"R²  - 3G_features: {analysis['descriptive']['r2_3g']['mean']:.4f} ± {analysis['descriptive']['r2_3g']['std']:.4f}")
        
        # Hypothesis testing results
        report.append("\n2. STATISTICAL HYPOTHESIS TESTING")
        report.append("-" * 40)
        mae_test = analysis['hypothesis_tests']['mae']
        r2_test = analysis['hypothesis_tests']['r2']
        
        report.append(f"MAE - Test: {mae_test['test']}, p-value: {mae_test['p_value']:.6f}")
        report.append(f"MAE - Significance: {'***' if mae_test['p_value'] < 0.001 else '**' if mae_test['p_value'] < 0.01 else '*' if mae_test['p_value'] < 0.05 else 'ns'}")
        report.append(f"MAE - Interpretation: {mae_test['interpretation']}")
        
        report.append(f"R²  - Test: {r2_test['test']}, p-value: {r2_test['p_value']:.6f}")
        report.append(f"R²  - Significance: {'***' if r2_test['p_value'] < 0.001 else '**' if r2_test['p_value'] < 0.01 else '*' if r2_test['p_value'] < 0.05 else 'ns'}")
        report.append(f"R²  - Interpretation: {r2_test['interpretation']}")
        
        # Effect sizes
        report.append("\n3. EFFECT SIZES")
        report.append("-" * 40)
        report.append(f"MAE - Cohen's d: {analysis['effect_sizes']['mae_cohen_d']:.3f}")
        report.append(f"R²  - Cohen's d: {analysis['effect_sizes']['r2_cohen_d']:.3f}")
        report.append("Interpretation: |d| < 0.2 (negligible), 0.2-0.5 (small), 0.5-0.8 (medium), >0.8 (large)")
        
        # Confidence intervals
        report.append("\n4. CONFIDENCE INTERVALS")
        report.append("-" * 40)
        report.append(f"MAE difference 95% CI: [{analysis['confidence_intervals']['mae_diff_95_ci'].low:.4f}, {analysis['confidence_intervals']['mae_diff_95_ci'].high:.4f}]")
        report.append(f"R² difference 95% CI: [{analysis['confidence_intervals']['r2_diff_95_ci'].low:.4f}, {analysis['confidence_intervals']['r2_diff_95_ci'].high:.4f}]")
        
        # Improvement analysis
        report.append("\n5. IMPROVEMENT ANALYSIS")
        report.append("-" * 40)
        report.append(f"MAE improvement: {analysis['improvement_analysis']['mae_improvement_percent']:.2f}%")
        report.append(f"R² improvement: {analysis['improvement_analysis']['r2_improvement_percent']:.2f}%")
        report.append(f"Absolute MAE improvement: {analysis['improvement_analysis']['mae_absolute_improvement']:.4f} years")
        report.append(f"Absolute R² improvement: {analysis['improvement_analysis']['r2_absolute_improvement']:.4f}")
        
        # Overall conclusion
        report.append("\n6. OVERALL CONCLUSION")
        report.append("-" * 40)
        mae_sig = mae_test['significant']
        r2_sig = r2_test['significant']
        
        if mae_sig and r2_sig:
            report.append("*** STRONG EVIDENCE: 3G_features statistically significantly outperform 2G_features ***")
        elif mae_sig or r2_sig:
            report.append("** MODERATE EVIDENCE: 3G_features show significant improvement in some metrics **")
        else:
            report.append("* NO SIGNIFICANT DIFFERENCE: 3G_features do not show statistically significant improvement *")
        
        report.append("="*80)
        
        return "\n".join(report)

# Usage example:
def run_statistical_comparison(df, target='age', group_col='app_id'):
    """
    Run complete statistical comparison between 2G and 3G features
    """
    # Define feature sets
    features_2g = [
        'amp1_2g', 'mean1_2g', 'sigma1_2g',
        'amp2_2g', 'mean2_2g', 'sigma2_2g',
        'peak_distance_2g', 'ratio_2g'
    ]
    
    features_3g = [
        'amp1_3g', 'mean1_3g', 'sigma1_3g',
        'amp2_3g', 'mean2_3g', 'sigma2_3g',
        'amp3_3g', 'mean3_3g', 'sigma3_3g',
        'ratio_3g_12', 'ratio_3g_13', 'ratio_3g_23',
        'total_area_3g',
        'peak_distance_12', 'peak_distance_13', 'peak_distance_23'
    ]
    
    # Prepare data
    X_2g = df[features_2g].copy()
    X_3g = df[features_3g].copy()
    y = df[target]
    groups = df[group_col]
    
    # Remove rows with missing values
    mask_2g = ~X_2g.isna().any(axis=1)
    mask_3g = ~X_3g.isna().any(axis=1)
    mask = mask_2g & mask_3g & ~y.isna()
    
    X_2g_clean = X_2g[mask]
    X_3g_clean = X_3g[mask]
    y_clean = y[mask]
    groups_clean = groups[mask]
    
    print(f"Clean dataset: {len(X_2g_clean)} samples")
    print(f"2G features: {len(features_2g)}")
    print(f"3G features: {len(features_3g)}")
    
    # Initialize comparator
    comparator = FeatureSetStatisticalComparison(alpha=0.05, n_repeats=100, n_splits=5)
    
    # Run comparison
    print("Running statistical comparison...")
    results = comparator.compare_feature_sets(X_2g_clean, X_3g_clean, y_clean, groups_clean)
    
    # Statistical analysis
    analysis = comparator.statistical_analysis(results)
    
    # Generate report
    report = comparator.generate_publication_report(analysis)
    
    return report, analysis, results

# Run the analysis
if __name__ == "__main__":
    # Load your data
    data_path = "data/train_rws_ppg_regression_dataset.csv"
    df = pd.read_csv(data_path)
    
    # Run comparison
    report, analysis, results = run_statistical_comparison(df)
    print(report)