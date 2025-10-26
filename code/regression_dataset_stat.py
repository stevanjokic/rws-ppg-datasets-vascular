import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import shapiro, normaltest
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output folder
output_folder = "results/"
os.makedirs(output_folder, exist_ok=True)

# File prefix
file_prefix = "regression_dataset_"

def parse_signal(signal_string):
    """Parse signal string to numpy array"""
    try:
        if isinstance(signal_string, str):
            signal_string = signal_string.strip('"')
            signal_array = np.fromstring(signal_string, sep=',')
            return signal_array
        else:
            return np.array([])
    except Exception as e:
        print(f"Error parsing signal: {e}")
        return np.array([])

def preprocess_template_signal(template_signal, od_index, do_index, signal_length=150):
    """
    Preprocess template signal by setting values outside od-do range to zero
    """
    processed_signal = np.zeros(signal_length)
    
    if len(template_signal) > 0:
        # Ensure od and do are within bounds
        od = max(0, min(int(od_index), signal_length - 1))
        do = max(0, min(int(do_index), signal_length - 1))
        
        # Make sure od <= do
        if od > do:
            od, do = do, od
            
        # Calculate how much of the template signal fits within the OD-DO range
        available_length = min(len(template_signal), signal_length)
        
        # Place the template signal starting from index 0, but only keep the part within OD-DO range
        if available_length > 0:
            # Copy the entire template signal to the beginning
            end_idx = min(available_length, signal_length)
            processed_signal[:end_idx] = template_signal[:end_idx]
            
            # Set values outside OD-DO range to zero
            processed_signal[:od] = 0  # Before OD index
            processed_signal[do+1:] = 0  # After DO index
    
    return processed_signal

def create_age_decade(age):
    """Create age decade label"""
    if 25 <= age < 35:
        return '25-34'
    elif 35 <= age < 45:
        return '35-44'
    elif 45 <= age < 55:
        return '45-54'
    elif 55 <= age < 65:
        return '55-64'
    elif 65 <= age < 75:
        return '65-74'
    elif age >= 75:
        return '75+'
    else:
        return None

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    print("Loading dataset...")
    df = pd.read_csv('data/train_rws_ppg_regression_dataset.csv')
    
    # Filter invalid data
    df = df[df['age'] >= 15]
    df = df[(df['hr'] >= 45) & (df['hr'] <= 105)]
    
    # Clean sex data - remove NaN values and ensure only 0 and 1
    df = df[df['sex'].notna()]
    df = df[df['sex'].isin([0, 1])]
    
    # Create age groups for analysis
    df['age_group'] = pd.cut(df['age'], 
                            bins=[15, 30, 40, 50, 60, 75, 100],
                            labels=['15-29', '30-39', '40-49', '50-59', '60-74', '75+'])
    
    # Create age decades for PPG signal analysis
    df['age_decade'] = df['age'].apply(create_age_decade)
    
    # CORRECTED: 0 = Male, 1 = Female
    df['sex_label'] = df['sex'].map({0: 'Male', 1: 'Female'})
    
    # Process template signals for PPG analysis
    print("Processing template signals for PPG analysis...")
    template_signals = []
    for idx, row in df.iterrows():
        template_ppg = parse_signal(row['template_ppg'])
        od_idx = row['od']
        do_idx = row['do']
        template_processed = preprocess_template_signal(template_ppg, od_idx, do_idx)
        template_signals.append(template_processed)
    
    df['processed_template'] = template_signals
    df['signal_length'] = df['processed_template'].apply(len)
    
    print(f"Data cleaning summary:")
    print(f"Original rows: {len(pd.read_csv('data/train_rws_ppg_regression_dataset.csv'))}")
    print(f"After cleaning: {len(df)}")
    print(f"Sex distribution: {df['sex_label'].value_counts().to_dict()}")
    
    return df

def calculate_average_signals(df):
    """Calculate average signals for each age decade and sex"""
    decades = ['25-34', '35-44', '45-54', '55-64', '65-74', '75+']
    results = {}
    
    for decade in decades:
        decade_data = df[df['age_decade'] == decade]
        
        if len(decade_data) == 0:
            continue
            
        decade_results = {}
        
        # Overall average for decade
        overall_signals = np.array([sig for sig in decade_data['processed_template'] if len(sig) == 150])
        if len(overall_signals) > 0:
            decade_results['overall'] = {
                'mean': np.mean(overall_signals, axis=0),
                'std': np.std(overall_signals, axis=0),
                'count': len(overall_signals)
            }
        
        # Male signals (sex = 0)
        male_data = decade_data[decade_data['sex'] == 0]
        male_signals = np.array([sig for sig in male_data['processed_template'] if len(sig) == 150])
        if len(male_signals) > 0:
            decade_results['male'] = {
                'mean': np.mean(male_signals, axis=0),
                'std': np.std(male_signals, axis=0),
                'count': len(male_signals)
            }
        
        # Female signals (sex = 1)
        female_data = decade_data[decade_data['sex'] == 1]
        female_signals = np.array([sig for sig in female_data['processed_template'] if len(sig) == 150])
        if len(female_signals) > 0:
            decade_results['female'] = {
                'mean': np.mean(female_signals, axis=0),
                'std': np.std(female_signals, axis=0),
                'count': len(female_signals)
            }
        
        results[decade] = decade_results
    
    return results

def plot_ppg_signals_by_decades(results):
    """Plot PPG signals by decades for males, females and overall"""
    print("Creating PPG signals by decades plots...")
    
    decades = list(results.keys())
    
    # 1. Overall comparison grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    color = '#2E86AB'
    
    for i, decade in enumerate(decades):
        if i >= len(axes):
            break
            
        ax = axes[i]
        decade_data = results[decade]
        
        if 'overall' in decade_data:
            mean_signal = decade_data['overall']['mean']
            std_signal = decade_data['overall']['std']
            count = decade_data['overall']['count']
            
            x = np.arange(len(mean_signal))
            ax.plot(x, mean_signal, color=color, linewidth=2.5, label='Mean')
            ax.fill_between(x, 
                          mean_signal - std_signal, 
                          mean_signal + std_signal, 
                          color=color, alpha=0.3, label='±1 STD')
            
            ax.set_title(f'{decade} Years\n(n={count})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(-0.5, 1.2)
    
    # Hide empty subplots
    for i in range(len(decades), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Average PPG Signals Across Age Decades (Overall)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_folder + file_prefix + 'ppg_decades_overall_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed plots by decade with males and females
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    colors = {'overall': '#2E86AB', 'male': '#A23B72', 'female': '#F18F01'}
    labels = {'overall': 'Overall', 'male': 'Male', 'female': 'Female'}
    
    for i, decade in enumerate(decades):
        if i >= len(axes):
            break
            
        ax = axes[i]
        decade_data = results[decade]
        
        for gender in ['overall', 'male', 'female']:
            if gender in decade_data:
                mean_signal = decade_data[gender]['mean']
                std_signal = decade_data[gender]['std']
                count = decade_data[gender]['count']
                
                # Plot mean signal
                x = np.arange(len(mean_signal))
                ax.plot(x, mean_signal, color=colors[gender], linewidth=2, 
                       label=f'{labels[gender]} (n={count})')
                
                # Plot dispersion as shaded area
                alpha = 0.2 if gender == 'overall' else 0.3
                ax.fill_between(x, 
                              mean_signal - std_signal, 
                              mean_signal + std_signal, 
                              color=colors[gender], alpha=alpha)
        
        ax.set_title(f'Age Decade: {decade}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-0.5, 1.2)
    
    # Hide empty subplots
    for i in range(len(decades), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_folder + file_prefix + 'ppg_decades_sex_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Sex comparison for each decade
    for decade in decades:
        decade_data = results[decade]
        
        if 'male' in decade_data and 'female' in decade_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Overlaid signals
            colors_sex = {'male': '#A23B72', 'female': '#F18F01'}
            
            for gender in ['male', 'female']:
                mean_signal = decade_data[gender]['mean']
                std_signal = decade_data[gender]['std']
                count = decade_data[gender]['count']
                
                x = np.arange(len(mean_signal))
                ax1.plot(x, mean_signal, color=colors_sex[gender], linewidth=2.5,
                        label=f'{gender.capitalize()} (n={count})')
                ax1.fill_between(x, 
                               mean_signal - std_signal, 
                               mean_signal + std_signal, 
                               color=colors_sex[gender], alpha=0.2)
            
            ax1.set_title(f'Sex Comparison - {decade} Years\nOverlaid Signals', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Sample Index')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_ylim(-0.5, 1.2)
            
            # Plot 2: Side by side comparison
            x = np.arange(len(decade_data['male']['mean']))
            ax2.plot(x, decade_data['male']['mean'], color='#A23B72', linewidth=2.5, 
                    label=f'Male (n={decade_data["male"]["count"]})', alpha=0.8)
            ax2.plot(x, decade_data['female']['mean'], color='#F18F01', linewidth=2.5,
                    label=f'Female (n={decade_data["female"]["count"]})', alpha=0.8)
            
            ax2.set_title(f'Sex Comparison - {decade} Years\nMean Signals', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_ylim(-0.5, 1.2)
            
            plt.tight_layout()
            plt.savefig(output_folder + file_prefix + f'sex_comparison_{decade}.png', dpi=300, bbox_inches='tight')
            plt.close()

def plot_heart_rate_by_decades(df):
    """Plot heart rate analysis by decades for males, females and overall"""
    print("Creating heart rate by decades plots...")
    
    # Filter data for decades analysis (ages 25+)
    df_decades = df[df['age_decade'].notnull()].copy()
    
    decades = ['25-34', '35-44', '45-54', '55-64', '65-74', '75+']
    
    # 1. Heart rate distribution by decades and sex
    plt.figure(figsize=(14, 8))
    
    # Overall heart rate by decades
    plt.subplot(2, 2, 1)
    hr_data_overall = []
    decade_labels = []
    for decade in decades:
        decade_data = df_decades[df_decades['age_decade'] == decade]
        if len(decade_data) > 0:
            hr_data_overall.append(decade_data['hr'].values)
            decade_labels.append(decade)
    
    plt.boxplot(hr_data_overall, labels=decade_labels)
    plt.title('A) Heart Rate Distribution by Age Decade (Overall)', fontweight='bold')
    plt.xlabel('Age Decade')
    plt.ylabel('Heart Rate (bpm)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Heart rate by decades - Males
    plt.subplot(2, 2, 2)
    hr_data_male = []
    for decade in decades:
        decade_data = df_decades[(df_decades['age_decade'] == decade) & (df_decades['sex'] == 0)]
        if len(decade_data) > 0:
            hr_data_male.append(decade_data['hr'].values)
    
    plt.boxplot(hr_data_male, labels=decade_labels)
    plt.title('B) Heart Rate Distribution - Males', fontweight='bold')
    plt.xlabel('Age Decade')
    plt.ylabel('Heart Rate (bpm)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Heart rate by decades - Females
    plt.subplot(2, 2, 3)
    hr_data_female = []
    for decade in decades:
        decade_data = df_decades[(df_decades['age_decade'] == decade) & (df_decades['sex'] == 1)]
        if len(decade_data) > 0:
            hr_data_female.append(decade_data['hr'].values)
    
    plt.boxplot(hr_data_female, labels=decade_labels)
    plt.title('C) Heart Rate Distribution - Females', fontweight='bold')
    plt.xlabel('Age Decade')
    plt.ylabel('Heart Rate (bpm)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Heart rate trends by decades
    plt.subplot(2, 2, 4)
    
    # Calculate mean HR for each decade and sex
    hr_means = {'overall': [], 'male': [], 'female': []}
    hr_stds = {'overall': [], 'male': [], 'female': []}
    
    for decade in decades:
        # Overall
        decade_data = df_decades[df_decades['age_decade'] == decade]
        if len(decade_data) > 0:
            hr_means['overall'].append(decade_data['hr'].mean())
            hr_stds['overall'].append(decade_data['hr'].std())
        
        # Male
        male_data = df_decades[(df_decades['age_decade'] == decade) & (df_decades['sex'] == 0)]
        if len(male_data) > 0:
            hr_means['male'].append(male_data['hr'].mean())
            hr_stds['male'].append(male_data['hr'].std())
        
        # Female
        female_data = df_decades[(df_decades['age_decade'] == decade) & (df_decades['sex'] == 1)]
        if len(female_data) > 0:
            hr_means['female'].append(female_data['hr'].mean())
            hr_stds['female'].append(female_data['hr'].std())
    
    x_pos = np.arange(len(decades))
    width = 0.25
    
    plt.bar(x_pos - width, hr_means['overall'], width, label='Overall', 
            color='#2E86AB', alpha=0.8, yerr=hr_stds['overall'], capsize=5)
    plt.bar(x_pos, hr_means['male'], width, label='Male', 
            color='#A23B72', alpha=0.8, yerr=hr_stds['male'], capsize=5)
    plt.bar(x_pos + width, hr_means['female'], width, label='Female', 
            color='#F18F01', alpha=0.8, yerr=hr_stds['female'], capsize=5)
    
    plt.title('D) Mean Heart Rate by Age Decade and Sex', fontweight='bold')
    plt.xlabel('Age Decade')
    plt.ylabel('Mean Heart Rate (bpm)')
    plt.xticks(x_pos, decades, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_folder + file_prefix + 'heart_rate_by_decades.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed heart rate statistics table by decades
    hr_stats = []
    for decade in decades:
        decade_data = df_decades[df_decades['age_decade'] == decade]
        
        if len(decade_data) > 0:
            # Overall
            hr_stats.append({
                'Decade': decade,
                'Group': 'Overall',
                'N': len(decade_data),
                'Mean_HR': decade_data['hr'].mean(),
                'Std_HR': decade_data['hr'].std(),
                'Min_HR': decade_data['hr'].min(),
                'Max_HR': decade_data['hr'].max()
            })
            
            # Male
            male_data = df_decades[(df_decades['age_decade'] == decade) & (df_decades['sex'] == 0)]
            if len(male_data) > 0:
                hr_stats.append({
                    'Decade': decade,
                    'Group': 'Male',
                    'N': len(male_data),
                    'Mean_HR': male_data['hr'].mean(),
                    'Std_HR': male_data['hr'].std(),
                    'Min_HR': male_data['hr'].min(),
                    'Max_HR': male_data['hr'].max()
                })
            
            # Female
            female_data = df_decades[(df_decades['age_decade'] == decade) & (df_decades['sex'] == 1)]
            if len(female_data) > 0:
                hr_stats.append({
                    'Decade': decade,
                    'Group': 'Female',
                    'N': len(female_data),
                    'Mean_HR': female_data['hr'].mean(),
                    'Std_HR': female_data['hr'].std(),
                    'Min_HR': female_data['hr'].min(),
                    'Max_HR': female_data['hr'].max()
                })
    
    hr_stats_df = pd.DataFrame(hr_stats)
    hr_stats_df.to_csv(output_folder + file_prefix + 'heart_rate_stats_by_decades.csv', index=False)
    
    return hr_stats_df

def comprehensive_statistical_summary(df):
    """Generate comprehensive statistical summary"""
    print("Generating comprehensive statistical summary...")
    
    # Select key variables for detailed analysis
    key_variables = ['age', 'sex', 'hr', 'rmssd', 'beat_corr_ratio', 'interbeat_corr_ratio']
    
    stats_list = []
    for col in key_variables:
        if col in df.columns:
            data = df[col].dropna()
            stats_list.append({
                'Variable': col,
                'N': len(data),
                'Mean': data.mean(),
                'Std': data.std(),
                'Min': data.min(),
                'Max': data.max(),
                'Median': data.median(),
                'Q1': data.quantile(0.25),
                'Q3': data.quantile(0.75),
                'IQR': data.quantile(0.75) - data.quantile(0.25),
                'Skewness': data.skew(),
                'Kurtosis': data.kurtosis(),
                'Missing': df[col].isnull().sum(),
                'Missing (%)': (df[col].isnull().sum() / len(df)) * 100
            })
    
    stats_df = pd.DataFrame(stats_list)
    
    # Save detailed statistics
    stats_df.to_csv(output_folder + file_prefix + "comprehensive_stats.csv", index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICAL SUMMARY")
    print("="*80)
    print(stats_df.round(3).to_string(index=False))
    
    return stats_df

def normality_tests(df):
    """Perform normality tests on key variables"""
    print("\nPerforming normality tests...")
    
    continuous_vars = ['age', 'hr', 'rmssd', 'beat_corr_ratio', 'interbeat_corr_ratio']
    normality_results = []
    
    for var in continuous_vars:
        if var in df.columns:
            data = df[var].dropna()
            if len(data) > 3:
                # Shapiro-Wilk test
                shapiro_stat, shapiro_p = shapiro(data)
                # D'Agostino's normality test
                dagostino_stat, dagostino_p = normaltest(data)
                
                normality_results.append({
                    'Variable': var,
                    'N': len(data),
                    'Shapiro-Wilk Statistic': shapiro_stat,
                    'Shapiro-Wilk p-value': shapiro_p,
                    'D\'Agostino Statistic': dagostino_stat,
                    'D\'Agostino p-value': dagostino_p,
                    'Normal (α=0.05)': shapiro_p > 0.05 and dagostino_p > 0.05
                })
    
    normality_df = pd.DataFrame(normality_results)
    normality_df.to_csv(output_folder + file_prefix + "normality_tests.csv", index=False)
    
    print("\nNORMALITY TESTS:")
    print("="*60)
    print(normality_df.round(4).to_string(index=False))
    
    return normality_df

def demographic_analysis(df):
    """Detailed demographic analysis"""
    print("\nPerforming demographic analysis...")
    
    # Age distribution by sex
    age_sex_stats = df.groupby('sex_label')['age'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    age_sex_stats.columns = ['N', 'Mean', 'Std', 'Min', 'Max']
    
    # Age group distribution
    age_group_dist = pd.crosstab(df['age_group'], df['sex_label'], margins=True)
    age_group_percent = pd.crosstab(df['age_group'], df['sex_label'], normalize=True) * 100
    
    # Age decade distribution
    age_decade_dist = pd.crosstab(df['age_decade'], df['sex_label'], margins=True)
    
    # Save demographic analysis
    age_sex_stats.to_csv(output_folder + file_prefix + "age_sex_stats.csv")
    age_group_dist.to_csv(output_folder + file_prefix + "age_group_distribution.csv")
    age_group_percent.round(2).to_csv(output_folder + file_prefix + "age_group_percentage.csv")
    age_decade_dist.to_csv(output_folder + file_prefix + "age_decade_distribution.csv")
    
    print("\nAGE STATISTICS BY SEX:")
    print("="*40)
    print(age_sex_stats)
    
    print("\nAGE DECADE DISTRIBUTION:")
    print("="*40)
    print(age_decade_dist)
    
    return age_sex_stats, age_group_dist

def correlation_analysis(df):
    """Comprehensive correlation analysis"""
    print("\nPerforming correlation analysis...")
    
    # Select numerical variables for correlation
    numerical_vars = ['age', 'hr', 'rmssd', 'beat_corr_ratio', 'interbeat_corr_ratio']
    corr_vars = [var for var in numerical_vars if var in df.columns]
    
    # Calculate correlation matrix
    corr_matrix = df[corr_vars].corr(method='pearson')
    corr_pvalues = pd.DataFrame(index=corr_vars, columns=corr_vars)
    
    # Calculate p-values for correlations
    for i in corr_vars:
        for j in corr_vars:
            if i != j:
                # Remove NaN values for correlation calculation
                valid_data = df[[i, j]].dropna()
                if len(valid_data) > 2:
                    corr, p_value = stats.pearsonr(valid_data[i], valid_data[j])
                    corr_pvalues.loc[i, j] = p_value
                else:
                    corr_pvalues.loc[i, j] = np.nan
            else:
                corr_pvalues.loc[i, j] = np.nan
    
    # Save correlation matrices
    corr_matrix.round(4).to_csv(output_folder + file_prefix + "correlation_matrix.csv")
    corr_pvalues.round(4).to_csv(output_folder + file_prefix + "correlation_pvalues.csv")
    
    print("\nPEARSON CORRELATION MATRIX:")
    print("="*50)
    print(corr_matrix.round(3))
    
    return corr_matrix, corr_pvalues

def create_publication_plots(df):
    """Create publication-quality plots"""
    print("\nCreating publication-quality plots...")
    
    # 1. Age distribution with density plot
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    n, bins, patches = plt.hist(df['age'], bins=20, color='skyblue', 
                               edgecolor='navy', alpha=0.7, density=True)
    plt.xlabel('Age (years)')
    plt.ylabel('Density')
    plt.title('A) Age Distribution', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add normal distribution curve
    mu, sigma = df['age'].mean(), df['age'].std()
    x = np.linspace(df['age'].min(), df['age'].max(), 100)
    y = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y, 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.1f}, σ={sigma:.1f}')
    plt.legend()
    
    # 1b. Age by sex
    plt.subplot(1, 2, 2)
    sex_colors = {'Male': 'steelblue', 'Female': 'coral'}
    
    # Filter only valid sex labels
    valid_sex_data = df[df['sex_label'].notna()]
    
    for sex in valid_sex_data['sex_label'].unique():
        subset = valid_sex_data[valid_sex_data['sex_label'] == sex]
        plt.hist(subset['age'], bins=15, alpha=0.6, 
                label=sex, color=sex_colors[sex], density=True)
    plt.xlabel('Age (years)')
    plt.ylabel('Density')
    plt.title('B) Age Distribution by Sex', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_folder + file_prefix + 'age_distribution_comprehensive.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Physiological parameters distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Heart rate
    axes[0,0].hist(df['hr'], bins=20, color='lightcoral', alpha=0.7, edgecolor='darkred')
    axes[0,0].set_xlabel('Heart Rate (bpm)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('A) Heart Rate Distribution', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # RMSSD
    if 'rmssd' in df.columns:
        rmssd_data = df['rmssd'].dropna()
        if len(rmssd_data) > 0:
            axes[0,1].hist(rmssd_data, bins=20, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
            axes[0,1].set_xlabel('RMSSD (ms)')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].set_title('B) RMSSD Distribution', fontweight='bold')
            axes[0,1].grid(True, alpha=0.3)
    
    # Beat correlation ratio
    if 'beat_corr_ratio' in df.columns:
        beat_corr_data = df['beat_corr_ratio'].dropna()
        if len(beat_corr_data) > 0:
            axes[1,0].hist(beat_corr_data, bins=20, color='gold', alpha=0.7, edgecolor='darkorange')
            axes[1,0].set_xlabel('Beat Correlation Ratio')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('C) Beat Correlation Ratio', fontweight='bold')
            axes[1,0].grid(True, alpha=0.3)
    
    # Inter-beat correlation ratio
    if 'interbeat_corr_ratio' in df.columns:
        interbeat_data = df['interbeat_corr_ratio'].dropna()
        if len(interbeat_data) > 0:
            axes[1,1].hist(interbeat_data, bins=20, color='violet', alpha=0.7, edgecolor='purple')
            axes[1,1].set_xlabel('Inter-beat Correlation Ratio')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_title('D) Inter-beat Correlation Ratio', fontweight='bold')
            axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_folder + file_prefix + 'physiological_parameters.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Age vs Physiological parameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Age vs Heart Rate
    hr_data = df[['age', 'hr']].dropna()
    if len(hr_data) > 1:
        axes[0,0].scatter(hr_data['age'], hr_data['hr'], alpha=0.5, color='steelblue', s=20)
        z = np.polyfit(hr_data['age'], hr_data['hr'], 1)
        p = np.poly1d(z)
        axes[0,0].plot(hr_data['age'], p(hr_data['age']), "r--", alpha=0.8)
        axes[0,0].set_xlabel('Age (years)')
        axes[0,0].set_ylabel('Heart Rate (bpm)')
        corr_coef = hr_data['age'].corr(hr_data['hr'])
        axes[0,0].set_title(f'A) Age vs Heart Rate\nr = {corr_coef:.3f}', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
    
    # Age vs RMSSD
    if 'rmssd' in df.columns:
        rmssd_corr_data = df[['age', 'rmssd']].dropna()
        if len(rmssd_corr_data) > 1:
            axes[0,1].scatter(rmssd_corr_data['age'], rmssd_corr_data['rmssd'], alpha=0.5, color='forestgreen', s=20)
            z = np.polyfit(rmssd_corr_data['age'], rmssd_corr_data['rmssd'], 1)
            p = np.poly1d(z)
            axes[0,1].plot(rmssd_corr_data['age'], p(rmssd_corr_data['rmssd']), "r--", alpha=0.8)
            axes[0,1].set_xlabel('Age (years)')
            axes[0,1].set_ylabel('RMSSD (ms)')
            corr_coef = rmssd_corr_data['age'].corr(rmssd_corr_data['rmssd'])
            axes[0,1].set_title(f'B) Age vs RMSSD\nr = {corr_coef:.3f}', fontweight='bold')
            axes[0,1].grid(True, alpha=0.3)
    
    # Age vs Beat Correlation (by sex)
    if 'beat_corr_ratio' in df.columns:
        beat_corr_plot_data = df[['age', 'beat_corr_ratio', 'sex_label']].dropna()
        valid_sex_data = beat_corr_plot_data[beat_corr_plot_data['sex_label'].notna()]
        
        for sex in valid_sex_data['sex_label'].unique():
            subset = valid_sex_data[valid_sex_data['sex_label'] == sex]
            axes[1,0].scatter(subset['age'], subset['beat_corr_ratio'], 
                             alpha=0.6, s=20, label=sex)
        axes[1,0].set_xlabel('Age (years)')
        axes[1,0].set_ylabel('Beat Correlation Ratio')
        axes[1,0].set_title('C) Age vs Beat Correlation Ratio', fontweight='bold')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # Age distribution by sex (boxplot)
    valid_sex_age_data = df[['age', 'sex_label']].dropna()
    male_data = valid_sex_age_data[valid_sex_age_data['sex_label'] == 'Male']['age']
    female_data = valid_sex_age_data[valid_sex_age_data['sex_label'] == 'Female']['age']
    
    if len(male_data) > 0 and len(female_data) > 0:
        bp = axes[1,1].boxplot([male_data, female_data],
                              labels=['Male', 'Female'], patch_artist=True)
        # Color the boxes
        colors = ['lightblue', 'lightpink']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axes[1,1].set_ylabel('Age (years)')
        axes[1,1].set_title('D) Age Distribution by Sex', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_folder + file_prefix + 'age_correlations.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation heatmap
    numerical_vars = ['age', 'hr', 'rmssd', 'beat_corr_ratio', 'interbeat_corr_ratio']
    corr_vars = [var for var in numerical_vars if var in df.columns]
    
    # Filter only variables with sufficient data
    corr_data = df[corr_vars].dropna()
    if len(corr_data) > 1 and len(corr_vars) > 1:
        plt.figure(figsize=(8, 6))
        corr_matrix = corr_data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.3f',
                   cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix of Key Variables', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_folder + file_prefix + 'correlation_heatmap.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Age group analysis
    plt.figure(figsize=(10, 6))
    
    # Age group distribution
    age_group_counts = df['age_group'].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0, 1, len(age_group_counts)))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(age_group_counts.index.astype(str), age_group_counts.values, color=colors)
    plt.xlabel('Age Group')
    plt.ylabel('Number of Subjects')
    plt.title('A) Age Group Distribution', fontweight='bold')
    plt.xticks(rotation=45)
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Age group by sex
    plt.subplot(1, 2, 2)
    age_sex_cross = pd.crosstab(df['age_group'], df['sex_label'])
    age_sex_cross.plot(kind='bar', color=['steelblue', 'coral'], ax=plt.gca())
    plt.xlabel('Age Group')
    plt.ylabel('Number of Subjects')
    plt.title('B) Age Group Distribution by Sex', fontweight='bold')
    plt.legend(title='Sex')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_folder + file_prefix + 'age_group_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Additional analysis: Signal quality metrics by age and sex
    if all(col in df.columns for col in ['beat_corr_ratio', 'interbeat_corr_ratio', 'age', 'sex_label']):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Beat correlation ratio by age group and sex
        beat_corr_data = df[['age_group', 'sex_label', 'beat_corr_ratio']].dropna()
        if len(beat_corr_data) > 0:
            sns.boxplot(data=beat_corr_data, x='age_group', y='beat_corr_ratio', hue='sex_label', 
                       ax=axes[0], palette=['steelblue', 'coral'])
            axes[0].set_xlabel('Age Group')
            axes[0].set_ylabel('Beat Correlation Ratio')
            axes[0].set_title('A) Beat Correlation Ratio by Age Group and Sex', fontweight='bold')
            axes[0].legend(title='Sex')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Inter-beat correlation ratio by age group and sex
        interbeat_data = df[['age_group', 'sex_label', 'interbeat_corr_ratio']].dropna()
        if len(interbeat_data) > 0:
            sns.boxplot(data=interbeat_data, x='age_group', y='interbeat_corr_ratio', hue='sex_label', 
                       ax=axes[1], palette=['steelblue', 'coral'])
            axes[1].set_xlabel('Age Group')
            axes[1].set_ylabel('Inter-beat Correlation Ratio')
            axes[1].set_title('B) Inter-beat Correlation Ratio by Age Group and Sex', fontweight='bold')
            axes[1].legend(title='Sex')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_folder + file_prefix + 'signal_quality_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def generate_publication_tables(df):
    """Generate publication-ready tables"""
    print("\nGenerating publication-ready tables...")
    
    # Table 1: Demographic characteristics
    demo_table = df.groupby('sex_label').agg({
        'age': ['count', 'mean', 'std', 'min', 'max'],
        'hr': ['mean', 'std'],
        'rmssd': ['mean', 'std'] if 'rmssd' in df.columns else None
    }).round(2)
    
    # Flatten column names
    demo_table.columns = ['_'.join(col).strip() for col in demo_table.columns.values]
    demo_table.to_csv(output_folder + file_prefix + 'table1_demographics.csv')
    
    # Table 2: Correlation matrix with significance
    numerical_vars = ['age', 'hr', 'rmssd', 'beat_corr_ratio', 'interbeat_corr_ratio']
    corr_vars = [var for var in numerical_vars if var in df.columns]
    
    if len(corr_vars) > 1:
        corr_matrix = df[corr_vars].corr()
        # Add significance stars
        corr_table = corr_matrix.round(3).astype(str)
        for i in range(len(corr_vars)):
            for j in range(i+1, len(corr_vars)):
                corr_data = df[[corr_vars[i], corr_vars[j]]].dropna()
                if len(corr_data) > 2:
                    corr, p_val = stats.pearsonr(corr_data[corr_vars[i]], corr_data[corr_vars[j]])
                    if p_val < 0.001:
                        star = '***'
                    elif p_val < 0.01:
                        star = '**'
                    elif p_val < 0.05:
                        star = '*'
                    else:
                        star = ''
                    corr_table.iloc[i, j] = f"{corr_table.iloc[i, j]}{star}"
        
        corr_table.to_csv(output_folder + file_prefix + 'table2_correlations.csv')
    
    # Table 3: Signal quality metrics by age group
    if all(col in df.columns for col in ['beat_corr_ratio', 'interbeat_corr_ratio', 'age_group']):
        signal_quality_table = df.groupby('age_group').agg({
            'beat_corr_ratio': ['count', 'mean', 'std', 'min', 'max'],
            'interbeat_corr_ratio': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        signal_quality_table.columns = ['_'.join(col).strip() for col in signal_quality_table.columns.values]
        signal_quality_table.to_csv(output_folder + file_prefix + 'table3_signal_quality.csv')
    
    print("Publication tables generated and saved.")

def main():
    """Main analysis function"""
    print("COMPREHENSIVE PPG REGRESSION DATASET ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_and_preprocess_data()
    
    print(f"Dataset loaded: {len(df)} records")
    print(f"Variables: {list(df.columns)}")
    
    # Perform analyses
    stats_df = comprehensive_statistical_summary(df)
    normality_df = normality_tests(df)
    age_sex_stats, age_group_dist = demographic_analysis(df)
    corr_matrix, corr_pvalues = correlation_analysis(df)
    
    # Create PPG signal analysis by decades
    print("\n" + "="*60)
    print("PPG SIGNAL ANALYSIS BY DECADES")
    print("="*60)
    
    # Calculate average PPG signals by decades
    ppg_results = calculate_average_signals(df)
    
    # Create PPG signal plots
    plot_ppg_signals_by_decades(ppg_results)
    
    # Create heart rate analysis by decades
    hr_stats_df = plot_heart_rate_by_decades(df)
    
    # Create other visualizations
    create_publication_plots(df)
    generate_publication_tables(df)
    
    # Summary report
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*80)
    print(f"Total subjects: {len(df)}")
    print(f"Age range: {df['age'].min()} - {df['age'].max()} years")
    print(f"Sex distribution: {df['sex_label'].value_counts().to_dict()}")
    print(f"Male (0): {len(df[df['sex'] == 0])} subjects")
    print(f"Female (1): {len(df[df['sex'] == 1])} subjects")
    print(f"Key variables analyzed: {len(stats_df)}")
    print(f"Output files saved to: {output_folder}")
    
    print("\nGenerated PPG analysis files:")
    print("- ppg_decades_overall_grid.png")
    print("- ppg_decades_sex_comparison.png")
    print("- sex_comparison_[decade].png (for each decade)")
    print("- heart_rate_by_decades.png")
    print("- heart_rate_stats_by_decades.csv")
    print("="*80)

if __name__ == "__main__":
    main()