import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
MAX_EPOCHS = 150
BATCH_SIZE = 32
PATIENCE = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === IMPROVED EARLY STOPPING ===
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, model, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            self.best_weights = model.state_dict().copy()
            return False
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        return False

# === CUSTOM DATASET ===
class PPGRegressionDataset(Dataset):
    def __init__(self, signals, targets, indices=None):
        self.signals = signals
        self.targets = targets
        self.indices = indices if indices is not None else np.arange(len(signals))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        signal = self.signals[actual_idx]
        target = self.targets[actual_idx]
        return torch.FloatTensor(signal), torch.FloatTensor([target])

# === MODEL DEFINITIONS ===
class PPGMLP(nn.Module):
    """Jednostavan MLP za PPG regresiju"""
    def __init__(self, input_length=100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_length, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class PPG1DCNN(nn.Module):
    def __init__(self, input_length=100):
        super(PPG1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(25)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 25, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# === CORRECTED RESNET IMPLEMENTATION ===
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class PPGResNet(nn.Module):
    def __init__(self, input_length=100):
        super(PPGResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(2)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# === SIGNAL PROCESSING ===
def parse_signal(signal_string, target_length=100):
    """Parse signal string and normalize"""
    try:
        if isinstance(signal_string, str):
            signal_string = signal_string.strip('"')
            signal_array = np.fromstring(signal_string, sep=',')
            
            # Normalize signal
            if len(signal_array) > 0:
                signal_array = (signal_array - np.mean(signal_array)) / (np.std(signal_array) + 1e-8)
            
            # Pad or truncate to target length
            if len(signal_array) < target_length:
                signal_array = np.pad(signal_array, (0, target_length - len(signal_array)), 
                                    mode='constant')
            elif len(signal_array) > target_length:
                signal_array = signal_array[:target_length]
                
            return signal_array
        return np.zeros(target_length)
    except Exception as e:
        print(f"Error parsing signal: {e}")
        return np.zeros(target_length)

def zero_signal_region(signal, od_index, do_index):
    """Set signal regions to zero based on od and do indices"""
    signal = signal.copy()
    if od_index > 0 and od_index < len(signal):
        signal[:od_index] = 0
    if do_index < len(signal) and do_index > 0:
        signal[do_index:] = 0
    return signal

def load_and_preprocess_data(data_path, signal_columns, target_column='age', group_column='app_id'):
    """Load data and preprocess signals"""
    df = pd.read_csv(data_path)
    
    # Filter required columns
    required_cols = [target_column, group_column, 'od', 'do'] + signal_columns
    df = df.dropna(subset=required_cols)
    
    # Check target distribution
    print(f"Target statistics:")
    print(f"  Min: {df[target_column].min()}, Max: {df[target_column].max()}")
    print(f"  Mean: {df[target_column].mean():.2f}, Std: {df[target_column].std():.2f}")
    
    processed_data = {}
    
    for signal_col in signal_columns:
        print(f"Processing {signal_col}...")
        signals = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            try:
                # Parse signal
                signal = parse_signal(row[signal_col])
                
                # Apply zero masking based on od/do indices
                od_idx = int(row['od'])
                do_idx = int(row['do'])
                signal = zero_signal_region(signal, od_idx, do_idx)
                
                signals.append(signal)
                valid_indices.append(idx)
                
            except Exception as e:
                continue
        
        processed_data[signal_col] = {
            'signals': np.array(signals),
            'indices': valid_indices,
            'targets': df.iloc[valid_indices][target_column].values,
            'groups': df.iloc[valid_indices][group_column].values
        }
        
        print(f"  Processed {len(signals)} signals for {signal_col}")
    
    return processed_data, df

# === IMPROVED TRAINING WITH HUBER LOSS ===
def train_model(model, train_loader, val_loader, model_name, results_dir):
    """Train model with early stopping and Huber loss"""
    model.to(DEVICE)
    
    # Use Huber loss for better optimization
    criterion = nn.HuberLoss(delta=1.0)  # Better than MAE for optimization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Remove verbose parameter for compatibility
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for batch_signals, batch_targets in train_loader:
            batch_signals, batch_targets = batch_signals.to(DEVICE), batch_targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_signals)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_signals, batch_targets in val_loader:
                batch_signals, batch_targets = batch_signals.to(DEVICE), batch_targets.to(DEVICE)
                outputs = model(batch_signals)
                val_loss += criterion(outputs, batch_targets).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping check
        if early_stopping(model, val_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{results_dir}/regression_dl_{model_name}_best.pth")
        
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
    
    training_time = time.time() - start_time
    
    # Load best model
    if os.path.exists(f"{results_dir}/regression_dl_{model_name}_best.pth"):
        model.load_state_dict(torch.load(f"{results_dir}/regression_dl_{model_name}_best.pth"))
    
    # Save learning curves
    learning_data = pd.DataFrame({
        'epoch': range(len(train_losses)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    learning_data.to_csv(f"{results_dir}/regression_dl_{model_name}_learning_curve.csv", index=False)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.title(f'Learning Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{results_dir}/regression_dl_{model_name}_learning_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, training_time

# === EVALUATION ===
def evaluate_model(model, test_loader, model_name, signal_name, fold, results_dir):
    """Evaluate model and return metrics"""
    model.eval()
    predictions, targets = [], []
    
    start_time = time.time()
    with torch.no_grad():
        for batch_signals, batch_targets in test_loader:
            batch_signals = batch_signals.to(DEVICE)
            outputs = model(batch_signals)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_targets.numpy())
    
    prediction_time = time.time() - start_time
    
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    # Save predictions
    results_df = pd.DataFrame({
        'actual_age': targets,
        'predicted_age': predictions,
        'error': predictions - targets,
        'absolute_error': np.abs(predictions - targets)
    })
    results_df.to_csv(f"{results_dir}/regression_dl_{model_name}_{signal_name}_fold{fold}_predictions.csv", index=False)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(targets, predictions, alpha=0.6)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title(f'{model_name} - {signal_name}\nR² = {r2:.3f}, MAE = {mae:.3f}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = predictions - targets
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Age')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/regression_dl_{model_name}_{signal_name}_fold{fold}_results.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'model': model_name,
        'signal': signal_name,
        'fold': fold,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'prediction_time': prediction_time
    }

# === MAIN PIPELINE ===
def run_regression_dl_benchmark(data_path, results_dir="results", prefix="regression_dl_"):
    """Main pipeline for deep learning regression benchmark"""
    os.makedirs(results_dir, exist_ok=True)
    
    # Signal columns to process
    signal_columns = ['template_ppg', 'sd_template_ppg']
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    processed_data, original_df = load_and_preprocess_data(data_path, signal_columns)
    
    # Model definitions - SAMO MLP, CNN i ResNet
    models = {
        'MLP': PPGMLP,
        'CNN': PPG1DCNN,
        'ResNet': PPGResNet
    }
    
    all_results = []
    
    for signal_name, signal_data in processed_data.items():
        print(f"\n{'='*50}")
        print(f"Processing signal: {signal_name}")
        print(f"{'='*50}")
        
        X = signal_data['signals']
        y = signal_data['targets']
        groups = signal_data['groups']
        
        # Reshape for CNN models (batch, channels, length)
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
        
        # Group K-Fold cross-validation
        gkf = GroupKFold(n_splits=5)
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_reshaped, y, groups)):
            print(f"\nFold {fold + 1}/5")
            
            # Split data
            X_train, X_test = X_reshaped[train_idx], X_reshaped[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create validation split
            val_size = int(0.1 * len(X_train))
            X_val, y_val = X_train[:val_size], y_train[:val_size]
            X_train, y_train = X_train[val_size:], y_train[val_size:]
            
            # Create data loaders
            train_dataset = PPGRegressionDataset(X_train, y_train)
            val_dataset = PPGRegressionDataset(X_val, y_val)
            test_dataset = PPGRegressionDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            # Train and evaluate each model
            for model_name, model_class in models.items():
                print(f"  Training {model_name}...")
                
                try:
                    # Initialize model
                    model = model_class(input_length=X.shape[1])
                    
                    # Train model
                    trained_model, training_time = train_model(
                        model, train_loader, val_loader, 
                        f"{model_name}_{signal_name}_fold{fold}", 
                        results_dir
                    )
                    
                    # Evaluate model
                    metrics = evaluate_model(
                        trained_model, test_loader, 
                        model_name, signal_name, fold, results_dir
                    )
                    metrics['training_time'] = training_time
                    all_results.append(metrics)
                    
                    print(f"    {model_name}: MAE = {metrics['mae']:.3f}, R² = {metrics['r2']:.3f}")
                    
                except Exception as e:
                    print(f"    Error training {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Save all results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{results_dir}/{prefix}comprehensive_results.csv", index=False)
        
        # Generate summary statistics
        summary_df = results_df.groupby(['model', 'signal']).agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std'],
            'training_time': ['mean', 'std'],
            'prediction_time': ['mean', 'std']
        }).round(4)
        
        summary_df.to_csv(f"{results_dir}/{prefix}summary_statistics.csv")
        
        try:
            summary_df.to_latex(f"{results_dir}/{prefix}summary_statistics.tex")
        except:
            print("Could not generate LaTeX table")
        
        # Create comparison plots
        create_comparison_plots(results_df, results_dir, prefix)
        
        print(f"\n{'='*50}")
        print("BENCHMARK COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: {results_dir}")
        print(f"{'='*50}")
    
    return all_results

def create_comparison_plots(results_df, results_dir, prefix):
    """Create comprehensive comparison plots"""
    
    # Model comparison across signals
    plt.figure(figsize=(15, 10))
    
    metrics = ['mae', 'rmse', 'r2']
    titles = ['Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)', 'R² Score']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        # Prepare data for boxplot
        plot_data = []
        for model in results_df['model'].unique():
            for signal in results_df['signal'].unique():
                values = results_df[(results_df['model'] == model) & 
                                  (results_df['signal'] == signal)][metric].values
                for val in values:
                    plot_data.append({'model': model, 'signal': signal, 'value': val})
        
        plot_df = pd.DataFrame(plot_data)
        
        if not plot_df.empty:
            sns.boxplot(data=plot_df, x='model', y='value', hue='signal', 
                       palette=['#66c2a5', '#fc8d62'])
            plt.title(titles[i], fontweight='bold')
            plt.xlabel('Model')
            plt.ylabel(metric.upper())
            plt.legend(title='Signal Type')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{prefix}model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance by signal type
    plt.figure(figsize=(12, 8))
    
    for i, signal in enumerate(results_df['signal'].unique()):
        plt.subplot(2, 2, i+1)
        signal_data = results_df[results_df['signal'] == signal]
        
        for model in signal_data['model'].unique():
            model_data = signal_data[signal_data['model'] == model]
            plt.scatter([model] * len(model_data), model_data['mae'], 
                       alpha=0.6, s=60, label=model)
        
        plt.title(f'MAE Distribution - {signal}', fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('MAE')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{prefix}signal_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

# === MAIN EXECUTION ===
if __name__ == "__main__":
    DATA_PATH = "data/train_rws_ppg_regression_dataset.csv"
    RESULTS_DIR = "results"
    PREFIX = "regression_dl_"
    
    # Save environment info
    with open(f"{RESULTS_DIR}/regression_dl_environment_info.txt", "w") as f:
        f.write(f"Deep Learning PPG Regression Benchmark\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"Device: {DEVICE}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA device: {torch.cuda.get_device_name()}\n")
    
    # Run benchmark
    run_regression_dl_benchmark(DATA_PATH, RESULTS_DIR, PREFIX)
    