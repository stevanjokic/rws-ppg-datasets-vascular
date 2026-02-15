"""
================================================================================
Integrated Gradients for PPG Signal Classification
================================================================================

This script demonstrates the use of Integrated Gradients (IG) to explain
predictions of ML classifier trained on raw PPG signals.

The pipeline:
1. Load PPG signals (template_ppg_norm) from the classification dataset
2. Balance classes (100 samples per class)
3. Train a MLP classifier
4. Apply Integrated Gradients to understand which parts of the signal
   drive the classification decisions

================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Configuration parameters for Integrated Gradients analysis."""
    
    # Paths
    DATA_PATH = "data/train_rws_ppg_classification_dataset.csv"
    OUTPUT_DIR = "results/integrated_gradients_analysis"
    
    # Data parameters
    SIGNAL_COL = 'template_ppg_norm'  # Column containing normalized PPG signals
    TARGET_LENGTH = 100  # Expected signal length
    SAMPLES_PER_CLASS = 200  # Number of samples to use per class
    TEST_SIZE = 0.4  # Proportion for test set
    RANDOM_SEED = 42
    
    # Model parameters
    HIDDEN_LAYERS = [256, 128]  # Hidden layer sizes
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    EPOCHS = 120
    PATIENCE = 13  # Early stopping patience
    
    # Integrated Gradients parameters
    IG_STEPS = 100  # Number of steps for approximation
    BASELINE = 'zero'  # 'zero' or 'uniform'
    
    # Visualization parameters
    CLASS_NAMES = ['Class 1 (Young)', 'Class 2', 'Class 3', 'Class 4 (Aged)']
    CLASS_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    FIGSIZE = (12, 10)
    DPI = 300
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================

class PPGDataLoader:
    """Load and preprocess PPG signals for MLP classification."""
    
    def __init__(self, config: Config):
        self.config = config
        self.df = None
        self.signals = None
        self.labels = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self) -> 'PPGDataLoader':
        """Load dataset and extract signals."""
        print(f"\n📊 Loading data from {self.config.DATA_PATH}")
        self.df = pd.read_csv(self.config.DATA_PATH)
        
        # Check if signal column exists
        if self.config.SIGNAL_COL not in self.df.columns:
            raise ValueError(f"Signal column '{self.config.SIGNAL_COL}' not found in dataset")
        
        print(f"   Total samples: {len(self.df)}")
        print(f"   Available columns: {list(self.df.columns)}")
        
        return self
    
    def parse_signals(self) -> 'PPGDataLoader':
        """Parse signal strings into numpy arrays."""
        print("\n🔧 Parsing PPG signals...")
        
        signals = []
        valid_indices = []
        parse_errors = 0
        
        for idx, signal_str in enumerate(self.df[self.config.SIGNAL_COL]):
            try:
                if not isinstance(signal_str, str) or not signal_str.strip():
                    parse_errors += 1
                    continue
                
                # Parse comma-separated string
                signal_list = [float(x) for x in signal_str.split(',') if x.strip()]
                
                # Check if signal has reasonable length
                if len(signal_list) < 10:  # Too short
                    parse_errors += 1
                    continue
                
                # Pad or truncate to target length
                if len(signal_list) < self.config.TARGET_LENGTH:
                    # Pad with edge values (maintains signal shape)
                    pad_left = (self.config.TARGET_LENGTH - len(signal_list)) // 2
                    pad_right = self.config.TARGET_LENGTH - len(signal_list) - pad_left
                    
                    # Use edge values for padding
                    left_pad = [signal_list[0]] * pad_left
                    right_pad = [signal_list[-1]] * pad_right
                    signal_list = left_pad + signal_list + right_pad
                    
                elif len(signal_list) > self.config.TARGET_LENGTH:
                    # Center crop
                    start = (len(signal_list) - self.config.TARGET_LENGTH) // 2
                    signal_list = signal_list[start:start + self.config.TARGET_LENGTH]
                
                # Normalize to [0, 1] (if not already normalized)
                signal_array = np.array(signal_list, dtype=np.float32)
                signal_array = (signal_array - signal_array.min()) / (signal_array.max() - signal_array.min() + 1e-8)
                
                signals.append(signal_array)
                valid_indices.append(idx)
                
            except Exception as e:
                parse_errors += 1
                continue
        
        self.signals = np.array(signals)
        self.labels = self.df.iloc[valid_indices]['class_label'].values
        
        print(f"   Successfully parsed: {len(self.signals)} signals")
        print(f"   Parse errors: {parse_errors}")
        print(f"   Signal shape: {self.signals.shape}")
        print(f"   Class distribution: {np.bincount(self.labels.astype(int))}")
        
        return self
    
    def balance_classes(self) -> 'PPGDataLoader':
        """Balance classes by sampling equal number from each class."""
        print("\n Balancing classes...")
        
        unique_classes = np.unique(self.labels)
        balanced_signals = []
        balanced_labels = []
        
        for class_idx in unique_classes:
            class_indices = np.where(self.labels == class_idx)[0]
            
            if len(class_indices) >= self.config.SAMPLES_PER_CLASS:
                # Sample without replacement
                np.random.seed(self.config.RANDOM_SEED + int(class_idx))
                selected = np.random.choice(class_indices, 
                                          self.config.SAMPLES_PER_CLASS, 
                                          replace=False)
            else:
                # Sample with replacement if not enough samples
                print(f"    Class {class_idx} has only {len(class_indices)} samples, using with replacement")
                np.random.seed(self.config.RANDOM_SEED + int(class_idx))
                selected = np.random.choice(class_indices, 
                                          self.config.SAMPLES_PER_CLASS, 
                                          replace=True)
            
            balanced_signals.append(self.signals[selected])
            balanced_labels.append(self.labels[selected])
        
        self.signals = np.vstack(balanced_signals)
        self.labels = np.concatenate(balanced_labels)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(self.signals))
        self.signals = self.signals[shuffle_idx]
        self.labels = self.labels[shuffle_idx]
        
        print(f"   Balanced dataset: {len(self.signals)} samples")
        print(f"   New distribution: {np.bincount(self.labels.astype(int))}")
        
        return self
    
    def get_train_test_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.signals, self.labels,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_SEED,
            stratify=self.labels
        )
        
        print(f"\n Train-test split:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Test:  {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test


# ==============================================================================
# MLP MODEL DEFINITION
# ==============================================================================

class SimpleMLP(nn.Module):
    """
    Simple MLP for PPG signal classification.
    Input: 1D signal of length `input_dim`
    Output: logits for 4 classes
    """
    
    def __init__(self, input_dim: int = 100, hidden_layers: List[int] = [128, 64],
                 num_classes: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)


# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================

def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, config: Config) -> Dict:
    """
    Train the MLP model with early stopping.
    """
    print("\n Training MLP model...")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(config.DEVICE)
    y_train_t = torch.LongTensor(y_train).to(config.DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(config.DEVICE)
    y_val_t = torch.LongTensor(y_val).to(config.DEVICE)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_pred = val_outputs.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_pred)
        
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"{config.OUTPUT_DIR}/best_mlp_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"   Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(f"{config.OUTPUT_DIR}/best_mlp_model.pth"))
    
    print(f"\n Training complete! Best validation accuracy: {max(history['val_acc']):.4f}")
    
    return history


# ==============================================================================
# INTEGRATED GRADIENTS IMPLEMENTATION
# ==============================================================================

class IntegratedGradients:
    """
    Integrated Gradients for MLP models.
    
    Computes attribution scores for each input feature (time point) indicating
    how much it contributed to the model's prediction.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def _compute_gradients(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Compute gradients of output w.r.t input.
        """
        x = x.clone().detach().requires_grad_(True)
        outputs = self.model(x)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Compute gradients for target class
        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class] = 1.0
        outputs.backward(gradient=one_hot)
        
        return x.grad
    
    def explain(self, x: torch.Tensor, target_class: Optional[int] = None,
                steps: int = 50, baseline: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Compute Integrated Gradients for input x.
        
        Args:
            x: Input tensor of shape (1, input_dim)
            target_class: Target class (if None, uses predicted class)
            steps: Number of steps for approximation
            baseline: Baseline input (if None, uses zeros)
        
        Returns:
            attributions: Array of shape (input_dim,) with attribution scores
        """
        x = x.to(self.device)
        
        # Determine target class
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(x)
                target_class = outputs.argmax(dim=1).item()
        
        # Create baseline
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.to(self.device)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps + 1).to(self.device)
        interpolated = baseline + alphas[:, None] * (x - baseline)
        
        # Compute gradients for each interpolated point
        total_gradients = torch.zeros_like(x)
        
        for i, alpha in enumerate(alphas):
            interp = interpolated[i:i+1]
            grad = self._compute_gradients(interp, target_class)
            total_gradients += grad
        
        # Approximate integral
        avg_gradients = total_gradients / steps
        
        # Compute attributions
        attributions = (x - baseline) * avg_gradients
        
        return attributions.squeeze().cpu().numpy()
    
    def explain_batch(self, X: torch.Tensor, targets: Optional[List[int]] = None,
                     steps: int = 50) -> np.ndarray:
        """
        Compute Integrated Gradients for a batch of inputs.
        """
        attributions = []
        
        for i in range(len(X)):
            target = targets[i] if targets is not None else None
            attr = self.explain(X[i:i+1], target, steps)
            attributions.append(attr)
        
        return np.array(attributions)


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

class IGVisualizer:
    """Create publication-quality visualizations of Integrated Gradients results."""
    
    def __init__(self, config: Config, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set publication style
        plt.style.use('default')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['figure.titlesize'] = 14
    
    def plot_single_sample(self, signal: np.ndarray, attribution: np.ndarray,
                          true_class: int, pred_class: int, confidence: float,
                          sample_id: int) -> plt.Figure:
        """
        Create a single-panel figure showing signal with attribution overlay.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        time = np.arange(len(signal)) / 100  # Convert to seconds (assuming 100 Hz)
        
        # Top panel: Signal with colored attribution
        ax1.plot(time, signal, 'k-', linewidth=1.5, alpha=0.7, label='PPG Signal')
        
        # Color attribution regions
        pos_mask = attribution > 0
        neg_mask = attribution < 0
        
        if pos_mask.any():
            ax1.fill_between(time, signal.min(), signal, where=pos_mask,
                            color='red', alpha=0.3, label='Positive contribution')
        if neg_mask.any():
            ax1.fill_between(time, signal.min(), signal, where=neg_mask,
                            color='blue', alpha=0.3, label='Negative contribution')
        
        # Mark physiologically relevant regions
        ax1.axvspan(0.15, 0.35, alpha=0.1, color='green', label='Systolic region')
        ax1.axvspan(0.4, 0.6, alpha=0.1, color='yellow', label='Dicrotic notch region')
        
        ax1.set_ylabel('Normalized Amplitude', fontsize=11)
        ax1.set_title(f'PPG Signal with Attribution - {self.config.CLASS_NAMES[true_class]} '
                     f'(Conf: {confidence:.3f})', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)
        
        # Bottom panel: Attribution values
        colors = ['red' if a > 0 else 'blue' for a in attribution]
        ax2.bar(time, attribution, width=0.01, color=colors, alpha=0.7, edgecolor='none')
        ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax2.set_xlabel('Time (seconds)', fontsize=11)
        ax2.set_ylabel('Attribution', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save
        filename = f"{self.output_dir}/ig_sample_{sample_id}_class{true_class}.png"
        plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_average_attributions(self, attributions: np.ndarray, labels: np.ndarray,
                                 signals: np.ndarray) -> plt.Figure:
        """
        Plot average attribution per class.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        time = np.arange(attributions.shape[1]) / 100
        
        for class_idx in range(4):
            ax = axes[class_idx]
            class_mask = labels == class_idx
            
            if not class_mask.any():
                ax.text(0.5, 0.5, f'No samples for {self.config.CLASS_NAMES[class_idx]}',
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            class_attrs = attributions[class_mask]
            class_signals = signals[class_mask]
            
            # Average signal
            mean_signal = class_signals.mean(axis=0)
            std_signal = class_signals.std(axis=0)
            
            # Average attribution
            mean_attr = class_attrs.mean(axis=0)
            std_attr = class_attrs.std(axis=0)
            
            # Plot signal
            ax.plot(time, mean_signal, color='black', linewidth=2, label='Mean signal')
            ax.fill_between(time, mean_signal - std_signal, mean_signal + std_signal,
                           color='black', alpha=0.1)
            
            # Plot attribution as colored bars
            ax2 = ax.twinx()
            ax2.bar(time, mean_attr, width=0.01, color=self.config.CLASS_COLORS[class_idx],
                   alpha=0.5, label='Mean attribution', edgecolor='none')
            ax2.fill_between(time, mean_attr - std_attr, mean_attr + std_attr,
                            color=self.config.CLASS_COLORS[class_idx], alpha=0.2)
            ax2.set_ylabel('Attribution', fontsize=10, color=self.config.CLASS_COLORS[class_idx])
            ax2.tick_params(axis='y', labelcolor=self.config.CLASS_COLORS[class_idx])
            ax2.set_ylim(-0.5, 0.5)
            
            # Styling
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.set_title(self.config.CLASS_NAMES[class_idx], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Highlight regions
            ax.axvspan(0.15, 0.35, alpha=0.1, color='green')
            ax.axvspan(0.4, 0.6, alpha=0.1, color='yellow')
        
        plt.suptitle('Average Integrated Gradients by Class', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f"{self.output_dir}/ig_average_by_class.png"
        plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_attribution_heatmap(self, attributions: np.ndarray, labels: np.ndarray,
                                sort_by_class: bool = True) -> plt.Figure:
        """
        Create a heatmap of attributions for all samples.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by class if requested
        if sort_by_class:
            sort_idx = np.argsort(labels)
            attributions_sorted = attributions[sort_idx]
            labels_sorted = labels[sort_idx]
            
            # Add class boundaries
            class_boundaries = []
            current_class = -1
            for i, label in enumerate(labels_sorted):
                if label != current_class:
                    class_boundaries.append(i)
                    current_class = label
            class_boundaries.append(len(labels_sorted))
        else:
            attributions_sorted = attributions
            labels_sorted = labels
        
        # Create heatmap
        im = ax.imshow(attributions_sorted, aspect='auto', cmap='RdBu_r',
                      vmin=-0.5, vmax=0.5, interpolation='bilinear')
        
        # Add class boundaries
        if sort_by_class:
            for i in range(1, len(class_boundaries) - 1):
                ax.axhline(y=class_boundaries[i] - 0.5, color='white', linewidth=2)
            
            # Add class labels
            for i in range(4):
                y_pos = (class_boundaries[i] + class_boundaries[i+1]) / 2
                ax.text(-5, y_pos, self.config.CLASS_NAMES[i].replace(' (Aged)', ''),
                       ha='right', va='center', fontsize=10, fontweight='bold')
        
        # Labels
        ax.set_xlabel('Time Point', fontsize=12)
        ax.set_ylabel('Sample', fontsize=12)
        ax.set_title('Integrated Gradients Attribution Heatmap', fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attribution', rotation=270, labelpad=20, fontsize=11)
        
        plt.tight_layout()
        
        filename = f"{self.output_dir}/ig_attribution_heatmap.png"
        plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_detailed_attribution_heatmap(self, attributions: np.ndarray, labels: np.ndarray):
        """
        Detailed heatmap with samples on y-axis (for supplementary materials).
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Sort by class
        sort_idx = np.argsort(labels)
        attributions_sorted = attributions[sort_idx]
        labels_sorted = labels[sort_idx]
        
        # Create heatmap
        im = ax.imshow(attributions_sorted, aspect='auto', cmap='RdBu_r',
                    vmin=-0.5, vmax=0.5, interpolation='bilinear')
        
        # Add class boundaries and labels
        boundaries = [0]
        current_class = labels_sorted[0]
        
        for i, label in enumerate(labels_sorted):
            if label != current_class:
                boundaries.append(i)
                # Add class label at the middle of the region
                mid = (boundaries[-2] + i) // 2
                ax.text(-5, mid, self.config.CLASS_NAMES[current_class],
                    ha='right', va='center', fontsize=10, fontweight='bold')
                current_class = label
        boundaries.append(len(labels_sorted))
        mid = (boundaries[-2] + len(labels_sorted)) // 2
        ax.text(-5, mid, self.config.CLASS_NAMES[current_class],
            ha='right', va='center', fontsize=10, fontweight='bold')
        
        # Draw boundary lines
        for i in range(1, len(boundaries) - 1):
            ax.axhline(y=boundaries[i] - 0.5, color='white', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time Point', fontsize=12)
        ax.set_ylabel('Sample (sorted by class)', fontsize=12)
        ax.set_title('Integrated Gradients Attribution Heatmap (Detailed)', 
                    fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attribution', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/ig_attribution_heatmap_detailed.png", 
                    dpi=self.config.DPI, bbox_inches='tight')
        plt.close()

    def plot_comprehensive_figure(self, attributions: np.ndarray, labels: np.ndarray,
                             signals: np.ndarray, preds: np.ndarray,
                             confidences: np.ndarray) -> plt.Figure:
        """
        Create a comprehensive figure for publication combining multiple views.
        Fixed: Perfect alignment between upper plots and heatmap.
        """
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3, 
                            height_ratios=[1, 1, 1, 0.8, 0.8])
        
        # Create a common time axis for ALL subplots
        n_time_points = signals.shape[1]
        time_seconds = np.arange(n_time_points) / 100  # 0 to 1.0 seconds
        time_pixels = np.arange(n_time_points)  # 0 to 99 for image coordinates
        
        # ==========================================================================
        # Rows 1-2: Representative samples from each class
        # ==========================================================================
        for class_idx in range(4):
            class_mask = labels == class_idx
            if not class_mask.any():
                continue
            
            class_confs = confidences[class_mask]
            sorted_idx = np.argsort(class_confs)
            median_pos = len(sorted_idx) // 2
            
            idx1 = max(0, median_pos - 2)
            idx2 = min(len(sorted_idx) - 1, median_pos + 2)
            rep_indices = [idx1, idx2]
            
            for row, rep_idx in enumerate(rep_indices):
                actual_idx = np.where(class_mask)[0][rep_idx]
                
                ax = fig.add_subplot(gs[row, class_idx])
                
                signal = signals[actual_idx]
                attr = attributions[actual_idx]
                
                # Use time_seconds for x-axis
                ax.plot(time_seconds, signal, 'k-', linewidth=1.5, alpha=0.7)
                
                # Attribution as colored fill
                pos_mask = attr > 0
                neg_mask = attr < 0
                
                if pos_mask.any():
                    ax.fill_between(time_seconds, signal.min(), signal, where=pos_mask,
                                color='red', alpha=0.3)
                if neg_mask.any():
                    ax.fill_between(time_seconds, signal.min(), signal, where=neg_mask,
                                color='blue', alpha=0.3)
                
                ax.text(0.02, 0.95, f'conf: {confidences[actual_idx]:.2f}',
                    transform=ax.transAxes, fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                if row == 0:
                    ax.set_title(f'{self.config.CLASS_NAMES[class_idx]}', 
                            fontsize=10, fontweight='bold')
                
                # CRITICAL: Set x-axis limits to match time_seconds exactly
                ax.set_xlim(0, 1.0)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.grid(True, alpha=0.2)
        
        # ==========================================================================
        # Row 3: Average attributions by class
        # ==========================================================================
        ax_avg = fig.add_subplot(gs[2, :])
        
        for class_idx in range(4):
            class_mask = labels == class_idx
            if class_mask.any():
                mean_attr = attributions[class_mask].mean(axis=0)
                std_attr = attributions[class_mask].std(axis=0)
                
                ax_avg.plot(time_seconds, mean_attr, color=self.config.CLASS_COLORS[class_idx],
                        linewidth=2, label=self.config.CLASS_NAMES[class_idx])
                ax_avg.fill_between(time_seconds, mean_attr - std_attr, mean_attr + std_attr,
                                color=self.config.CLASS_COLORS[class_idx], alpha=0.2)
        
        ax_avg.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax_avg.set_xlim(0, 1.0)  # Match time_seconds
        ax_avg.set_xlabel('Time (seconds)', fontsize=11)
        ax_avg.set_ylabel('Mean Attribution', fontsize=11)
        ax_avg.set_title('Average Attribution by Class', fontsize=12, fontweight='bold')
        ax_avg.grid(True, alpha=0.3)
        ax_avg.legend(loc='upper right', fontsize=9, ncol=2)
        
        # ==========================================================================
        # Rows 4-5: Heatmap with class labels on y-axis
        # ==========================================================================
        ax_heat = fig.add_subplot(gs[3:5, :])
        
        # Sort by class
        sort_idx = np.argsort(labels)
        attributions_sorted = attributions[sort_idx]
        labels_sorted = labels[sort_idx]
        
        # CRITICAL: Use extent to map pixels to time_seconds
        # This ensures the image spans exactly from 0 to 1.0 on x-axis
        im = ax_heat.imshow(attributions_sorted, aspect='auto', cmap='RdBu_r',
                        vmin=-0.5, vmax=0.5, interpolation='bilinear',
                        extent=[0, 1.0, len(attributions), 0])  # [left, right, bottom, top]
        
        # Add class boundaries and labels on y-axis
        boundaries = [0]
        current_class = labels_sorted[0]
        class_positions = []
        class_labels_added = []
        
        for i, label in enumerate(labels_sorted):
            if label != current_class:
                boundaries.append(i)
                # Store middle position for class label (in pixel coordinates)
                class_positions.append((boundaries[-2] + i) / 2)
                class_labels_added.append(self.config.CLASS_NAMES[current_class])
                current_class = label
        boundaries.append(len(labels_sorted))
        class_positions.append((boundaries[-2] + len(labels_sorted)) / 2)
        class_labels_added.append(self.config.CLASS_NAMES[current_class])
        
        # Draw boundary lines - now using pixel coordinates
        for i in range(1, len(boundaries) - 1):
            ax_heat.axhline(y=boundaries[i] - 0.5, color='white', linewidth=2, alpha=0.8)
        
        # Set y-axis ticks to class positions
        ax_heat.set_yticks(class_positions)
        ax_heat.set_yticklabels(class_labels_added, fontsize=10, fontweight='bold')
        
        # CRITICAL: x-axis is now in seconds (0 to 1.0) because of extent
        ax_heat.set_xlabel('Time (seconds)', fontsize=11)
        ax_heat.set_ylabel('Class', fontsize=11)
        ax_heat.set_title('Attribution Heatmap (samples sorted by class)', 
                        fontsize=12, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.02)
        cbar.set_label('Attribution', rotation=270, labelpad=20, fontsize=11)
        
        plt.suptitle('Integrated Gradients Analysis of PPG Morphological Classes',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        filename = f"{self.output_dir}/ig_comprehensive_figure.png"
        plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')
        plt.close()
        
        return fig
    

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("INTEGRATED GRADIENTS FOR PPG SIGNAL CLASSIFICATION")
    print("="*70)
    
    # Initialize configuration
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print(f"\n Output directory: {config.OUTPUT_DIR}")
    print(f" Device: {config.DEVICE}")
    
    # Load and prepare data
    data_loader = PPGDataLoader(config)
    data_loader.load_data().parse_signals().balance_classes()
    X_train, X_test, y_train, y_test = data_loader.get_train_test_split()
    
    # Further split training for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=config.RANDOM_SEED,
        stratify=y_train
    )
    
    print(f"\n Final splits:")
    print(f"   Train: {len(X_train)}")
    print(f"   Val:   {len(X_val)}")
    print(f"   Test:  {len(X_test)}")
    
    # Create and train model
    model = SimpleMLP(
        input_dim=config.TARGET_LENGTH,
        hidden_layers=config.HIDDEN_LAYERS,
        num_classes=4,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)
    
    history = train_model(model, X_train, y_train, X_val, y_val, config)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/training_history.png", dpi=config.DPI)
    plt.close()
    
    # Evaluate on test set
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(config.DEVICE)
    y_test_t = torch.LongTensor(y_test).to(config.DEVICE)
    
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_pred = test_outputs.argmax(dim=1).cpu().numpy()
        test_probs = torch.softmax(test_outputs, dim=1).cpu().numpy()
    
    print("\n📊 Test Set Performance:")
    print(f"   Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, test_pred, 
                                target_names=config.CLASS_NAMES))
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/confusion_matrix.png", dpi=config.DPI)
    plt.close()
    
    # Compute Integrated Gradients
    print("\n Computing Integrated Gradients...")
    ig = IntegratedGradients(model, config.DEVICE)
    
    # Select a subset of test samples for explanation
    n_explain = min(100, len(X_test))
    explain_indices = np.random.choice(len(X_test), n_explain, replace=False)
    
    X_explain = X_test[explain_indices]
    y_explain = y_test[explain_indices]
    pred_explain = test_pred[explain_indices]
    conf_explain = test_probs[explain_indices].max(axis=1)
    
    # Convert to tensors
    X_explain_t = torch.FloatTensor(X_explain).to(config.DEVICE)
    
    # Compute attributions
    attributions = ig.explain_batch(X_explain_t, steps=config.IG_STEPS)
    
    print(f"   Computed attributions for {len(attributions)} samples")
    print(f"   Attribution shape: {attributions.shape}")
    
    # Save numerical results
    results_df = pd.DataFrame({
        'sample_idx': explain_indices,
        'true_class': y_explain,
        'pred_class': pred_explain,
        'confidence': conf_explain
    })
    
    # Add attribution columns
    for i in range(attributions.shape[1]):
        results_df[f'time_{i}'] = attributions[:, i]
    
    results_df.to_csv(f"{config.OUTPUT_DIR}/ig_results.csv", index=False)
    
    # Create visualizations
    print("\n🎨 Generating visualizations...")
    visualizer = IGVisualizer(config, config.OUTPUT_DIR)
    
    # Plot individual samples
    for i in range(min(12, len(X_explain))):
        visualizer.plot_single_sample(
            X_explain[i], attributions[i],
            y_explain[i], pred_explain[i], conf_explain[i],
            explain_indices[i]
        )
    
    # Plot average attributions by class
    visualizer.plot_average_attributions(attributions, y_explain, X_explain)
    
    # Plot attribution heatmap
    visualizer.plot_attribution_heatmap(attributions, y_explain)
    
    # Plot comprehensive figure
    visualizer.plot_comprehensive_figure(attributions, y_explain, X_explain,
                                       pred_explain, conf_explain)
    
    # Generate summary report
    print("\n Generating summary report...")
    with open(f"{config.OUTPUT_DIR}/summary_report.txt", 'w') as f:
        f.write("INTEGRATED GRADIENTS ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset: {config.DATA_PATH}\n")
        f.write(f"Total samples analyzed: {len(X_explain)}\n")
        f.write(f"Samples per class: {np.bincount(y_explain)}\n\n")
        
        f.write("Model Architecture:\n")
        f.write(f"  Input dimension: {config.TARGET_LENGTH}\n")
        f.write(f"  Hidden layers: {config.HIDDEN_LAYERS}\n")
        f.write(f"  Dropout: {config.DROPOUT_RATE}\n\n")
        
        f.write("Test Performance:\n")
        f.write(f"  Accuracy: {accuracy_score(y_test, test_pred):.4f}\n\n")
        
        f.write("Key Attribution Findings:\n")
        f.write("-"*30 + "\n")
        
        # Find most important time regions per class
        for class_idx in range(4):
            class_mask = y_explain == class_idx
            if class_mask.any():
                mean_attr = attributions[class_mask].mean(axis=0)
                top_regions = np.argsort(np.abs(mean_attr))[-5:]
                f.write(f"\n{config.CLASS_NAMES[class_idx]}:\n")
                f.write(f"  Most influential regions: {sorted(top_regions)}\n")
                f.write(f"  Peak attribution at: {np.argmax(np.abs(mean_attr))}\n")
    
    print(f"\n Analysis complete! Results saved to {config.OUTPUT_DIR}")
    print(f"\nGenerated files:")
    for f in sorted(os.listdir(config.OUTPUT_DIR)):
        print(f"   • {f}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()