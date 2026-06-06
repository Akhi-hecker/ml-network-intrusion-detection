"""
Network Intrusion Detection System
Implementation of: "A Novel Framework Design of Network Intrusion Detection 
Based on Machine Learning Techniques"

Uses Sparse Autoencoder (SAE) for feature compression
+ Random Forest (RF-10) [Base Paper]
+ XGBoost-100 [Extension]
+ Voting Ensemble (RF + XGBoost, Soft Voting) [Extension]
Dataset: CICIDS2017 (cleaned version with 52 features)

Author: Implementation based on paper by Zhang et al. (2021)
Extension: Voting Ensemble combining RF-10 and XGBoost-100 with soft voting

Label column : 'Attack Type'
Benign label : 'Normal Traffic'
Classes      : Normal Traffic, Port Scanning, Web Attacks, Brute Force, DDoS, Bots, DoS
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import time


# =============================================================================
# TEE LOGGER: Save all terminal output to a text file
# =============================================================================
class TeeLogger:
    """Writes output to both terminal and a log file simultaneously."""
    def __init__(self, log_file_path):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.encoding = getattr(self.terminal, 'encoding', 'utf-8')
        self.fileno = self.terminal.fileno
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    def isatty(self):
        return False
    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal

# Start logging all output
RESULTS_PATH = "../../results/cicids2017/"
os.makedirs(RESULTS_PATH, exist_ok=True)
LOG_FILE = RESULTS_PATH + "training_output_cicids2017.txt"
tee = TeeLogger(LOG_FILE)
sys.stdout = tee

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("NETWORK INTRUSION DETECTION SYSTEM - CICIDS2017")
print("Based on: Sparse Autoencoder + RF-10 + XGBoost-100 + Soft Voting Ensemble")
print("=" * 70)

# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================
print("\n[SECTION 1] DATA LOADING")
print("-" * 50)

DATA_PATH = "../../datasets/cicids2017/cicids2017_cleaned.csv"
print(f"Loading dataset from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Total samples: {len(df)}")
print(f"Total features: {df.shape[1] - 1}")
print(f"Label column: {df.columns[-1]}")
print(f"\nLabel distribution:")
print(df[df.columns[-1]].value_counts())

# =============================================================================
# SECTION 2: PREPROCESSING
# =============================================================================
print("\n[SECTION 2] PREPROCESSING")
print("-" * 50)

# 2.1 Separate features and labels
label_column = df.columns[-1]   # 'Attack Type'
X = df.drop(columns=[label_column])
y = df[label_column]

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# 2.2 Handle missing and infinite values
print("\nHandling missing/infinite values...")
X = X.replace([np.inf, -np.inf], np.nan)
missing_before = X.isnull().sum().sum()
X = X.fillna(0)
print(f"Replaced {missing_before} missing/infinite values with 0")

# 2.3 Ensure all columns are numeric
print("\nConverting all features to numeric...")
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

# 2.4 Create binary labels
# Benign = 0 (Normal Traffic), Attack = 1 (everything else)
print("\nCreating binary labels...")
y_binary = y.apply(lambda x: 0 if x == 'Normal Traffic' else 1)
print(f"Binary label distribution:")
print(f"  Benign (0): {(y_binary == 0).sum()}")
print(f"  Attack (1): {(y_binary == 1).sum()}")

# 2.5 Create multiclass labels
print("\nEncoding multiclass labels...")
label_encoder = LabelEncoder()
y_multiclass = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
print(f"Number of classes: {len(class_names)}")
print(f"Classes: {list(class_names)}")

# 2.6 Min-Max Normalization to [0, 1]
print("\nApplying Min-Max normalization...")
scaler = MinMaxScaler(feature_range=(0, 1))
X_normalized = scaler.fit_transform(X)
print(f"Normalized features range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")

# 2.7 Train/Test Split (70:30 as per paper)
print("\nSplitting data into train/test (70:30)...")
X_train, X_test, y_train_binary, y_test_binary, y_train_multi, y_test_multi = train_test_split(
    X_normalized, y_binary, y_multiclass,
    test_size=0.3,
    random_state=42,
    stratify=y_multiclass
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# =============================================================================
# SECTION 3: SPARSE AUTOENCODER MODEL DEFINITION
# =============================================================================
print("\n[SECTION 3] SPARSE AUTOENCODER MODEL DEFINITION")
print("-" * 50)

# Architecture adapted for 52 features (proportional to paper's 79→68→64)
# Original: 79 → 68 → 64 → 68 → 79
# Adapted:  52 → 45 → 42 → 45 → 52
INPUT_DIM = X_train.shape[1]
HIDDEN_1 = 45   # First encoder layer
HIDDEN_2 = 42   # Bottleneck layer (compressed features)

# SAE hyperparameters as per paper
SPARSITY_TARGET = 0.05   # rho
SPARSITY_WEIGHT = 0.1    # beta
L2_WEIGHT = 0.0001       # lambda

print(f"SAE Architecture: {INPUT_DIM} -> {HIDDEN_1} -> {HIDDEN_2} -> {HIDDEN_1} -> {INPUT_DIM}")
print(f"Activation: ELU | Epochs: 50 | Batch Size: 256")
print(f"Sparsity target (rho): {SPARSITY_TARGET} | Sparsity weight (beta): {SPARSITY_WEIGHT}")
print(f"L2 regularization (lambda): {L2_WEIGHT}")


def kl_divergence(rho, rho_hat):
    """
    KL divergence for sparsity constraint.
    KL(rho || rho_hat) = rho*log(rho/rho_hat) + (1-rho)*log((1-rho)/(1-rho_hat))
    """
    epsilon = 1e-10
    rho_hat = K.clip(rho_hat, epsilon, 1 - epsilon)
    return rho * K.log(rho / rho_hat) + (1 - rho) * K.log((1 - rho) / (1 - rho_hat))


class SparseAutoencoder(Model):
    """
    Sparse Autoencoder with custom loss:
    Loss = MSE (reconstruction) + L2 (weight regularization) + KL-Divergence (sparsity)
    """

    def __init__(self, input_dim, hidden_1, hidden_2, sparsity_target, sparsity_weight, l2_weight):
        super(SparseAutoencoder, self).__init__()
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

        self.encoder_1 = layers.Dense(hidden_1, activation='elu',
                                      kernel_regularizer=regularizers.l2(l2_weight),
                                      name='encoder_1')
        self.encoder_2 = layers.Dense(hidden_2, activation='elu',
                                      kernel_regularizer=regularizers.l2(l2_weight),
                                      name='encoder_2_bottleneck')
        self.decoder_1 = layers.Dense(hidden_1, activation='elu',
                                      kernel_regularizer=regularizers.l2(l2_weight),
                                      name='decoder_1')
        self.decoder_2 = layers.Dense(input_dim, activation='linear',
                                      kernel_regularizer=regularizers.l2(l2_weight),
                                      name='decoder_2_output')

    def call(self, inputs, training=False):
        h1 = self.encoder_1(inputs)
        bottleneck = self.encoder_2(h1)
        h3 = self.decoder_1(bottleneck)
        reconstructed = self.decoder_2(h3)

        if training:
            rho_hat_1 = K.mean(h1, axis=0)
            rho_hat_2 = K.mean(bottleneck, axis=0)
            sparsity_loss_1 = K.sum(kl_divergence(self.sparsity_target, rho_hat_1))
            sparsity_loss_2 = K.sum(kl_divergence(self.sparsity_target, rho_hat_2))
            sparsity_loss = self.sparsity_weight * (sparsity_loss_1 + sparsity_loss_2)
            self.add_loss(sparsity_loss)

        return reconstructed

    def encode(self, inputs):
        """Extract compressed features from bottleneck layer."""
        h1 = self.encoder_1(inputs)
        return self.encoder_2(h1)


# Build the model
sae_model = SparseAutoencoder(
    input_dim=INPUT_DIM,
    hidden_1=HIDDEN_1,
    hidden_2=HIDDEN_2,
    sparsity_target=SPARSITY_TARGET,
    sparsity_weight=SPARSITY_WEIGHT,
    l2_weight=L2_WEIGHT
)

sae_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

_ = sae_model(X_train[:1])
print("\nSAE Model built successfully!")

# =============================================================================
# SECTION 4: SAE TRAINING
# =============================================================================
print("\n[SECTION 4] SAE TRAINING")
print("-" * 50)

EPOCHS = 50
BATCH_SIZE = 256

print(f"Training SAE for {EPOCHS} epochs with batch size {BATCH_SIZE}...")
print("Loss = MSE + L2 regularization + KL-divergence sparsity penalty")

start_time = time.time()

history = sae_model.fit(
    X_train, X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

training_time = time.time() - start_time
print(f"\nSAE Training completed in {training_time:.2f} seconds")
print(f"Final training loss: {history.history['loss'][-1]:.6f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")

# ---- SAE Training Loss Plot ----
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', color='royalblue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='coral', linewidth=2)
plt.title('SAE Training Progress - CICIDS2017')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE + L2 + KL)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_PATH + 'sae_training_progress_cicids2017.png', dpi=150)
plt.close()
print(f"\nSAE training loss plot saved.")

# =============================================================================
# SECTION 5: FEATURE COMPRESSION (BOTTLENECK EXTRACTION)
# =============================================================================
print("\n[SECTION 5] FEATURE COMPRESSION")
print("-" * 50)

print(f"Extracting {HIDDEN_2}-dimensional compressed features from bottleneck layer...")

X_train_compressed = sae_model.encode(X_train).numpy()
X_test_compressed = sae_model.encode(X_test).numpy()

print(f"Original feature dimension: {INPUT_DIM}")
print(f"Compressed feature dimension: {X_train_compressed.shape[1]}")
print(f"Compression ratio: {INPUT_DIM / X_train_compressed.shape[1]:.2f}x")
print(f"Training compressed shape: {X_train_compressed.shape}")
print(f"Testing compressed shape: {X_test_compressed.shape}")

# =============================================================================
# SECTION 6: CLASSIFIER TRAINING
# (RF-10 | XGBoost-100 | Soft Voting Ensemble — for Obj-4 Comparison)
# =============================================================================
print("\n[SECTION 6] CLASSIFIER TRAINING")
print("-" * 50)

N_ESTIMATORS_RF  = 10
N_ESTIMATORS_XGB = 100

# ---- 6.1 Random Forest (RF-10) — Base Paper Classifier ----
print("\n--- Training Random Forest RF-10 (Base Paper) ---")
rf_binary     = RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, random_state=42, n_jobs=-1)
rf_multiclass = RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, random_state=42, n_jobs=-1)

start_time = time.time()
rf_binary.fit(X_train_compressed, y_train_binary)
rf_binary_train_time = time.time() - start_time
print(f"RF Binary training time: {rf_binary_train_time:.2f} seconds")

start_time = time.time()
rf_multiclass.fit(X_train_compressed, y_train_multi)
rf_multiclass_train_time = time.time() - start_time
print(f"RF Multiclass training time: {rf_multiclass_train_time:.2f} seconds")

# ---- 6.2 XGBoost-100 — Extension Classifier ----
print("\n--- Training XGBoost-100 (Extension) ---")
xgb_binary     = XGBClassifier(n_estimators=N_ESTIMATORS_XGB, max_depth=6, learning_rate=0.1,
                                random_state=42, eval_metric='logloss', n_jobs=-1)
xgb_multiclass = XGBClassifier(n_estimators=N_ESTIMATORS_XGB, max_depth=6, learning_rate=0.1,
                                random_state=42, eval_metric='mlogloss', n_jobs=-1)

start_time = time.time()
xgb_binary.fit(X_train_compressed, y_train_binary)
xgb_binary_train_time = time.time() - start_time
print(f"XGBoost Binary training time: {xgb_binary_train_time:.2f} seconds")

start_time = time.time()
xgb_multiclass.fit(X_train_compressed, y_train_multi)
xgb_multiclass_train_time = time.time() - start_time
print(f"XGBoost Multiclass training time: {xgb_multiclass_train_time:.2f} seconds")

# ---- 6.3 Voting Ensemble (RF + XGBoost, Soft Voting) ----
print("\n--- Training Voting Ensemble (RF + XGBoost, Soft Voting) ---")
ensemble_binary = VotingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=N_ESTIMATORS_XGB, max_depth=6, learning_rate=0.1,
                              random_state=42, eval_metric='logloss', n_jobs=-1))
    ],
    voting='soft'
)

ensemble_multiclass = VotingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=N_ESTIMATORS_XGB, max_depth=6, learning_rate=0.1,
                              random_state=42, eval_metric='mlogloss', n_jobs=-1))
    ],
    voting='soft'
)

start_time = time.time()
ensemble_binary.fit(X_train_compressed, y_train_binary)
ensemble_binary_train_time = time.time() - start_time
print(f"Ensemble Binary training time: {ensemble_binary_train_time:.2f} seconds")

start_time = time.time()
ensemble_multiclass.fit(X_train_compressed, y_train_multi)
ensemble_multiclass_train_time = time.time() - start_time
print(f"Ensemble Multiclass training time: {ensemble_multiclass_train_time:.2f} seconds")

# =============================================================================
# SECTION 7: EVALUATION
# =============================================================================
print("\n[SECTION 7] EVALUATION")
print("-" * 50)


def evaluate_classifier(name, y_true, y_pred, task='binary'):
    """Compute and print Accuracy, Precision, Recall, F1-Score."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\n{name} -- {task.upper()} CLASSIFICATION")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    return acc, prec, rec, f1


# ---- RF-10 Results ----
print("\n" + "=" * 60)
print("RANDOM FOREST (RF-10) RESULTS")
print("=" * 60)

y_pred_rf_binary = rf_binary.predict(X_test_compressed)
y_pred_rf_multi  = rf_multiclass.predict(X_test_compressed)

rf_bin_acc,  rf_bin_prec,  rf_bin_rec,  rf_bin_f1  = evaluate_classifier(
    "RF-10", y_test_binary, y_pred_rf_binary, task='binary')
rf_mul_acc,  rf_mul_prec,  rf_mul_rec,  rf_mul_f1  = evaluate_classifier(
    "RF-10", y_test_multi,  y_pred_rf_multi,  task='multiclass')

# ---- XGBoost-100 Results ----
print("\n" + "=" * 60)
print("XGBOOST-100 RESULTS")
print("=" * 60)

y_pred_xgb_binary = xgb_binary.predict(X_test_compressed)
y_pred_xgb_multi  = xgb_multiclass.predict(X_test_compressed)

xgb_bin_acc, xgb_bin_prec, xgb_bin_rec, xgb_bin_f1 = evaluate_classifier(
    "XGBoost-100", y_test_binary, y_pred_xgb_binary, task='binary')
xgb_mul_acc, xgb_mul_prec, xgb_mul_rec, xgb_mul_f1 = evaluate_classifier(
    "XGBoost-100", y_test_multi,  y_pred_xgb_multi,  task='multiclass')

# ---- Voting Ensemble Results ----
print("\n" + "=" * 60)
print("VOTING ENSEMBLE (RF + XGBoost, Soft Voting) RESULTS")
print("=" * 60)

y_pred_ens_binary = ensemble_binary.predict(X_test_compressed)
y_pred_ens_multi  = ensemble_multiclass.predict(X_test_compressed)

ens_bin_acc, ens_bin_prec, ens_bin_rec, ens_bin_f1 = evaluate_classifier(
    "Voting Ensemble", y_test_binary, y_pred_ens_binary, task='binary')
ens_mul_acc, ens_mul_prec, ens_mul_rec, ens_mul_f1 = evaluate_classifier(
    "Voting Ensemble", y_test_multi,  y_pred_ens_multi,  task='multiclass')

# ---- Classification Reports ----
print("\n\nClassification Report -- Voting Ensemble Binary:")
print(classification_report(y_test_binary, y_pred_ens_binary,
                             target_names=['Normal Traffic', 'Attack']))

print("\nClassification Report -- Voting Ensemble Multiclass:")
print(classification_report(y_test_multi, y_pred_ens_multi,
                             target_names=class_names))

# =============================================================================
# SECTION 8: CONFUSION MATRICES
# =============================================================================
print("\n[SECTION 8] CONFUSION MATRICES")
print("-" * 50)

# ---- Binary Confusion Matrix (Voting Ensemble) ----
cm_binary = confusion_matrix(y_test_binary, y_pred_ens_binary)
print("\nVoting Ensemble Binary Confusion Matrix:")
print(f"  True Normal  : {cm_binary[0,0]} | False Attack : {cm_binary[0,1]}")
print(f"  False Normal : {cm_binary[1,0]} | True Attack  : {cm_binary[1,1]}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Normal Traffic', 'Attack'],
            yticklabels=['Normal Traffic', 'Attack'])
plt.title('Voting Ensemble (RF+XGBoost) -- Binary Confusion Matrix\nCICIDS2017')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(RESULTS_PATH + 'confusion_matrix_binary_ensemble.png', dpi=150)
plt.close()
print("Binary confusion matrix saved.")

# ---- Multiclass Confusion Matrix (Voting Ensemble) ----
cm_multi = confusion_matrix(y_test_multi, y_pred_ens_multi)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Voting Ensemble (RF+XGBoost) -- Multiclass Confusion Matrix\nCICIDS2017')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULTS_PATH + 'confusion_matrix_multiclass_ensemble.png', dpi=150)
plt.close()
print("Multiclass confusion matrix saved.")

# =============================================================================
# SECTION 9: PERFORMANCE COMPARISON BAR CHARTS (Obj-4)
# =============================================================================
print("\n[SECTION 9] PERFORMANCE COMPARISON CHARTS")
print("-" * 50)

metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
models = ['RF-10', 'XGBoost-100', 'Voting Ensemble']
colors = ['steelblue', 'coral', 'mediumseagreen']

x     = np.arange(len(metrics_labels))
width = 0.25

# ---- Binary Performance Chart ----
binary_values = [
    [rf_bin_acc,  rf_bin_prec,  rf_bin_rec,  rf_bin_f1],
    [xgb_bin_acc, xgb_bin_prec, xgb_bin_rec, xgb_bin_f1],
    [ens_bin_acc, ens_bin_prec, ens_bin_rec, ens_bin_f1],
]

fig, ax = plt.subplots(figsize=(11, 6))
for i, (model, vals, color) in enumerate(zip(models, binary_values, colors)):
    bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

ax.set_title('Performance Comparison -- Binary Classification (CICIDS2017)', fontsize=13)
ax.set_ylabel('Score')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.08)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_PATH + 'performance_comparison_binary_cicids2017.png', dpi=150)
plt.close()
print("Binary performance comparison chart saved.")

# ---- Multiclass Performance Chart ----
multi_values = [
    [rf_mul_acc,  rf_mul_prec,  rf_mul_rec,  rf_mul_f1],
    [xgb_mul_acc, xgb_mul_prec, xgb_mul_rec, xgb_mul_f1],
    [ens_mul_acc, ens_mul_prec, ens_mul_rec, ens_mul_f1],
]

fig, ax = plt.subplots(figsize=(11, 6))
for i, (model, vals, color) in enumerate(zip(models, multi_values, colors)):
    bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

ax.set_title('Performance Comparison -- Multiclass Classification (CICIDS2017)', fontsize=13)
ax.set_ylabel('Score')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.08)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_PATH + 'performance_comparison_multiclass_cicids2017.png', dpi=150)
plt.close()
print("Multiclass performance comparison chart saved.")

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY -- CICIDS2017")
print("=" * 70)

print(f"""
Dataset: CICIDS2017
  Total samples   : {len(df)}
  Features        : {INPUT_DIM}
  Compressed (SAE): {HIDDEN_2}
  Classes         : {len(class_names)} {list(class_names)}

SAE Architecture: {INPUT_DIM} -> {HIDDEN_1} -> {HIDDEN_2} -> {HIDDEN_1} -> {INPUT_DIM}
  Activation: ELU | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}
  Loss: MSE + L2 (lambda={L2_WEIGHT}) + KL-Divergence (beta={SPARSITY_WEIGHT}, rho={SPARSITY_TARGET})

BINARY CLASSIFICATION (Normal Traffic vs Attack):
+---------------------+-----------+--------------+------------------+
| Metric              | RF-10     | XGBoost-100  | Voting Ensemble  |
+---------------------+-----------+--------------+------------------+
| Accuracy            | {rf_bin_acc:.4f}    | {xgb_bin_acc:.4f}       | {ens_bin_acc:.4f}           |
| Precision           | {rf_bin_prec:.4f}    | {xgb_bin_prec:.4f}       | {ens_bin_prec:.4f}           |
| Recall              | {rf_bin_rec:.4f}    | {xgb_bin_rec:.4f}       | {ens_bin_rec:.4f}           |
| F1-Score            | {rf_bin_f1:.4f}    | {xgb_bin_f1:.4f}       | {ens_bin_f1:.4f}           |
+---------------------+-----------+--------------+------------------+

MULTICLASS CLASSIFICATION ({len(class_names)} classes):
+---------------------+-----------+--------------+------------------+
| Metric              | RF-10     | XGBoost-100  | Voting Ensemble  |
+---------------------+-----------+--------------+------------------+
| Accuracy            | {rf_mul_acc:.4f}    | {xgb_mul_acc:.4f}       | {ens_mul_acc:.4f}           |
| Precision           | {rf_mul_prec:.4f}    | {xgb_mul_prec:.4f}       | {ens_mul_prec:.4f}           |
| Recall              | {rf_mul_rec:.4f}    | {xgb_mul_rec:.4f}       | {ens_mul_rec:.4f}           |
| F1-Score            | {rf_mul_f1:.4f}    | {xgb_mul_f1:.4f}       | {ens_mul_f1:.4f}           |
+---------------------+-----------+--------------+------------------+
""")

print("Files generated in results/cicids2017/:")
print("  - sae_training_progress_cicids2017.png")
print("  - confusion_matrix_binary_ensemble.png")
print("  - confusion_matrix_multiclass_ensemble.png")
print("  - performance_comparison_binary_cicids2017.png")
print("  - performance_comparison_multiclass_cicids2017.png")
print("  - training_output_cicids2017.txt")
print("\n" + "=" * 70)
print("Implementation completed successfully!")
print("=" * 70)

tee.close()
