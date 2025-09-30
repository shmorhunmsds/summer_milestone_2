#!/usr/bin/env python3
"""
Advanced GPU-Accelerated Models for NFL Collision Injury Prediction
Leverages NVIDIA GPU for enhanced performance at extreme class imbalances
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Standard ML
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, recall_score, precision_score
)
from sklearn.preprocessing import StandardScaler

# GPU-Accelerated Libraries
try:
    import cudf
    import cupy as cp
    from cuml import LogisticRegression as cuLogisticRegression
    from cuml import RandomForestClassifier as cuRandomForestClassifier
    from cuml import SVC as cuSVC
    from cuml.ensemble import RandomForestClassifier as cuRF
    RAPIDS_AVAILABLE = True
    print("✅ RAPIDS cuML available - GPU acceleration enabled!")
except ImportError:
    RAPIDS_AVAILABLE = False
    print("⚠️ RAPIDS not available, will use CPU alternatives")

# XGBoost with GPU support
import xgboost as xgb

# LightGBM with GPU support
import lightgbm as lgb

# CatBoost with GPU support
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available")

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Advanced Imbalance Handling
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

# Hyperparameter Optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available for hyperparameter optimization")


class AdvancedGPUModels:
    """Advanced modeling suite with GPU acceleration and imbalance handling"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PyTorch using device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def load_data(self):
        """Load collision data"""
        print("\nLoading collision data...")

        # Load full dataset
        full_df = pd.read_csv('capstone_file/punt_analytics/full_collision_dataset.csv', low_memory=False)

        # Separate features and target
        metadata_cols = ['seasonyear', 'season_year', 'gamekey', 'playid',
                        'injured_player', 'partner_player', 'impact_type',
                        'player_activity', 'partner_activity', 'friendly_fire',
                        'is_injury']

        feature_cols = [col for col in full_df.columns if col not in metadata_cols]

        X = full_df[feature_cols].fillna(0)  # Simple imputation for now
        y = full_df['is_injury'].values

        print(f"  Features: {X.shape}")
        print(f"  Class distribution - Injuries: {sum(y)}, Normal: {len(y)-sum(y)}")

        return X, y, feature_cols

    def create_balanced_split(self, X, y, ratio=100, test_size=0.2):
        """Create train/test split with specified imbalance ratio"""

        # Separate classes
        injury_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]

        # Calculate samples needed
        n_injuries = len(injury_idx)
        n_normal_needed = min(n_injuries * ratio, len(normal_idx))

        # Sample normal cases
        normal_sample_idx = np.random.choice(normal_idx, n_normal_needed, replace=False)

        # Combine indices
        all_idx = np.concatenate([injury_idx, normal_sample_idx])
        np.random.shuffle(all_idx)

        X_balanced = X.iloc[all_idx]
        y_balanced = y[all_idx]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=test_size,
            random_state=self.random_state, stratify=y_balanced
        )

        print(f"\nDataset with 1:{ratio} ratio created")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def train_xgboost_gpu(self, X_train, X_test, y_train, y_test):
        """XGBoost with GPU acceleration and advanced tuning"""
        print("\n🚀 Training XGBoost with GPU...")

        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Advanced parameters for imbalanced data
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'aucpr'],
            'tree_method': 'gpu_hist',  # GPU acceleration
            'predictor': 'gpu_predictor',
            'gpu_id': 0,
            'scale_pos_weight': scale_pos_weight,
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': self.random_state
        }

        # Train with early stopping
        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        model = xgb.train(params, dtrain, num_boost_round=500,
                         evals=evallist, early_stopping_rounds=50,
                         verbose_eval=False)

        # Predictions
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Evaluate
        auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)

        print(f"  AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

        return model, {'auc': auc, 'recall': recall, 'precision': precision}

    def train_lightgbm_gpu(self, X_train, X_test, y_train, y_test):
        """LightGBM with GPU acceleration"""
        print("\n🚀 Training LightGBM with GPU...")

        # Calculate scale_pos_weight
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Parameters optimized for imbalanced data
        params = {
            'objective': 'binary',
            'metric': ['auc', 'average_precision'],
            'boosting_type': 'gbdt',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'scale_pos_weight': scale_pos_weight,
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'random_state': self.random_state,
            'verbose': -1
        }

        # Train
        model = lgb.train(params, train_data, valid_sets=[valid_data],
                         num_boost_round=500, callbacks=[lgb.early_stopping(50)])

        # Predictions
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Evaluate
        auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)

        print(f"  AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

        return model, {'auc': auc, 'recall': recall, 'precision': precision}

    def train_catboost_gpu(self, X_train, X_test, y_train, y_test):
        """CatBoost with GPU acceleration"""
        if not CATBOOST_AVAILABLE:
            print("⚠️ CatBoost not available")
            return None, None

        print("\n🚀 Training CatBoost with GPU...")

        # Calculate scale_pos_weight
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

        # Initialize model
        model = cb.CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            loss_function='Logloss',
            eval_metric='AUC',
            scale_pos_weight=scale_pos_weight,
            task_type='GPU',
            devices='0',
            random_state=self.random_state,
            verbose=False,
            early_stopping_rounds=50
        )

        # Train
        model.fit(X_train, y_train, eval_set=(X_test, y_test))

        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Evaluate
        auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)

        print(f"  AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

        return model, {'auc': auc, 'recall': recall, 'precision': precision}

    def train_rapids_models(self, X_train, X_test, y_train, y_test):
        """Train models using RAPIDS cuML (GPU-accelerated sklearn)"""
        if not RAPIDS_AVAILABLE:
            print("⚠️ RAPIDS not available")
            return None

        print("\n🚀 Training RAPIDS cuML models...")

        # Convert to GPU dataframes
        X_train_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
        X_test_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(X_test))
        y_train_gpu = cudf.Series(y_train)
        y_test_gpu = cudf.Series(y_test)

        results = {}

        # 1. GPU Logistic Regression
        print("  Training GPU Logistic Regression...")
        lr_gpu = cuLogisticRegression(max_iter=1000, penalty='l2')
        lr_gpu.fit(X_train_gpu, y_train_gpu)

        y_pred_lr = lr_gpu.predict(X_test_gpu).to_numpy()
        y_pred_proba_lr = lr_gpu.predict_proba(X_test_gpu)[1].to_numpy()

        results['LogisticRegression'] = {
            'auc': roc_auc_score(y_test, y_pred_proba_lr),
            'recall': recall_score(y_test, y_pred_lr),
            'precision': precision_score(y_test, y_pred_lr, zero_division=0)
        }

        # 2. GPU Random Forest
        print("  Training GPU Random Forest...")
        rf_gpu = cuRF(n_estimators=100, max_depth=10, random_state=self.random_state)
        rf_gpu.fit(X_train_gpu, y_train_gpu)

        y_pred_rf = rf_gpu.predict(X_test_gpu).to_numpy()
        y_pred_proba_rf = rf_gpu.predict_proba(X_test_gpu)[1].to_numpy()

        results['RandomForest'] = {
            'auc': roc_auc_score(y_test, y_pred_proba_rf),
            'recall': recall_score(y_test, y_pred_rf),
            'precision': precision_score(y_test, y_pred_rf, zero_division=0)
        }

        # 3. GPU SVM
        print("  Training GPU SVM...")
        svm_gpu = cuSVC(kernel='rbf', probability=True)
        svm_gpu.fit(X_train_gpu, y_train_gpu)

        y_pred_svm = svm_gpu.predict(X_test_gpu).to_numpy()
        y_pred_proba_svm = svm_gpu.predict_proba(X_test_gpu)[1].to_numpy()

        results['SVM'] = {
            'auc': roc_auc_score(y_test, y_pred_proba_svm),
            'recall': recall_score(y_test, y_pred_svm),
            'precision': precision_score(y_test, y_pred_svm, zero_division=0)
        }

        print("\n  RAPIDS Results:")
        for name, metrics in results.items():
            print(f"    {name}: AUC={metrics['auc']:.4f}, Recall={metrics['recall']:.4f}")

        return results


class FocalLossNN(nn.Module):
    """Neural Network with Focal Loss for extreme imbalance"""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super(FocalLossNN, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))


class FocalLoss(nn.Module):
    """Focal Loss for addressing extreme class imbalance"""

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        # Binary focal loss
        p = predictions
        ce_loss = -(targets * torch.log(p + 1e-8) + (1 - targets) * torch.log(1 - p + 1e-8))
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()


def train_deep_learning_model(X_train, X_test, y_train, y_test, device='cuda'):
    """Train neural network with focal loss"""
    print("\n🧠 Training Neural Network with Focal Loss...")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Calculate class weights for sampling
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[int(y)] for y in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model
    input_dim = X_train.shape[1]
    model = FocalLossNN(input_dim, hidden_dims=[256, 128, 64, 32], dropout=0.3)
    model = model.to(device)

    # Loss and optimizer
    criterion = FocalLoss(alpha=0.75, gamma=2)  # Higher alpha for minority class
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10)

    # Training
    best_auc = 0
    patience = 20
    patience_counter = 0

    for epoch in range(200):
        # Train
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_targets = []

            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X).squeeze()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.numpy())

            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)

            auc = roc_auc_score(all_targets, all_preds)

            if auc > best_auc:
                best_auc = auc
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            scheduler.step(auc)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, AUC={auc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        y_pred_proba = model(X_test_tensor).squeeze().cpu().numpy()
        y_pred = (y_pred_proba > 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)

    print(f"  Final - AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

    return model, {'auc': auc, 'recall': recall, 'precision': precision}


class AdvancedEnsembles:
    """Advanced ensemble techniques for imbalanced data"""

    @staticmethod
    def train_balanced_random_forest(X_train, X_test, y_train, y_test):
        """Balanced Random Forest - automatically handles imbalance"""
        print("\n🌲 Training Balanced Random Forest...")

        model = BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)

        print(f"  AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

        return model, {'auc': auc, 'recall': recall, 'precision': precision}

    @staticmethod
    def train_easy_ensemble(X_train, X_test, y_train, y_test):
        """Easy Ensemble - multiple balanced subsets"""
        print("\n👥 Training Easy Ensemble...")

        model = EasyEnsembleClassifier(
            n_estimators=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)

        print(f"  AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

        return model, {'auc': auc, 'recall': recall, 'precision': precision}


class SMOTETechniques:
    """Advanced SMOTE variants for synthetic sample generation"""

    @staticmethod
    def apply_smote_variants(X_train, y_train, variant='borderline'):
        """Apply different SMOTE variants"""
        print(f"\n🔬 Applying {variant.upper()} SMOTE...")

        if variant == 'regular':
            sampler = SMOTE(random_state=42)
        elif variant == 'borderline':
            sampler = BorderlineSMOTE(random_state=42, kind='borderline-1')
        elif variant == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif variant == 'smote_enn':
            sampler = SMOTEENN(random_state=42)
        elif variant == 'smote_tomek':
            sampler = SMOTETomek(random_state=42)
        else:
            raise ValueError(f"Unknown SMOTE variant: {variant}")

        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

        print(f"  Original: {len(X_train)} samples")
        print(f"  Resampled: {len(X_resampled)} samples")
        print(f"  New ratio: {sum(y_resampled==0)}/{sum(y_resampled==1):.1f}:1")

        return X_resampled, y_resampled


def run_advanced_model_suite(ratio=100):
    """Run complete suite of advanced models"""
    print("\n" + "="*60)
    print(f"🚀 ADVANCED GPU MODEL SUITE - Testing at 1:{ratio} ratio")
    print("="*60)

    # Initialize
    gpu_models = AdvancedGPUModels()

    # Load data
    X, y, feature_names = gpu_models.load_data()

    # Create balanced dataset
    X_train, X_test, y_train, y_test = gpu_models.create_balanced_split(X, y, ratio=ratio)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # 1. XGBoost GPU
    xgb_model, xgb_results = gpu_models.train_xgboost_gpu(
        X_train, X_test, y_train, y_test
    )
    results['XGBoost-GPU'] = xgb_results

    # 2. LightGBM GPU
    lgb_model, lgb_results = gpu_models.train_lightgbm_gpu(
        X_train, X_test, y_train, y_test
    )
    results['LightGBM-GPU'] = lgb_results

    # 3. CatBoost GPU
    if CATBOOST_AVAILABLE:
        cb_model, cb_results = gpu_models.train_catboost_gpu(
            X_train, X_test, y_train, y_test
        )
        if cb_results:
            results['CatBoost-GPU'] = cb_results

    # 4. RAPIDS cuML models
    if RAPIDS_AVAILABLE:
        rapids_results = gpu_models.train_rapids_models(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        if rapids_results:
            for name, metrics in rapids_results.items():
                results[f'RAPIDS-{name}'] = metrics

    # 5. Deep Learning with Focal Loss
    if torch.cuda.is_available():
        dl_model, dl_results = train_deep_learning_model(
            pd.DataFrame(X_train_scaled), pd.DataFrame(X_test_scaled),
            y_train, y_test
        )
        results['NeuralNet-FocalLoss'] = dl_results

    # 6. Advanced Ensembles
    brf_model, brf_results = AdvancedEnsembles.train_balanced_random_forest(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    results['BalancedRandomForest'] = brf_results

    ee_model, ee_results = AdvancedEnsembles.train_easy_ensemble(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    results['EasyEnsemble'] = ee_results

    # 7. SMOTE + XGBoost
    print("\n🧪 Testing SMOTE Variants + XGBoost...")
    for variant in ['borderline', 'adasyn']:
        X_resampled, y_resampled = SMOTETechniques.apply_smote_variants(
            X_train, y_train, variant=variant
        )

        # Train XGBoost on resampled data
        dtrain = xgb.DMatrix(X_resampled, label=y_resampled)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'gpu_hist',
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }

        model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
        y_pred_proba = model.predict(dtest)

        auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, (y_pred_proba > 0.5).astype(int))

        results[f'SMOTE-{variant}+XGBoost'] = {
            'auc': auc,
            'recall': recall,
            'precision': precision_score(y_test, (y_pred_proba > 0.5).astype(int), zero_division=0)
        }

    # Print summary
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)

    # Sort by AUC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)

    print(f"\n{'Model':<30} {'AUC':>8} {'Recall':>8} {'Precision':>10}")
    print("-" * 58)

    for name, metrics in sorted_results:
        print(f"{name:<30} {metrics['auc']:>8.4f} {metrics['recall']:>8.4f} {metrics['precision']:>10.4f}")

    # Best model
    best_model = sorted_results[0]
    print(f"\n🏆 BEST MODEL: {best_model[0]}")
    print(f"   AUC: {best_model[1]['auc']:.4f}")
    print(f"   Recall: {best_model[1]['recall']:.4f}")
    print(f"   Precision: {best_model[1]['precision']:.4f}")

    return results


if __name__ == "__main__":
    # Test at different ratios
    ratios_to_test = [50, 100, 200, 500]

    all_results = {}
    for ratio in ratios_to_test:
        print(f"\n{'#'*70}")
        print(f"Testing at 1:{ratio} imbalance ratio")
        print(f"{'#'*70}")

        results = run_advanced_model_suite(ratio=ratio)
        all_results[ratio] = results

    # Save results
    import pickle
    with open('capstone_file/punt_analytics/advanced_model_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print("\n✅ Advanced model evaluation complete!")
    print("📁 Results saved to capstone_file/punt_analytics/advanced_model_results.pkl")