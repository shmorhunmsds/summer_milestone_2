#!/usr/bin/env python3
"""
Hyperparameter Optimization for NFL Collision Models
Uses Optuna with GPU acceleration to find optimal parameters
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, make_scorer
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    print("⚠️ Optuna not available. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False
    exit()


class HyperparameterOptimizer:
    """Automated hyperparameter optimization for imbalanced data"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_params = {}
        self.study_results = {}

    def load_data(self, ratio=100):
        """Load and prepare data"""
        print(f"Loading data with 1:{ratio} ratio...")

        # Load full dataset
        full_df = pd.read_csv('capstone_file/punt_analytics/full_collision_dataset.csv', low_memory=False)

        # Separate features and target
        metadata_cols = ['seasonyear', 'season_year', 'gamekey', 'playid',
                        'injured_player', 'partner_player', 'impact_type',
                        'player_activity', 'partner_activity', 'friendly_fire',
                        'is_injury']

        feature_cols = [col for col in full_df.columns if col not in metadata_cols]

        # Get injury and normal samples
        injury_df = full_df[full_df['is_injury'] == 1]
        normal_df = full_df[full_df['is_injury'] == 0]

        # Sample to create desired ratio
        n_injuries = len(injury_df)
        n_normal = min(n_injuries * ratio, len(normal_df))

        normal_sample = normal_df.sample(n=n_normal, random_state=self.random_state)

        # Combine
        balanced_df = pd.concat([injury_df, normal_sample], ignore_index=True)

        X = balanced_df[feature_cols].fillna(0)
        y = balanced_df['is_injury'].values

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f"  Dataset: {len(X)} samples ({sum(y)} injuries)")

        return X_scaled, y

    def optimize_xgboost(self, X, y, n_trials=100):
        """Optimize XGBoost hyperparameters"""
        print("\n🎯 Optimizing XGBoost with Optuna...")

        def objective(trial):
            # Suggest hyperparameters
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'gpu_hist',  # GPU acceleration
                'predictor': 'gpu_predictor',
                'gpu_id': 0,
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 50),
                'random_state': self.random_state
            }

            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = []

            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)

                model = xgb.train(
                    params, dtrain,
                    num_boost_round=params['n_estimators'],
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=50,
                    verbose_eval=False
                )

                # Predict
                y_pred = model.predict(dval)

                # Calculate metrics
                auc = roc_auc_score(y_val, y_pred)
                recall = recall_score(y_val, (y_pred > 0.5).astype(int))

                # Combined metric (prioritize both AUC and recall)
                combined_score = 0.7 * auc + 0.3 * recall
                scores.append(combined_score)

                # Prune unpromising trials
                trial.report(np.mean(scores), len(scores) - 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Store results
        self.best_params['xgboost'] = study.best_params
        self.study_results['xgboost'] = study

        print(f"\n✅ Best XGBoost params found:")
        print(f"   Score: {study.best_value:.4f}")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")

        return study.best_params

    def optimize_lightgbm(self, X, y, n_trials=100):
        """Optimize LightGBM hyperparameters"""
        print("\n🎯 Optimizing LightGBM with Optuna...")

        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 50),
                'random_state': self.random_state,
                'verbose': -1
            }

            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = []

            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val)

                model = lgb.train(
                    params, train_data,
                    valid_sets=[val_data],
                    num_boost_round=params['n_estimators'],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )

                # Predict
                y_pred = model.predict(X_val, num_iteration=model.best_iteration)

                # Metrics
                auc = roc_auc_score(y_val, y_pred)
                recall = recall_score(y_val, (y_pred > 0.5).astype(int))

                combined_score = 0.7 * auc + 0.3 * recall
                scores.append(combined_score)

                trial.report(np.mean(scores), len(scores) - 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Store results
        self.best_params['lightgbm'] = study.best_params
        self.study_results['lightgbm'] = study

        print(f"\n✅ Best LightGBM params found:")
        print(f"   Score: {study.best_value:.4f}")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")

        return study.best_params

    def optimize_neural_network(self, X, y, n_trials=50):
        """Optimize neural network architecture and hyperparameters"""
        print("\n🎯 Optimizing Neural Network with Optuna...")

        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def create_model(trial, input_dim):
            # Suggest architecture
            n_layers = trial.suggest_int('n_layers', 2, 5)
            layers = []
            prev_dim = input_dim

            for i in range(n_layers):
                hidden_dim = trial.suggest_int(f'n_units_l{i}', 32, 512)
                dropout = trial.suggest_float(f'dropout_l{i}', 0.1, 0.5)

                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())

            return nn.Sequential(*layers)

        def objective(trial):
            # Model parameters
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

            # Focal loss parameters
            focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.9)
            focal_gamma = trial.suggest_float('focal_gamma', 0.5, 5.0)

            # Cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scores = []

            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Convert to tensors
                X_train_t = torch.FloatTensor(X_train).to(device)
                y_train_t = torch.FloatTensor(y_train).to(device)
                X_val_t = torch.FloatTensor(X_val).to(device)
                y_val_t = torch.FloatTensor(y_val).to(device)

                # Create model
                model = create_model(trial, X.shape[1]).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                # Focal loss
                def focal_loss(pred, target):
                    ce_loss = -(target * torch.log(pred + 1e-8) +
                               (1 - target) * torch.log(1 - pred + 1e-8))
                    p_t = pred * target + (1 - pred) * (1 - target)
                    loss = ce_loss * ((1 - p_t) ** focal_gamma)
                    alpha_t = focal_alpha * target + (1 - focal_alpha) * (1 - target)
                    return (alpha_t * loss).mean()

                # Train
                dataset = TensorDataset(X_train_t, y_train_t)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                for epoch in range(50):
                    model.train()
                    for batch_X, batch_y in loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X).squeeze()
                        loss = focal_loss(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                # Evaluate
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_val_t).squeeze().cpu().numpy()

                auc = roc_auc_score(y_val, y_pred)
                recall = recall_score(y_val, (y_pred > 0.5).astype(int))

                combined_score = 0.7 * auc + 0.3 * recall
                scores.append(combined_score)

                trial.report(np.mean(scores), len(scores) - 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Store results
        self.best_params['neural_network'] = study.best_params
        self.study_results['neural_network'] = study

        print(f"\n✅ Best Neural Network params found:")
        print(f"   Score: {study.best_value:.4f}")
        for key, value in study.best_params.items():
            if not key.startswith('n_units_') and not key.startswith('dropout_'):
                print(f"   {key}: {value}")

        return study.best_params

    def train_optimized_models(self, X, y, test_size=0.2):
        """Train models with optimized hyperparameters"""
        print("\n" + "="*60)
        print("🏆 Training Models with Optimized Hyperparameters")
        print("="*60)

        from sklearn.model_selection import train_test_split

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        results = {}

        # 1. Optimized XGBoost
        if 'xgboost' in self.best_params:
            print("\n📊 Training Optimized XGBoost...")
            params = self.best_params['xgboost'].copy()
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                'gpu_id': 0,
                'random_state': self.random_state
            })

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            model = xgb.train(
                params, dtrain,
                num_boost_round=params.get('n_estimators', 500),
                evals=[(dtest, 'eval')],
                early_stopping_rounds=50,
                verbose_eval=False
            )

            y_pred_proba = model.predict(dtest)
            y_pred = (y_pred_proba > 0.5).astype(int)

            results['XGBoost-Optimized'] = {
                'auc': roc_auc_score(y_test, y_pred_proba),
                'recall': recall_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'model': model,
                'params': params
            }

        # 2. Optimized LightGBM
        if 'lightgbm' in self.best_params:
            print("\n📊 Training Optimized LightGBM...")
            params = self.best_params['lightgbm'].copy()
            params.update({
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'device': 'gpu',
                'random_state': self.random_state,
                'verbose': -1
            })

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_test, label=y_test)

            model = lgb.train(
                params, train_data,
                valid_sets=[val_data],
                num_boost_round=params.get('n_estimators', 500),
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
            y_pred = (y_pred_proba > 0.5).astype(int)

            results['LightGBM-Optimized'] = {
                'auc': roc_auc_score(y_test, y_pred_proba),
                'recall': recall_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'model': model,
                'params': params
            }

        # Print results
        print("\n" + "="*60)
        print("📊 OPTIMIZED MODEL RESULTS")
        print("="*60)

        print(f"\n{'Model':<25} {'AUC':>8} {'Recall':>8} {'Precision':>10}")
        print("-" * 53)

        for name, metrics in results.items():
            print(f"{name:<25} {metrics['auc']:>8.4f} {metrics['recall']:>8.4f} "
                  f"{metrics['precision']:>10.4f}")

        return results

    def plot_optimization_history(self):
        """Visualize optimization history"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(self.study_results), figsize=(5*len(self.study_results), 5))

        if len(self.study_results) == 1:
            axes = [axes]

        for idx, (name, study) in enumerate(self.study_results.items()):
            ax = axes[idx]

            # Get trial history
            trials = study.trials
            values = [t.value for t in trials if t.value is not None]
            best_values = np.maximum.accumulate(values)

            ax.plot(values, alpha=0.5, label='Trial Value')
            ax.plot(best_values, linewidth=2, label='Best Value', color='red')
            ax.set_xlabel('Trial')
            ax.set_ylabel('Objective Value')
            ax.set_title(f'{name.upper()} Optimization History')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Hyperparameter Optimization Progress', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('capstone_file/punt_analytics/optimization_history.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n📊 Optimization history saved to capstone_file/punt_analytics/optimization_history.png")


def run_hyperparameter_optimization(ratios=[100], n_trials=50):
    """Run complete hyperparameter optimization pipeline"""
    print("="*70)
    print("🚀 HYPERPARAMETER OPTIMIZATION PIPELINE")
    print("="*70)

    optimizer = HyperparameterOptimizer()
    all_results = {}

    for ratio in ratios:
        print(f"\n{'='*60}")
        print(f"Optimizing at 1:{ratio} ratio")
        print(f"{'='*60}")

        # Load data
        X, y = optimizer.load_data(ratio=ratio)

        # Optimize each model
        optimizer.optimize_xgboost(X, y, n_trials=n_trials)
        optimizer.optimize_lightgbm(X, y, n_trials=n_trials)

        # Optional: Optimize neural network (slower)
        # optimizer.optimize_neural_network(X, y, n_trials=n_trials//2)

        # Train with optimized parameters
        results = optimizer.train_optimized_models(X, y)
        all_results[ratio] = results

        # Plot optimization history
        optimizer.plot_optimization_history()

    # Save results
    import pickle
    import json

    # Save models and results
    with open('capstone_file/punt_analytics/optimized_models.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # Save best parameters
    best_params_json = {}
    for model_name, params in optimizer.best_params.items():
        best_params_json[model_name] = {k: float(v) if isinstance(v, (np.integer, np.floating))
                                        else v for k, v in params.items()}

    with open('capstone_file/punt_analytics/best_hyperparameters.json', 'w') as f:
        json.dump(best_params_json, f, indent=2)

    print("\n✅ Hyperparameter optimization complete!")
    print("📁 Results saved to capstone_file/punt_analytics/")
    print("\nBest hyperparameters saved to best_hyperparameters.json")

    return optimizer, all_results


if __name__ == "__main__":
    # Run optimization
    optimizer, results = run_hyperparameter_optimization(
        ratios=[100, 500],  # Test at different imbalance ratios
        n_trials=30  # Reduce for faster testing, increase for better optimization
    )

    print("\n🎯 Optimization complete! Use the best parameters for production models.")