#!/usr/bin/env python3
"""
NFL Collision Model Validation with Progressive Ratios
Tests model performance as class imbalance increases
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')


class ProgressiveModelValidator:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=random_state, max_iter=1000, class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=random_state, class_weight='balanced', n_estimators=100
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=random_state, n_estimators=100
            ),
            'SVM (RBF)': SVC(
                random_state=random_state, probability=True, class_weight='balanced'
            )
        }

    def load_collision_data(self):
        """Load the collision feature datasets"""
        print("Loading collision data...")

        # Try to load from the new full dataset first
        try:
            full_df = pd.read_csv('capstone_file/punt_analytics/full_collision_dataset.csv', low_memory=False)
            print(f"  Loaded full dataset: {len(full_df):,} total collisions")

            # Split into injury and normal
            injury_df = full_df[full_df['is_injury'] == 1].copy()
            normal_df = full_df[full_df['is_injury'] == 0].copy()

            print(f"  Injury collisions: {len(injury_df):,}")
            print(f"  Non-injury collisions: {len(normal_df):,}")
            print(f"  Actual ratio in data: 1:{int(len(normal_df)/len(injury_df))}")

        except FileNotFoundError:
            # Fall back to original files
            print("  Full dataset not found, loading original files...")
            injury_df = pd.read_csv('scripts/punt_analytics/injury_collision_features.csv')
            normal_df = pd.read_csv('scripts/punt_analytics/normal_collision_features.csv')

            print(f"  Loaded {len(injury_df)} injury collisions")
            print(f"  Loaded {len(normal_df)} normal collisions")

        return injury_df, normal_df

    def prepare_features(self, df):
        """Extract and clean features from dataframe"""
        # Identify feature columns (excluding metadata)
        metadata_cols = ['seasonyear', 'season_year', 'gamekey', 'playid',
                        'injured_player', 'partner_player', 'impact_type',
                        'player_activity', 'partner_activity', 'friendly_fire',
                        'is_injury']

        feature_cols = [col for col in df.columns if col not in metadata_cols]

        # Handle missing values
        imputer = KNNImputer(n_neighbors=5)
        X = pd.DataFrame(
            imputer.fit_transform(df[feature_cols]),
            columns=feature_cols
        )

        return X, feature_cols

    def create_balanced_dataset(self, injury_df, normal_df, ratio):
        """Create a dataset with specified injury to normal ratio"""
        n_injuries = len(injury_df)
        n_normal_needed = int(n_injuries * ratio)

        # Sample normal collisions
        if len(normal_df) >= n_normal_needed:
            normal_sample = normal_df.sample(n=n_normal_needed, random_state=self.random_state)
        else:
            # If we don't have enough, use all available with warning
            normal_sample = normal_df.copy()
            print(f"  ⚠️ Warning: Only {len(normal_df)} normal samples available for {ratio}:1 ratio")

        # Combine datasets
        combined_df = pd.concat([injury_df, normal_sample], ignore_index=True)

        # Prepare features
        X, feature_cols = self.prepare_features(combined_df)
        y = combined_df['is_injury'].values

        return X, y, feature_cols

    def evaluate_models(self, X, y, ratio):
        """Evaluate all models on the given dataset"""
        print(f"\nEvaluating models at {ratio}:1 ratio...")
        print(f"  Dataset: {len(X)} samples ({sum(y)} injuries, {len(y)-sum(y)} normal)")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ratio_results = {}

        for model_name, model in self.models.items():
            print(f"  Training {model_name}...")

            # Train model
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'avg_precision': average_precision_score(y_test, y_pred_proba)
            }

            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
            metrics['cv_auc_mean'] = cv_scores.mean()
            metrics['cv_auc_std'] = cv_scores.std()

            # Store confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)

            ratio_results[model_name] = metrics

        return ratio_results

    def run_progressive_validation(self, ratios=[10, 20, 50, 100, 200]):
        """Run validation with progressively increasing ratios"""
        print("="*70)
        print("PROGRESSIVE MODEL VALIDATION")
        print("="*70)

        # Load data
        injury_df, normal_df = self.load_collision_data()

        # Test each ratio
        for ratio in ratios:
            print(f"\n{'='*50}")
            print(f"Testing ratio 1:{ratio} (injury:normal)")
            print(f"{'='*50}")

            # Create balanced dataset
            X, y, feature_cols = self.create_balanced_dataset(injury_df, normal_df, ratio)

            # Evaluate models
            ratio_results = self.evaluate_models(X, y, ratio)

            # Store results
            self.results[ratio] = ratio_results

            # Print summary
            self.print_ratio_summary(ratio, ratio_results)

    def print_ratio_summary(self, ratio, ratio_results):
        """Print summary of results for a specific ratio"""
        print(f"\nSummary for 1:{ratio} ratio:")
        print("-" * 50)

        # Create summary table
        summary_data = []
        for model_name, metrics in ratio_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1': f"{metrics['f1']:.3f}",
                'AUC': f"{metrics['roc_auc']:.3f}",
                'AP': f"{metrics['avg_precision']:.3f}"
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # Find best model for this ratio
        best_model = max(ratio_results.keys(),
                        key=lambda x: ratio_results[x]['roc_auc'])
        best_auc = ratio_results[best_model]['roc_auc']
        print(f"\n🏆 Best model: {best_model} (AUC: {best_auc:.3f})")

        # Print confusion matrices for all models
        print(f"\nConfusion Matrices (Actual vs Predicted):")
        print("-" * 40)
        for model_name, metrics in ratio_results.items():
            cm = metrics['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()

            print(f"\n{model_name}:")
            print(f"              Predicted")
            print(f"              No Injury | Injury")
            print(f"  No Injury:  {tn:5d}    | {fp:5d}   (Specificity: {tn/(tn+fp):.3f})")
            print(f"  Injury:     {fn:5d}    | {tp:5d}   (Sensitivity/Recall: {tp/(tp+fn) if (tp+fn) > 0 else 0:.3f})")

            # Additional helpful metrics
            if tp + fp > 0:
                ppv = tp / (tp + fp)  # Positive Predictive Value
                print(f"  PPV (Precision): {ppv:.3f}")

            if tn + fn > 0:
                npv = tn / (tn + fn)  # Negative Predictive Value
                print(f"  NPV: {npv:.3f}")

            print(f"  False Positive Rate: {fp/(fp+tn) if (fp+tn) > 0 else 0:.3f}")
            print(f"  False Negative Rate: {fn/(fn+tp) if (fn+tp) > 0 else 0:.3f}")

    def plot_performance_trends(self):
        """Create enhanced visualizations of performance trends"""
        if not self.results:
            print("No results to plot. Run validation first.")
            return

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style for better looking plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        ratios = sorted(self.results.keys())
        models = list(self.results[ratios[0]].keys())

        # Define colors for each model
        colors = {
            'Logistic Regression': '#FF6B6B',
            'Random Forest': '#4ECDC4',
            'Gradient Boosting': '#45B7D1',
            'SVM (RBF)': '#96CEB4'
        }

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('🏈 NFL Collision Model Performance Across Class Imbalance Ratios',
                     fontsize=20, fontweight='bold', y=1.02)

        # 1. Main performance metrics (larger subplot)
        ax1 = plt.subplot(2, 3, (1, 4))

        for model in models:
            auc_values = [self.results[r][model]['roc_auc'] for r in ratios]
            ax1.plot(ratios, auc_values, marker='o', label=model,
                    linewidth=3, markersize=8, color=colors.get(model, 'gray'),
                    alpha=0.8)

        ax1.set_xlabel('Class Imbalance Ratio (Non-Injury : Injury)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ROC AUC Score', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance (ROC AUC) vs Class Imbalance', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.set_xticks(ratios)
        ax1.set_xticklabels([f'1:{r}' for r in ratios], rotation=45)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', frameon=True, shadow=True, fontsize=10)
        ax1.set_ylim([0.5, 1.05])

        # Add performance zones
        ax1.axhspan(0.9, 1.05, alpha=0.1, color='green', label='Excellent')
        ax1.axhspan(0.8, 0.9, alpha=0.1, color='yellow')
        ax1.axhspan(0.7, 0.8, alpha=0.1, color='orange')
        ax1.text(ratios[0], 0.95, 'Excellent', fontsize=9, alpha=0.5)
        ax1.text(ratios[0], 0.85, 'Good', fontsize=9, alpha=0.5)
        ax1.text(ratios[0], 0.75, 'Fair', fontsize=9, alpha=0.5)

        # 2. Recall (Sensitivity) - Critical for injury detection
        ax2 = plt.subplot(2, 3, 2)
        for model in models:
            recall_values = [self.results[r][model]['recall'] for r in ratios]
            ax2.plot(ratios, recall_values, marker='s', label=model,
                    linewidth=2.5, markersize=7, color=colors.get(model, 'gray'),
                    alpha=0.8)

        ax2.set_xlabel('Class Imbalance Ratio', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Recall (Injury Detection Rate)', fontsize=11, fontweight='bold')
        ax2.set_title('⚠️ Injury Detection Capability', fontsize=12, fontweight='bold')
        ax2.set_xscale('log')
        ax2.set_xticks(ratios)
        ax2.set_xticklabels([f'1:{r}' for r in ratios], rotation=45)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim([-0.05, 1.05])

        # 3. Precision
        ax3 = plt.subplot(2, 3, 3)
        for model in models:
            precision_values = [self.results[r][model]['precision'] for r in ratios]
            ax3.plot(ratios, precision_values, marker='^', label=model,
                    linewidth=2.5, markersize=7, color=colors.get(model, 'gray'),
                    alpha=0.8)

        ax3.set_xlabel('Class Imbalance Ratio', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Precision', fontsize=11, fontweight='bold')
        ax3.set_title('🎯 Precision (False Alarm Rate)', fontsize=12, fontweight='bold')
        ax3.set_xscale('log')
        ax3.set_xticks(ratios)
        ax3.set_xticklabels([f'1:{r}' for r in ratios], rotation=45)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_ylim([-0.05, 1.05])

        # 4. F1 Score
        ax4 = plt.subplot(2, 3, 5)
        for model in models:
            f1_values = [self.results[r][model]['f1'] for r in ratios]
            ax4.plot(ratios, f1_values, marker='D', label=model,
                    linewidth=2.5, markersize=7, color=colors.get(model, 'gray'),
                    alpha=0.8)

        ax4.set_xlabel('Class Imbalance Ratio', fontsize=11, fontweight='bold')
        ax4.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax4.set_title('⚖️ Balanced Performance (F1)', fontsize=12, fontweight='bold')
        ax4.set_xscale('log')
        ax4.set_xticks(ratios)
        ax4.set_xticklabels([f'1:{r}' for r in ratios], rotation=45)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_ylim([-0.05, 1.05])

        # 5. Heatmap of confusion matrix values at highest ratio
        ax5 = plt.subplot(2, 3, 6)

        # Create a summary matrix for the highest ratio
        highest_ratio = max(ratios)
        heatmap_data = []

        for model in models:
            cm = self.results[highest_ratio][model]['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp

            # Calculate key rates
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            heatmap_data.append([tpr * 100, (1 - fpr) * 100, precision * 100])

        # Create heatmap
        heatmap_df = pd.DataFrame(heatmap_data,
                                  index=[m.replace(' ', '\n') for m in models],
                                  columns=['Recall\n(%)', 'Specificity\n(%)', 'Precision\n(%)'])

        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=50, vmin=0, vmax=100, ax=ax5, cbar_kws={'label': 'Performance (%)'})
        ax5.set_title(f'📊 Model Performance at 1:{highest_ratio} Ratio', fontsize=12, fontweight='bold')
        ax5.set_xlabel('')
        ax5.set_ylabel('Models', fontsize=11, fontweight='bold')

        plt.tight_layout()

        # Save multiple versions
        plt.savefig('capstone_file/punt_analytics/performance_trends_enhanced.png',
                   dpi=300, bbox_inches='tight', facecolor='white')

        print("\n📊 Enhanced performance plots saved to capstone_file/punt_analytics/performance_trends_enhanced.png")

        # Create a second figure focusing on the trade-offs
        self.plot_precision_recall_tradeoff()

    def plot_precision_recall_tradeoff(self):
        """Create a focused visualization on the precision-recall trade-off"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        ratios = sorted(self.results.keys())
        models = list(self.results[ratios[0]].keys())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('🎯 Precision-Recall Trade-off Analysis for Injury Detection',
                     fontsize=18, fontweight='bold')

        # Define colors
        colors = {
            'Logistic Regression': '#FF6B6B',
            'Random Forest': '#4ECDC4',
            'Gradient Boosting': '#45B7D1',
            'SVM (RBF)': '#96CEB4'
        }

        # 1. Precision vs Recall curves
        ax1 = axes[0]
        for model in models:
            precisions = [self.results[r][model]['precision'] for r in ratios]
            recalls = [self.results[r][model]['recall'] for r in ratios]

            # Plot the curve
            ax1.plot(recalls, precisions, marker='o', label=model,
                    linewidth=2.5, markersize=8, color=colors.get(model, 'gray'),
                    alpha=0.8)

            # Annotate each point with the ratio
            for i, (rec, prec, ratio) in enumerate(zip(recalls, precisions, ratios)):
                if i % 2 == 0:  # Only annotate every other point to avoid clutter
                    ax1.annotate(f'1:{ratio}', (rec, prec),
                               textcoords="offset points", xytext=(5, 5),
                               fontsize=8, alpha=0.6)

        ax1.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax1.set_title('Precision-Recall Trade-off', fontsize=14)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', frameon=True, shadow=True)
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])

        # Add diagonal reference line
        ax1.plot([0, 1], [1, 0], 'k--', alpha=0.3, label='Random')

        # 2. Model recommendation zones
        ax2 = axes[1]

        # Create zones based on use case
        zone_data = []
        for model in models:
            # Get performance at different ratios
            perf_20 = self.results[20][model]
            perf_100 = self.results[100][model]
            perf_500 = self.results[500][model]

            zone_data.append({
                'Model': model.replace(' ', '\n'),
                'Low Imbalance\n(1:20)': perf_20['roc_auc'],
                'Medium Imbalance\n(1:100)': perf_100['roc_auc'],
                'High Imbalance\n(1:500)': perf_500['roc_auc']
            })

        zone_df = pd.DataFrame(zone_data)
        zone_df = zone_df.set_index('Model')

        # Create heatmap
        sns.heatmap(zone_df.T, annot=True, fmt='.3f', cmap='YlOrRd',
                   vmin=0.5, vmax=1.0, ax=ax2, cbar_kws={'label': 'ROC AUC'})
        ax2.set_title('Model Recommendations by Imbalance Level', fontsize=14)
        ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Imbalance Level', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig('capstone_file/punt_analytics/precision_recall_tradeoff.png',
                   dpi=300, bbox_inches='tight', facecolor='white')

        print("📊 Precision-Recall trade-off plot saved to capstone_file/punt_analytics/precision_recall_tradeoff.png")

    def generate_final_report(self):
        """Generate a comprehensive report of all results"""
        if not self.results:
            print("No results to report. Run validation first.")
            return

        print("\n" + "="*70)
        print("FINAL VALIDATION REPORT")
        print("="*70)

        ratios = sorted(self.results.keys())

        # Find overall best model across all ratios
        best_performances = {}
        for model in self.models.keys():
            avg_auc = np.mean([self.results[r][model]['roc_auc'] for r in ratios])
            best_performances[model] = avg_auc

        overall_best = max(best_performances.keys(), key=lambda x: best_performances[x])

        print(f"\n🏆 OVERALL BEST MODEL: {overall_best}")
        print(f"   Average AUC across all ratios: {best_performances[overall_best]:.3f}")

        # Performance at different ratios
        print("\n📊 PERFORMANCE SUMMARY BY RATIO:")
        print("-" * 50)

        for ratio in ratios:
            print(f"\nRatio 1:{ratio}:")
            best_at_ratio = max(self.results[ratio].keys(),
                              key=lambda x: self.results[ratio][x]['roc_auc'])
            best_auc = self.results[ratio][best_at_ratio]['roc_auc']
            best_recall = self.results[ratio][best_at_ratio]['recall']
            print(f"  Best: {best_at_ratio} (AUC: {best_auc:.3f}, Recall: {best_recall:.3f})")

        # Model robustness analysis
        print("\n🔍 MODEL ROBUSTNESS ANALYSIS:")
        print("-" * 50)

        for model in self.models.keys():
            aucs = [self.results[r][model]['roc_auc'] for r in ratios]
            recalls = [self.results[r][model]['recall'] for r in ratios]

            print(f"\n{model}:")
            print(f"  AUC range: {min(aucs):.3f} - {max(aucs):.3f} (std: {np.std(aucs):.3f})")
            print(f"  Recall range: {min(recalls):.3f} - {max(recalls):.3f} (std: {np.std(recalls):.3f})")

            # Check if performance degrades significantly
            auc_drop = aucs[0] - aucs[-1]  # Drop from lowest to highest ratio
            if auc_drop > 0.1:
                print(f"  ⚠️ Warning: Significant performance drop ({auc_drop:.3f}) at high imbalance")
            else:
                print(f"  ✅ Robust to class imbalance (drop: {auc_drop:.3f})")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 50)

        # Find the ratio where performance starts to degrade significantly
        for model in self.models.keys():
            aucs = [self.results[r][model]['roc_auc'] for r in ratios]
            for i in range(1, len(aucs)):
                if aucs[i-1] - aucs[i] > 0.05:  # 5% drop
                    print(f"{model}: Performance degrades significantly beyond 1:{ratios[i-1]} ratio")
                    break
            else:
                print(f"{model}: Maintains stable performance across all tested ratios")

        # Save detailed results to CSV
        self.save_results_to_csv()

        return overall_best, best_performances

    def save_results_to_csv(self):
        """Save detailed results to CSV files"""
        import os
        os.makedirs('capstone_file/punt_analytics', exist_ok=True)

        # Create a comprehensive results dataframe
        results_list = []
        for ratio in self.results.keys():
            for model in self.results[ratio].keys():
                result = {
                    'ratio': ratio,
                    'model': model
                }
                result.update({k: v for k, v in self.results[ratio][model].items()
                             if k != 'confusion_matrix'})
                results_list.append(result)

        results_df = pd.DataFrame(results_list)
        results_df.to_csv('capstone_file/punt_analytics/validation_results.csv', index=False)
        print("\n📁 Detailed results saved to capstone_file/punt_analytics/validation_results.csv")


def main():
    """Main execution function"""
    print("🏈 NFL Collision Model Validation - Progressive Ratio Testing")
    print("="*70)

    # Initialize validator
    validator = ProgressiveModelValidator(random_state=42)

    # Define ratios to test (starting at 20:1 as requested)
    ratios = [20, 50, 100, 200, 500]

    # Run progressive validation
    validator.run_progressive_validation(ratios=ratios)

    # Generate visualizations
    validator.plot_performance_trends()

    # Generate final report
    best_model, performances = validator.generate_final_report()

    print("\n✅ Validation complete!")
    print(f"📊 Best overall model: {best_model}")
    print("📁 Results saved to capstone_file/punt_analytics/")

    return validator


if __name__ == "__main__":
    validator = main()