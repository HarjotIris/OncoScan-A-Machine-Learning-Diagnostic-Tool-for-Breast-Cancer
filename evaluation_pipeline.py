import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV, 
    RandomizedSearchCV, learning_curve, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import joblib
import time
import warnings
warnings.filterwarnings('ignore')


class MLTrainingPipeline:
    """
    Comprehensive ML Training and Evaluation Pipeline
    My complete template for binary classification projects
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.results = {}
        self.cv_results = {}

    def initialize_models(self):
        """
        Step 1: Initialize different types of algorithms
        Why: Different algorithms work better for different types of data
        Learn: Each algorithm's strengths and typical hyperparameters
        """
        print("Initializng Models")
        
        self.models = {
            # Linear Models ---> Fast, interpretable, good baseline
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga']
                },
                'type': 'linear'
            },
            
            # Tree-based Models - Handle non-linearity, feature interactions
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'type': 'ensemble'
            },
            
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'type': 'ensemble'
            },
            
            'gboost': {
                'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'type': 'ensemble'
            },
            
            # Instance-based - Simple, effective for smaller datasets
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                'type': 'instance'
            },
            
            # Support Vector Machine - Powerful for complex decision boundaries
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                },
                'type': 'kernel'
            },
            
            # Probabilistic - Fast, good baseline
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                },
                'type': 'probabilistic'
            },
            
            # Single Decision Tree - Interpretable, prone to overfitting
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                },
                'type': 'tree'
            }
        }
        
        print(f"Initialized {len(self.models)} different algorithms:")
        for name, info in self.models.items():
            print(f"{name.replace('_', ' ').title()} ({info['type']})")
        print()


    def quick_model_comparison(self, x_train, y_train, cv_folds=5):
        """
        Step 2: Quick comparison of all models with default parameters
        Why: Identify which algorithms work well with your data
        Learn: Cross-validation, model comparison methodology
        """
        print("QUICK MODEL COMPARISON (Default Parameters)")
        
        # Stratified K-Fold ensures balanced class distribution in each fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        comparison_results = {}
        
        for name, model_info in self.models.items():
            print(f"Training {name.replace('_', ' ').title()}...", end=' ')
            
            start_time = time.time()
            
            # Cross-validation with multiple metrics
            cv_scores = cross_val_score(
                model_info['model'], x_train, y_train, 
                cv=skf, scoring='roc_auc'
            )
            
            # Additional metrics
            accuracy_scores = cross_val_score(
                model_info['model'], x_train, y_train, 
                cv=skf, scoring='accuracy'
            )
            
            f1_scores = cross_val_score(
                model_info['model'], x_train, y_train, 
                cv=skf, scoring='f1'
            )
            
            training_time = time.time() - start_time
            
            comparison_results[name] = {
                'roc_auc_mean': cv_scores.mean(),
                'roc_auc_std': cv_scores.std(),
                'accuracy_mean': accuracy_scores.mean(),
                'accuracy_std': accuracy_scores.std(),
                'f1_mean': f1_scores.mean(),
                'f1_std': f1_scores.std(),
                'training_time': training_time
            }
            
            print(f"ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f}) | Time: {training_time:.2f}s")
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(comparison_results).T
        results_df = results_df.sort_values('roc_auc_mean', ascending=False)
        
        print(f"QUICK COMPARISON RESULTS (Ranked by ROC-AUC):")
        print(results_df.round(4))
        
        # Visualize results
        self._plot_model_comparison(results_df)
        
        return results_df
    
    def hyperparameter_tuning(self, x_train, y_train, top_n_models=3, 
                            search_method='random', n_iter=50, cv_folds=5):
        """
        Step 3: Hyperparameter tuning for top performing models
        Why: Squeeze out maximum performance from promising algorithms
        Learn: GridSearch vs RandomSearch, hyperparameter impact
        """
        print(f"HYPERPARAMETER TUNING (Method: {search_method.title()})")
        
        # Get top performing models from quick comparison
        if not hasattr(self, 'quick_results'):
            # If quick_model_comparison() hasn’t been run, do it now
            #  to get performance rankings.
            print("Running quick comparison first...")
            self.quick_results = self.quick_model_comparison(x_train, y_train)
        
        top_models = self.quick_results.head(top_n_models).index.tolist()
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name in top_models:
            print(f"\nTuning {model_name.replace('_', ' ').title()}...")
            
            model_info = self.models[model_name]
            base_model = model_info['model']
            param_grid = model_info['params']
            
            start_time = time.time()
            
            if search_method == 'grid':
                # Grid Search ---> Exhaustive but slower
                search = GridSearchCV(
                    base_model, param_grid, cv=skf, 
                    scoring='roc_auc', n_jobs=-1, verbose=0
                )
            else:
                # Random Search ---> Faster, often just as good
                search = RandomizedSearchCV(
                    base_model, param_grid, cv=skf, 
                    scoring='roc_auc', n_jobs=-1, verbose=0,
                    n_iter=n_iter, random_state=self.random_state
                )
            
            search.fit(x_train, y_train)
            tuning_time = time.time() - start_time
            
            # Store best model
            self.best_models[model_name] = search.best_estimator_
            
            print(f"  Best ROC-AUC: {search.best_score_:.4f}")
            print(f"  Best Parameters: {search.best_params_}")
            print(f"  Tuning Time: {tuning_time:.2f}s")
            
            # Store detailed results
            self.cv_results[model_name] = {
                'best_score': search.best_score_,
                'best_params': search.best_params_,
                'cv_results': search.cv_results_
            }
        
        print(f"Hyperparameter tuning complete for {len(top_models)} models!")

    def evaluate_models(self, x_train, y_train, x_test, y_test):
        """
        Step 4: Comprehensive model evaluation
        Why: Understand model performance from multiple angles
        Learn: All essential classification metrics and when to use them
        """
        print("COMPREHENSIVE MODEL EVALUATION")
        
        evaluation_results = {}
        
        # Use best models if available, otherwise use default models
        models_to_evaluate = self.best_models if self.best_models else {
            name: info['model'] for name, info in self.models.items()
        }
        
        for name, model in models_to_evaluate.items():
            print(f"\nEvaluating {name.replace('_', ' ').title()}...")
            
            # Train model
            start_time = time.time()
            model.fit(x_train, y_train)
            training_time = time.time() - start_time
            
            # Predictions
            start_time = time.time()
            y_pred = model.predict(x_test)
            y_pred_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, 'predict_proba') else None
            prediction_time = time.time() - start_time
            
            # Calculate all metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics['training_time'] = training_time
            metrics['prediction_time'] = prediction_time
            
            evaluation_results[name] = metrics
            
            # Print key metrics
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Create results DataFrame
        self.results = pd.DataFrame(evaluation_results).T
        self.results = self.results.sort_values('roc_auc', ascending=False)
        
        print(f"FINAL MODEL RANKINGS:")
        print(self.results[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].round(4))
        
        return self.results
    ''' 
        Learning Points: 
        Accuracy: Overall correctness

        Precision: How many predicted positives were correct

        Recall: How many actual positives you caught

        F1: Balance between precision and recall

        ROC-AUC: Quality of probability scores (higher = better)

        Training Time: Useful for cost vs. speed analysis

        Prediction Time: Helps in real-time vs. batch scenario decisions
    ''' 

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'specificity': self._calculate_specificity(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        else:
            metrics['roc_auc'] = np.nan
            metrics['avg_precision'] = np.nan
            
        return metrics
    
    
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (True Negative Rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    '''
    You don't need to memorize every formula.. just understand the intuition:

    Precision = How correct are your positive predictions

    Recall = How complete are your positive predictions

    Specificity = How good are you at catching negatives

    ROC-AUC = Can your model tell a positive apart from a negative using probability?

    Once you internalize this, you'll be able to explain your model with confidence in interviews.
    '''

    def detailed_model_analysis(self, x_train, y_train, x_test, y_test, model_name):
        """
        Step 5: Deep dive analysis of best model
        Why: Understand model behavior, identify potential issues
        Learn: Advanced evaluation techniques
        """
        print(f"DETAILED ANALYSIS: {model_name.replace('_', ' ').title()}")

        # Get the best model
        if model_name in self.best_models:
            model = self.best_models[model_name]
        else:
            model = self.models[model_name]['model']
            model.fit(x_train, y_train)
        
        # Predictions
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Detailed Analysis: {model_name.replace("_", " ").title()}', fontsize=16)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            '''
            fpr = false positive rate

            tpr = true positive rate

            AUC close to 1 = great, closer to 0.5 = random guessing
            '''
            axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {roc_auc:.4f})')
            axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend(loc="lower right")
        
        # 3. Precision-Recall Curve
        if y_pred_proba is not None:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            axes[0, 2].plot(recall, precision, color='blue', lw=2,
                           label=f'PR curve (AP = {avg_precision:.4f})')
            axes[0, 2].set_xlabel('Recall')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].set_title('Precision-Recall Curve')
            axes[0, 2].legend(loc="lower left")
        
        # 4. Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_names = x_train.columns if hasattr(x_train, 'columns') else [f'Feature_{i}' for i in range(x_train.shape[1])]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            axes[1, 0].barh(range(len(importance_df)), importance_df['importance'])
            axes[1, 0].set_yticks(range(len(importance_df)))
            axes[1, 0].set_yticklabels(importance_df['feature'])
            axes[1, 0].set_title('Top 10 Feature Importances')
            axes[1, 0].set_xlabel('Importance')
        elif hasattr(model, 'coef_'):
            # For linear models, show coefficient magnitudes
            feature_names = x_train.columns if hasattr(x_train, 'columns') else [f'Feature_{i}' for i in range(x_train.shape[1])]
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': np.abs(model.coef_[0])
            }).sort_values('coefficient', ascending=False).head(10)
            
            axes[1, 0].barh(range(len(coef_df)), coef_df['coefficient'])
            axes[1, 0].set_yticks(range(len(coef_df)))
            axes[1, 0].set_yticklabels(coef_df['feature'])
            axes[1, 0].set_title('Top 10 Coefficient Magnitudes')
            axes[1, 0].set_xlabel('|Coefficient|')
        
        # 5. Learning Curve
        train_sizes, train_scores, val_scores = learning_curve(
            model, x_train, y_train, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10), scoring='roc_auc'
        )
        '''
        Shows how model performs with increasing data:

        If gap between training & validation is large → overfitting
        If both low → underfitting

        You'll see 2 lines:

        Blue = Training
        Red = Cross-validation
        '''
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        axes[1, 1].plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        axes[1, 1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        axes[1, 1].plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
        axes[1, 1].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        axes[1, 1].set_xlabel('Training Set Size')
        axes[1, 1].set_ylabel('ROC-AUC Score')
        axes[1, 1].set_title('Learning Curve')
        axes[1, 1].legend(loc='best')
        axes[1, 1].grid(True)
        
        # 6. Prediction Distribution
        if y_pred_proba is not None:
            axes[1, 2].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.5, label='Benign', color='blue')
            axes[1, 2].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.5, label='Malignant', color='red')
            axes[1, 2].set_xlabel('Predicted Probability')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Prediction Probability Distribution')
            axes[1, 2].legend()
            '''
            Helps you visualize separation between classes

            Ideally, benign and malignant should form two separate clusters of probabilities
            '''
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed classification report
        print("DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
        
        return model
    
    def _plot_model_comparison(self, results_df):
        """Visualize model comparison results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # ROC-AUC comparison
        axes[0].barh(range(len(results_df)), results_df['roc_auc_mean'])
        axes[0].set_yticks(range(len(results_df)))
        axes[0].set_yticklabels([name.replace('_', ' ').title() for name in results_df.index])
        axes[0].set_xlabel('ROC-AUC Score')
        axes[0].set_title('Model Comparison: ROC-AUC')
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy comparison
        axes[1].barh(range(len(results_df)), results_df['accuracy_mean'])
        axes[1].set_yticks(range(len(results_df)))
        axes[1].set_yticklabels([name.replace('_', ' ').title() for name in results_df.index])
        axes[1].set_xlabel('Accuracy Score')
        axes[1].set_title('Model Comparison: Accuracy')
        axes[1].grid(True, alpha=0.3)
        
        # Training time comparison
        axes[2].barh(range(len(results_df)), results_df['training_time'])
        axes[2].set_yticks(range(len(results_df)))
        axes[2].set_yticklabels([name.replace('_', ' ').title() for name in results_df.index])
        axes[2].set_xlabel('Training Time (seconds)')
        axes[2].set_title('Model Comparison: Training Time')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def save_best_model(self, model_name, filepath):
        """
        Step 6: Save the best model for production use
        Why: Preserve trained model for deployment
        Learn: Model persistence and versioning
        """
        if model_name in self.best_models:
            model = self.best_models[model_name]
            joblib.dump(model, filepath)
            print(f"Model saved: {filepath}")
            
            # Save model metadata
            metadata = {
                'model_name': model_name,
                'model_type': self.models[model_name]['type'],
                'best_params': self.cv_results.get(model_name, {}).get('best_params', {}),
                'performance_metrics': self.results.loc[model_name].to_dict() if hasattr(self, 'results') else {}
            }
            
            metadata_filepath = filepath.replace('.joblib', '_metadata.json')
            import json
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"Metadata saved: {metadata_filepath}")
        else:
            print(f"Model {model_name} not found in best_models")


    def complete_ml_pipeline(self, x_train, y_train, x_test, y_test, 
                           tune_hyperparameters=True, save_model=True, 
                           model_save_path='best_model.joblib'):
        """
        Step 7: Complete end-to-end ML pipeline
        My template for future projects!
        """
        print("COMPLETE ML PIPELINE EXECUTION")
        
        # Step 1: Initialize models
        self.initialize_models()
        
        # Step 2: Quick comparison
        self.quick_results = self.quick_model_comparison(x_train, y_train)
        
        # Step 3: Hyperparameter tuning (optional)
        if tune_hyperparameters:
            self.hyperparameter_tuning(x_train, y_train, top_n_models=3)
        
        # Step 4: Final evaluation
        final_results = self.evaluate_models(x_train, y_train, x_test, y_test)
        
        # Step 5: Detailed analysis of best model
        best_model_name = final_results.index[0]

        best_model = self.detailed_model_analysis(x_train, y_train, x_test, y_test, best_model_name)
        
        # Step 6: Save best model
        if save_model:
            self.save_best_model(best_model_name, model_save_path)
        
        print(f"PIPELINE COMPLETE!")
        print(f"Best Model: {best_model_name.replace('_', ' ').title()}")
        print(f"Best ROC-AUC: {final_results.iloc[0]['roc_auc']:.4f}")
        
        return {
            'best_model_name': best_model_name,
            'best_model': best_model,
            'results': final_results,
            'quick_results': self.quick_results
        }
