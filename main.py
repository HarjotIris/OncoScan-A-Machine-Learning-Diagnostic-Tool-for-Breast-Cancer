"""
Complete Breast Cancer Prediction Pipeline
My Master Template for Binary Classification Projects
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from breast_cancer_preprocessing import BreastCancerPreprocessor
from evaluation_pipeline import MLTrainingPipeline

def main():
    """
    Complete end-to-end ML pipeline execution
    This is my reusable template for future binary classification projects
    """

    print("="*60)
    print("BREAST CANCER PREDICTION - COMPLETE ML PIPELINE")
    print("="*60)
    
    # Step 1: Load and basic prep
    print("\n1. LOADING DATA")
    path = r"C:\Desktop\BREAST CANCER PREDICTOR\data\Breast_cancer_dataset.csv"
    df = pd.read_csv(path)

    # Basic cleaning (from your EDA)
    df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({"M": 1, "B": 0})
    
    print(f"Dataset loaded: {df.shape}")
    print(f"Class distribution: {df['diagnosis'].value_counts().to_dict()}")
    
    # Step 2: Comprehensive Preprocessing
    print("\n2. PREPROCESSING PIPELINE")
    preprocessor = BreastCancerPreprocessor()
    
    x_train, x_test, y_train, y_test = preprocessor.full_preprocessing_pipeline(
        df=df,
        use_pca=False,  # Try both True and False to learn the difference
        scaling_method='standard',  # Try: 'standard', 'robust', 'minmax'
        feature_selection_method='rfe',  # Try: 'rfe', 'selectkbest', 'importance'
        n_features=15
    )

    # Step 3: Complete ML Pipeline
    print("\n3. MACHINE LEARNING PIPELINE")
    ml_pipeline = MLTrainingPipeline(random_state=42)
    
    # Execute complete pipeline
    results = ml_pipeline.complete_ml_pipeline(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        tune_hyperparameters=True,  # Set False for quick testing
        save_model=True,
        model_save_path= r'C:\Desktop\BREAST CANCER PREDICTOR\models\best_model.joblib'

    )

    # Step 4: Final Summary and Learning Points
    print("\n4. FINAL SUMMARY")
    print("="*50)
    
    best_model = results['best_model_name']
    best_score = results['results'].iloc[0]['roc_auc']
    
    print(f"BEST MODEL: {best_model.replace('_', ' ').title()}")
    print(f"PERFORMANCE: {best_score:.4f} ROC-AUC")
    
    # Learning summary
    print(f"MODELS TESTED: {len(results['results'])}")
    print(f"FEATURES USED: {x_train.shape[1]}")
    print(f"TRAINING SAMPLES: {x_train.shape[0]}")
    print(f"TEST SAMPLES: {x_test.shape[0]}")

    # Performance breakdown
    print(f"\nTOP 3 MODELS:")
    top_3 = results['results'].head(3)
    for i, (model_name, row) in enumerate(top_3.iterrows(), 1):
        print(f"{i}. {model_name.replace('_', ' ').title()}: {row['roc_auc']:.4f}")
    
    # Key learnings to remember
    print(f"\nKEY LEARNINGS FOR FUTURE PROJECTS:")
    print("1. Always start with EDA to understand your data")
    print("2. Handle multicollinearity before feature selection")
    print("3. Scale features for distance-based algorithms")
    print("4. Use cross-validation for reliable model comparison")
    print("5. Hyperparameter tuning can significantly improve performance")
    print("6. Always evaluate multiple models - no single best algorithm")
    
    return results

def experiment_with_different_configs():
    """
    Run experiments with different configurations to learn their impact
    This is how you build intuition about ML techniques
    """
    
    print("\n" + "="*60)
    print("EXPERIMENTAL CONFIGURATIONS")
    print("="*60)
    
    # Load data once
    path = r"C:\Desktop\BREAST CANCER PREDICTOR\data\Breast_cancer_dataset.csv"

    df = pd.read_csv(path)
    df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({"M": 1, "B": 0})
    
    # Experiment 1: Different scaling methods
    scaling_methods = ['standard', 'robust', 'minmax']
    scaling_results = {}
    
    for scaling_method in scaling_methods:
        print(f"\nTesting {scaling_method} scaling...")
        
        preprocessor = BreastCancerPreprocessor()
        x_train, x_test, y_train, y_test = preprocessor.full_preprocessing_pipeline(
            df=df.copy(),
            scaling_method=scaling_method,
            feature_selection_method='rfe',
            n_features=10
        )
        
        # Quick model test
        ml_pipeline = MLTrainingPipeline(random_state=42)
        ml_pipeline.initialize_models()
        quick_results = ml_pipeline.quick_model_comparison(x_train, y_train, cv_folds=3)
        
        scaling_results[scaling_method] = quick_results['roc_auc_mean'].max()
        print(f"Best ROC-AUC with {scaling_method}: {scaling_results[scaling_method]:.4f}")
    
    print(f"\nSCALING METHOD COMPARISON:")
    for method, score in sorted(scaling_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method.title()}: {score:.4f}")
    
    # Experiment 2: Feature selection impact
    print(f"\n" + "-"*40)
    print("FEATURE SELECTION COMPARISON")
    print("-"*40)
    
    feature_counts = [5, 10, 15, 20]
    feature_results = {}
    
    for n_features in feature_counts:
        print(f"\nTesting {n_features} features...")
        
        preprocessor = BreastCancerPreprocessor()
        x_train, x_test, y_train, y_test = preprocessor.full_preprocessing_pipeline(
            df=df.copy(),
            scaling_method='standard',
            feature_selection_method='rfe',
            n_features=n_features
        )
        
        # Quick model test
        ml_pipeline = MLTrainingPipeline(random_state=42)
        ml_pipeline.initialize_models()
        quick_results = ml_pipeline.quick_model_comparison(x_train, y_train, cv_folds=3)
        
        feature_results[n_features] = quick_results['roc_auc_mean'].max()
        print(f"Best ROC-AUC with {n_features} features: {feature_results[n_features]:.4f}")
    
    print(f"\nFEATURE COUNT COMPARISON:")
    for count, score in sorted(feature_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{count} features: {score:.4f}")

if __name__ == "__main__":
    # Main pipeline execution
    main_results = main()
    
    # Optional: Run experiments to build intuition
    run_experiments = input("\nRun experimental configurations? (y/n): ").lower().strip()
    if run_experiments == 'y':
        experiment_with_different_configs()
    
    print(f"\nPROJECT COMPLETE! Model saved and ready for deployment.")