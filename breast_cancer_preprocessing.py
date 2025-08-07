import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.feature_selection import SelectKBest, f_classif, RFE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    import joblib
    import os
    import streamlit as st
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.stop()


class BreastCancerPreprocessor:
    """
    A comprehensive preprocessing pipeline for breast cancer prediction.
    This class demonstrates key preprocessing techniques I will use across ML projects.
    """

    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.pca = None
        self.features_to_drop = []

    def handle_multicollinearity(self, df, threshold = 0.95):
        """
        Step 1: Remove highly correlated features
        Why: Prevents redundant information and improves model stability
        When to use: Always check for correlation > 0.9-0.95
        """

        print("Handling Multicollinearity")

        features = df.drop('diagnosis', axis = 1)

        corr_matrix = features.corr().abs()

        # abs here because we care about how strong the relationship is here not the direction of the relationship

        # find highly correlated pairs

        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

        high_corr_pairs = [(corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]) 
                           for i in range(len(corr_matrix.columns))
                           for j in range(i+1, len(corr_matrix.columns))
                           if corr_matrix.iloc[i, j] > threshold]
        '''
        So for every i, j where i < j:

        If corr_matrix[i][j] > 0.95, we store (feature1, feature2, correlation value) in high_corr_pairs.
        '''
        # drop the features with lower correlation with the target
        features_to_drop = set()
        target_corr = df.corr()['diagnosis'].abs()

        # Because we want to keep the feature more related to the target and drop the less useful one.

        for feat1, feat2, corr_val in high_corr_pairs:
            if target_corr[feat1] > target_corr[feat2]:
                features_to_drop.add(feat2)

            else:
                features_to_drop.add(feat1)

            '''
            For each highly correlated pair:
            - Keep the one more correlated with the target (diagnosis)
            - Drop the other one
            '''
        
        # saving it into the class variable
        self.features_to_drop = list(features_to_drop)

        print(f"Dropping {len(self.features_to_drop)} highly correlated features: ")

        print(self.features_to_drop[:5], "..." if len(self.features_to_drop) > 5 else "")

        return df.drop(columns = self.features_to_drop)


    def handle_outliers(self, df, method='iqr'):
        """
        Step 2: Outlier detection and handling
        Why: Outliers can skew model performance, especially for linear models
        Methods: IQR, Z-score, or domain knowledge
        """
        print(f"HANDLING OUTLIERS (method: {method})")

        features = df.select_dtypes(include = np.number).columns.drop('diagnosis')

        outlier_counts = {}

        if method == 'iqr':
            for col in features:
                Q1 = df[col].quantile(0.25) # Q1 = 25th percentile
                Q3 = df[col].quantile(0.75) # Q3 = 75th percentile
                IQR = Q3 - Q1
                # IQR = Interquartile range = Q3 - Q1
                # this is the middle spread of our data

                # formulas for calculating lower and upper bounds
                # Anything below lower_bound or above upper_bound is
                # considered an outlier
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_counts[col] = outliers
                
                # Cap outliers instead of removing (preserves data size)
                df[col] = np.clip(df[col], lower_bound, upper_bound)
        

        # z-score measures how many standard deviations away a value
        # is from the mean. If it's more than 3, it's an outlier
        elif method == 'zscore':
            for col in features:
                z_scores = np.abs(stats.zscore(df[col]))
                outliers = (z_scores > 3).sum()
                outlier_counts[col] = outliers
                
                # Cap outliers at 3 standard deviations
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[col] = np.clip(df[col], mean_val - 3*std_val, mean_val + 3*std_val)
        
        total_outliers = sum(outlier_counts.values())
        print(f"Total outliers handled: {total_outliers}")
        print(f"Top 5 features with most outliers: {sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        return df


    def handle_skewness(self, df, threshold = 1.0):
        """
        Step 3: Transform skewed features
        Why: Many ML algorithms assume normal distribution
        When: Skewness > 1.0 or < -1.0
        """
        '''
        This function checks which features are heavily skewed (i.e., not symmetric like a normal bell curve) and fixes them.

        Why this matters:

        Many machine learning models — especially linear models — perform better when features are normally (symmetrically) distributed.

        Note: Skips columns with negative values to preserve data integrity
        '''

        print(f"Handling Skewness for threshold : {threshold}")
        features = df.select_dtypes(include=[np.number]).columns.drop('diagnosis')
        skewed_features = []
        skipped_features = []

        for col in features:
            skewness = df[col].skew()
            if abs(skewness) > threshold: # if skewness is too high or low
                skewed_features.append((col, skewness))

                # Apply log transformation (add 1 to handle zeros)
                if df[col].min() >= 0:
                    df[f'{col}_log'] = np.log1p(df[col])
                    df = df.drop(columns=[col])  # Replace original with transformed
                    df = df.rename(columns={f'{col}_log': col})

                else:
                    skipped_features.append((col, skewness))

        print(f"Transformed {len(skewed_features)} skewed features")
        if skewed_features:
            print("Features transformed:", [feat[0] for feat in skewed_features[:5]])

        if skipped_features:
            print(f"Skipped {len(skipped_features)} features due to negative values")
            print("Skipped features:", [feat[0] for feat in skipped_features[:5]])

        return df
    
    def scale_features(self, x_train, x_test, method='standard'):
        """
        Step 4: Feature scaling
        Why: Algorithms like SVM, KNN, Neural Networks are sensitive to feature scales
        Methods: StandardScaler (most common), RobustScaler (for outliers), MinMaxScaler
        """

        print(f"Scaling features with method : {method}")

        if method == 'standard':
            self.scaler = StandardScaler()

        elif method == 'robust':
            self.scaler = RobustScaler()

        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaling method. Choose from: 'standard', 'robust', 'minmax'.")

        # fitting only on training data to prevent data leakage
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        import joblib
        import os
        joblib.dump(self.scaler, MODELS_DIR / 'scaler.joblib')


        # converting back to a dataframe to maintain column names
        x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)

        x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)

        print(f"Features scaled using {method} scaling")
        print(f"Training set mean: {x_train_scaled.mean().mean():.4f}")
        print(f"Training set std: {x_train_scaled.std().mean():.4f}")
        
        return x_train_scaled, x_test_scaled
    
    def select_features(self, x_train, y_train, x_test, method='rfe', k=15):
        """
        Step 5: Feature selection
        Why: Reduces overfitting, improves interpretability, faster training
        Methods: 
        - SelectKBest: Statistical tests (univariate)
        - RFE: Recursive Feature Elimination (model-based)
        - Feature importance from tree models
        """
        joblib.dump(self.scaler, MODELS_DIR / 'scaler.joblib')

        print(f"Feature Selection : Method : {method}, k : {k}")
        if method == 'selectkbest':
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            #SelectKBest picks the top k features using a statistical test
            #f_classif is ANOVA F-test (good for classification)

            x_train_selected = self.feature_selector.fit_transform(x_train, y_train)
            x_test_selected = self.feature_selector.transform(x_test)
            
            # Get selected feature names
            selected_mask = self.feature_selector.get_support()
            self.selected_features = x_train.columns[selected_mask].tolist()
            
        elif method == 'rfe':
            # Use RandomForest as the estimator for RFE
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            self.feature_selector = RFE(estimator, n_features_to_select=k)
            x_train_selected = self.feature_selector.fit_transform(x_train, y_train)
            x_test_selected = self.feature_selector.transform(x_test)
            
            # Get selected feature names
            selected_mask = self.feature_selector.get_support()
            self.selected_features = x_train.columns[selected_mask].tolist()
            # .get_support() returns a boolean array like: [True, False, True, ...]

            # This tells you which features were selected

            # You store those names in self.selected_features

            # Great for later usage or logging.
        
        elif method == 'importance':
            # Use feature importance from Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(x_train, y_train)
            
            feature_importance = pd.DataFrame({
                'feature': x_train.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.selected_features = feature_importance.head(k)['feature'].tolist()
            x_train_selected = x_train[self.selected_features]
            x_test_selected = x_test[self.selected_features]
        
        print(f"Selected {len(self.selected_features)} features:")
        print(self.selected_features[:5], "..." if len(self.selected_features) > 5 else "")
        
        # Convert back to DataFrame if needed
        if not isinstance(x_train_selected, pd.DataFrame):
            x_train_selected = pd.DataFrame(x_train_selected, columns=self.selected_features, index=x_train.index)
            x_test_selected = pd.DataFrame(x_test_selected, columns=self.selected_features, index=x_test.index)
        
        return x_train_selected, x_test_selected
    

    def apply_pca(self, x_train, x_test, variance_threshold=0.95):
        """
        Step 6 (Optional): Principal Component Analysis
        Why: Dimensionality reduction, noise reduction
        When: High-dimensional data, want to reduce features while preserving variance
        """
        print(f"Applying PCA (variance threshold: {variance_threshold})")
        
        self.pca = PCA()
        x_train_pca = self.pca.fit_transform(x_train)
        
        # Find number of components for desired variance
        cumsum_variance = np.cumsum(self.pca.explained_variance_ratio_)
        
        # explained_variance_ratio_: list like [0.35, 0.2, 0.15, ...]

        # These are how much each PC (Principal Component) contributes to total variance

        # np.cumsum() → running sum. So now you get:
        # [0.35, 0.55, 0.70, 0.82, 0.91, 0.96, ...]

        n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        # Finds the first index where cumulative variance ≥ 0.95
        # +1 because Python indexing starts at 0

        # So if you want 95% variance, it’ll return something like 6,   meaning:
        # “Keep the first 6 components, they explain at least 95% of the dataset.”

        # Refit with optimal number of components
        # actual dimensionality reduction step
        self.pca = PCA(n_components=n_components)
        x_train_pca = self.pca.fit_transform(x_train)
        x_test_pca = self.pca.transform(x_test)
        
        print(f"Reduced from {x_train.shape[1]} to {n_components} components")
        print(f"Explained variance: {cumsum_variance[n_components-1]:.4f}")
        
        # Create component names
        component_names = [f'PC{i+1}' for i in range(n_components)]
        x_train_pca = pd.DataFrame(x_train_pca, columns=component_names, index=x_train.index)
        x_test_pca = pd.DataFrame(x_test_pca, columns=component_names, index=x_test.index)
        
        return x_train_pca, x_test_pca
    
    def full_preprocessing_pipeline(self, df, use_pca=False, scaling_method='standard', 
                                  feature_selection_method='rfe', n_features=15):
        """
        Complete preprocessing pipeline
        This is my template for future projects!
        """
        print("STARTING FULL PREPROCESSING PIPELINE")
        
        # Step 1: Handle multicollinearity
        df = self.handle_multicollinearity(df, threshold=0.95)
        
        # Step 2: Handle outliers
        df = self.handle_outliers(df, method='iqr')
        
        # Step 3: Handle skewness
        df = self.handle_skewness(df, threshold=1.0)
        
        # Split features and target
        x = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        
        # Step 5: Train-test split (BEFORE scaling to prevent data leakage)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        ) # stratification is important for classification
        
        print(f"\nTrain-test split:")
        print(f"Training set: {x_train.shape[0]} samples")
        print(f"Test set: {x_test.shape[0]} samples")
        
        # Step 6: Scale features
        x_train_scaled, x_test_scaled = self.scale_features(
            x_train, x_test, method=scaling_method
        )
        
        # Step 7: Feature selection
        x_train_final, x_test_final = self.select_features(
            x_train_scaled, y_train, x_test_scaled, 
            method=feature_selection_method, k=n_features
        )
        
        # Step 8 (Optional): PCA
        if use_pca:
            x_train_final, x_test_final = self.apply_pca(
                x_train_final, x_test_final, variance_threshold=0.95
            )
        
        print("\nPREPROCESSING COMPLETE!")
        print(f"Final feature shape: {x_train_final.shape}")
        print(f"Final features: {list(x_train_final.columns)[:5]}")
        
        return x_train_final, x_test_final, y_train, y_test
    
    def visualize_preprocessing_effects(self, original_df, processed_x):
        """
        Visualize the effects of preprocessing
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original distribution
        original_features = original_df.select_dtypes(include=[np.number]).drop('diagnosis', axis=1)
        axes[0, 0].hist(original_features.iloc[:, 0], bins=30, alpha=0.7)
        axes[0, 0].set_title('Original Feature Distribution')
        
        # Processed distribution
        axes[0, 1].hist(processed_x.iloc[:, 0], bins=30, alpha=0.7)
        axes[0, 1].set_title('Processed Feature Distribution')
        
        # Feature correlation heatmap
        if len(processed_x.columns) <= 20:  # Only if manageable number of features
            sns.heatmap(processed_x.corr(), ax=axes[1, 0], cmap='coolwarm', center=0)
            axes[1, 0].set_title('Feature Correlations After Preprocessing')
        
        # Feature importance (if available)
        if hasattr(self, 'selected_features') and self.selected_features:
            feature_names = self.selected_features[:10]
            importance_values = np.random.rand(len(feature_names))  # Placeholder
            # Replace with actual importance values from model:
                # importance_values = model.feature_importances_[selected_feature_indices]

            axes[1, 1].barh(feature_names, importance_values)
            axes[1, 1].set_title('Selected Features')
        
        plt.tight_layout()
        plt.show()
