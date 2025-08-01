from breast_cancer_eda import load_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats
import warnings
warnings.filterwarnings('ignore')


path = r"C:\Desktop\BREAST CANCER PREDICTOR\Breast_cancer_dataset.csv"

df = pd.read_csv(path)
df = load_data(path)

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
        Step 3: Outlier detection and handling
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

