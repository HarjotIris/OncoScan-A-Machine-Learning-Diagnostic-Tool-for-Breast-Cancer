# Semi-Advanced EDA
import pandas as pd

path = r"C:\Desktop\BREAST CANCER PREDICTOR\Breast_cancer_dataset.csv"
df = pd.read_csv(path)

print("Dataset shape and basic information")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.to_list()}")

'''
['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32']
'''


print("First 5 rows")
print(df.head(5))


print("Data types and missing values:")
print(df.dtypes)
print(df.info())
print("Missing values per column")
print(df.isnull().sum())
# Unnamed: 32 ---> 569 : means another reason to drop this column
# no missing values otherwise

df = df.drop(columns=['id', 'Unnamed: 32'])

print("Diagnosis distribution")
print(df['diagnosis'].value_counts())
print("Diagnosis percentage")
print(df['diagnosis'].value_counts(normalize=True)*100)
# B    62.741652
# M    37.258348

import numpy as np
print("Numerical and Categorical columns: ")
num_cols = df.select_dtypes(include=np.number).columns.to_list()
cat_cols = df.select_dtypes(include='object').columns.to_list()
print(f"Numerical Features ({len(num_cols)}) : {num_cols}")
print(f"Categorical Features ({len(cat_cols)}) : {cat_cols}")

# all features are numerical except the target feature

print("Feature Statistics")
print(df.describe())


print("Check for unusual values")

for col in df.columns:
    unique_vals = df[col].nunique()
    print(f"{col} : {unique_vals} unique values")

    if unique_vals < 10:
        print(f"Values : {df[col].unique()}")
    print()


print("Mapping the values in the target to a binary equivalent")
df['diagnosis'] = df['diagnosis'].map({"M" : 1, "B": 0})
#print(df['diagnosis'])
'''
print("Separating features and target")
x = df.drop(columns=['diagnosis', 'id'], axis=1)
y = df['diagnosis']

print(f"Features Shape : {x.shape}")
print(f"Target Shape : {y.shape}")
print(f"Target : {y.name}")
'''
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Matrix")
plt.show()

# correlation ---> heatmap
print("Correlation with Target")
corr_with_target = corr_matrix['diagnosis'].drop('diagnosis').sort_values(ascending=True)
print(f"Top features positively correlated with the target : {corr_with_target.tail(10)}")
print(f"Top features negatively correlated with the target : {corr_with_target.head(10)}")


# class distribution ---> countplot
sns.countplot(data=df, x='diagnosis')
plt.title("Class Distribution of Diagnosis")
plt.xticks([0, 1], ('Benign (0)', 'Malignant (1)'))
plt.show()

''' figures saved in repo, commented out to save runtime
# outlier detection ---> boxplot
for i in range(0, len(df.columns), 5):
    for col in df.columns[i:i+5]:
        sns.boxplot(data=df, x='diagnosis', y=col)
        plt.title(f"Boxplot of {col} by diagnosis")
        plt.show()
'''

sns.pairplot(df[['radius_mean', 'texture_mean', 'area_mean', 'concavity_mean', 'diagnosis']], hue='diagnosis')
plt.show()


from scipy.stats import skew

skews = df.skew().sort_values(ascending=False)
print("Top skewed features:")
print(skews.head())

for col in df.columns[:5]:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

'''
Top skewed features:
area_se                 5.447186
concavity_se            5.110463
fractal_dimension_se    3.923969
perimeter_se            3.443615
radius_se               3.088612
'''
#print("Feature Variances")
#print(df.var().sort_values())

'''
Feature Variances
fractal_dimension_se            0.000007
smoothness_se                   0.000009
concave points_se               0.000038
fractal_dimension_mean          0.000050
symmetry_se                     0.000068
smoothness_mean                 0.000198
compactness_se                  0.000321
fractal_dimension_worst         0.000326
smoothness_worst                0.000521
symmetry_mean                   0.000752
concavity_se                    0.000911
concave points_mean             0.001506
compactness_mean                0.002789
symmetry_worst                  0.003828
concave points_worst            0.004321
concavity_mean                  0.006355
compactness_worst               0.024755
concavity_worst                 0.043524
radius_se                       0.076902
diagnosis                       0.234177
texture_se                      0.304316
perimeter_se                    4.087896
radius_mean                    12.418920
texture_mean                   18.498909
radius_worst                   23.360224
texture_worst                  37.776483
perimeter_mean                590.440480
perimeter_worst              1129.130847
area_se                      2069.431583
area_mean                  123843.554318
area_worst                 324167.385102
'''


# duplicate rows check
print(f"Duplicated rows : {df.duplicated().sum()}")
# no duplicated rows

low_var = df.var()[df.var() < 0.001]
print("Low Variance Features:")
print(low_var)

'''
Low Variance Features:
smoothness_mean            0.000198
symmetry_mean              0.000752
fractal_dimension_mean     0.000050
smoothness_se              0.000009
compactness_se             0.000321
concavity_se               0.000911
concave points_se          0.000038
symmetry_se                0.000068
fractal_dimension_se       0.000007
smoothness_worst           0.000521
fractal_dimension_worst    0.000326
'''


# feature redundancy check!
def get_high_corr_pairs(corr_matrix, threshold=0.9):
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper = corr_matrix.where(mask)
    pairs = upper.stack().sort_values(ascending=False)
    return pairs[pairs > threshold]

high_corr_pairs = get_high_corr_pairs(df.corr(), 0.9)
print(high_corr_pairs)

''' # drop a feature from each highly collinear pair
radius_mean          perimeter_mean          0.997855
radius_worst         perimeter_worst         0.993708
radius_mean          area_mean               0.987357
perimeter_mean       area_mean               0.986507
radius_worst         area_worst              0.984015
perimeter_worst      area_worst              0.977578
radius_se            perimeter_se            0.972794
perimeter_mean       perimeter_worst         0.970387
radius_mean          radius_worst            0.969539
perimeter_mean       radius_worst            0.969476
radius_mean          perimeter_worst         0.965137
area_mean            radius_worst            0.962746
                     area_worst              0.959213
                     perimeter_worst         0.959120
radius_se            area_se                 0.951830
perimeter_mean       area_worst              0.941550
radius_mean          area_worst              0.941082
perimeter_se         area_se                 0.937655
concavity_mean       concave points_mean     0.921391
texture_mean         texture_worst           0.912045
concave points_mean  concave points_worst    0.910155
'''

# feature distribution by class, for a target variable with two classes
def class_mean_diff(df, target='diagnosis', top_n=10):
    group_stats = df.groupby(target).mean().T
    group_stats['diff'] = group_stats[1] - group_stats[0]
    return group_stats.sort_values('diff', ascending=False).head(top_n)

print(class_mean_diff(df, 'diagnosis', 10))

''' # higher diff means the feature separates better
diagnosis                 0            1        diff
area_worst       558.899440  1422.286321  863.386881
area_mean        462.790196   978.376415  515.586219
perimeter_worst   87.005938   141.370330   54.364392
area_se           21.135148    72.672406   51.537257
perimeter_mean    78.075406   115.365377   37.289971
radius_worst      13.379801    21.134811    7.755010
texture_worst     23.515070    29.318208    5.803138
radius_mean       12.146524    17.462830    5.316306
texture_mean      17.914762    21.604906    3.690144
perimeter_se       2.000321     4.323929    2.323608
'''

# visualising skew correction potential
for col in ['area_se', 'concavity_se', 'perimeter_se']:
    df[f'log_{col}'] = np.log1p(df[col])
    sns.histplot(df[f'log_{col}'], kde=True)
    plt.title(f'Log Distribution of {col}')
    plt.show()
