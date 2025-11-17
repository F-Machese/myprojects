# Student Performance Analysis
# Author: Machese Fred Isaac
# Date: 03-11-2025 to 15-11-2025
# Description: Analyze student exam performance using Student_performance_10kz.csv dataset


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import ttest_ind, f_oneway, chi2_contingency, pearsonr


# Load dataset
df = pd.read_csv('Student_performance_10k.csv')

# Display first few rows
df.head(5)

# Check the shape of the dataset
print("Shape of dataset:", df.shape)

# Display column names
print("Columns:", df.columns)

# Fix column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Examine data types and non-null values
print(df.info())

# Summary of numeric columns
print(df.describe())

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Handle missing values for specific columns
# Remove rows with missing values in the 'rollno' column
df = df.dropna(subset=['roll_no'])

# For numeric columns, replace missing values with the column mean
numeric_columns = ['test_preparation_course', 'reading_score', 'writing_score', 'science_score', 'total_score']
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

# For categorical columns, replace missing values with the column mode
categorical_columns = ['math_score', 'lunch', 'gender', 'race_ethnicity', 'grade', 'parental_level_of_education']
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# List of numeric subject columnns
subject_cols = ['math_score', 'reading_score', 'writing_score', 'science_score']

# Convert subject columns to numeric, coercing errors to NaN
for col in subject_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# Create average_score column
df['average_score'] = df[subject_cols].mean(axis=1)

# Check for duplicate rows
print("Number of duplicate rows:", df.duplicated().sum())

# Check unique values for categorical columns
categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'grade']
for col in categorical_columns:
    print(f"Unique values in '{col}': {df[col].unique()}")

# Clean gender column
df['gender'] = df['gender'].replace({'Boy': 'male', 'Girl': 'female', '\\tmale': 'male'})

# Clean race_ethnicity column
df['race_ethnicity'] = df['race_ethnicity'].replace({
    'A': 'group A', 'B': 'group B', 'C': 'group C', 'D': 'group D', 'E': 'group E', 
    'group C\\n': 'group C'
})

# Clean parental_level_of_education column (just in case there are extra spaces)
df['parental_level_of_education'] = df['parental_level_of_education'].str.strip()

# Clean grade column (if any inconsistency found)
df['grade'] = df['grade'].replace({'Fail': 'F'})  # standardize 'Fail' to 'F' if needed

# Check unique values for categorical columns
categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'grade']
for col in categorical_columns:
    print(f"Unique values in '{col}': {df[col].unique()}")

# List of numeric columns
numeric_columns = ['math_score', 'reading_score', 'writing_score', 'science_score', 'total_score', 'average_score']

# Convert all numeric columns to numeric, replacing invalid entries with NaN
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle NaN values (impute with mean or drop)
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())  # Impute with column mean

# Remove rows with negative values (if negative values are invalid)
for col in numeric_columns:
    df = df[df[col] >= 0]  # Drop rows where column values are negative

# Replace inf values across the entire DataFrame
df = df.replace([float('inf'), float('-inf')], float('nan'))

# Check if there are still any missing values
print("Missing values per column after cleaning:\n", df.isnull().sum())

# Convert categorical columns to category type
categorical_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'grade']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Ensure numeric columns are integers or floats
numeric_cols = ['math_score', 'reading_score', 'writing_score', 'science_score', 'total_score']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

# Examine data types
print(df.info())

# Save cleaned dataset
df.to_csv('student_performance_cleaned.csv', index=False)

# Subject columns
subject_cols = ['math_score','reading_score','writing_score','science_score']

# Descriptive statistics
for col in subject_cols:
    mean_val = df[col].mean()
    median_val = df[col].median()
    mode_val = df[col].mode()[0]  # mode() returns a Series
    std_val = df[col].std()
    print(f"{col} - Mean: {mean_val:.2f}, Median: {median_val:.2f}, Mode: {mode_val:.2f}, Std: {std_val:.2f}")

# Group by categorical features and calculate average performance per category
    # Categories to group by
categories = ['gender', 'lunch', 'parental_level_of_education']

for cat in categories:
    print(f"\nAverage performance by {cat}:")
    grouped = df.groupby(cat)[['average_score']].mean().sort_values(by='average_score', ascending=False)
    print(grouped)

# Identify Top and Bottom Performers
# Top 10 students by average_score
top10 = df.nlargest(10, 'average_score')
print("\nTop 10 Students:")
print(top10[['roll_no','average_score','gender','lunch','parental_level_of_education']])

# Bottom 10 students by average_score
bottom10 = df.nsmallest(10, 'average_score')
print("\nBottom 10 Students:")
print(bottom10[['roll_no','average_score','gender','lunch','parental_level_of_education']])

# Distribution of categorical variables
print(df['gender'].value_counts())

# Boxplots for Average Scores by Gender, Lunch, Test Preparation
import matplotlib.pyplot as plt
import seaborn as sns

# Categories to compare
boxplot_cols = ['gender', 'lunch', 'test_preparation_course']

fig, axes = plt.subplots(1, 3, figsize=(18,5))
for ax, col in zip(axes, boxplot_cols):
    sns.boxplot(x=df[col], y=df['average_score'], palette='Set2', ax=ax)
    ax.set_title(f"Average Score by {col}")
    ax.set_ylabel("Average Score")
    ax.set_xlabel(col)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Plot histograms for numeric variables
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()  # Flatten for easier indexing

for ax, col in zip(axes, numeric_cols):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")

for ax in axes[len(numeric_cols):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()

# Use boxplots to detect outliers
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
axes = axes.flatten()
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(f"Boxplot of {col}")

for ax in axes[len(numeric_cols):]:
    ax.set_visible(False)
plt.tight_layout()
plt.show()

# One-hot encoding for categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Select only numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
corr_matrix = numeric_df.corr()
print(corr_matrix)

# Visualize correlation matrix
# Set figure size larger
plt.figure(figsize=(12, 10))  # Increase width/height as needed

# Create heatmap with annotations
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right', fontsize=6)
plt.yticks(rotation=0, fontsize=7)  # Keep y labels horizontal

plt.title("Correlation Matrix", fontsize=14)
plt.tight_layout()  # Adjust layout to avoid clipping
plt.show()

# Create scatter plots with line of best fit for all pairs of numeric columns
pairs = [(numeric_cols[i], numeric_cols[j]) 
         for i in range(len(numeric_cols)) 
         for j in range(i + 1, len(numeric_cols))]

fig, axes = plt.subplots(4, 3, figsize=(18, 15))  # Increased figure size
axes = axes.flatten()

for ax, (x_col, y_col) in zip(axes, pairs):
    sns.regplot(x=df[x_col], y=df[y_col], ax=ax, line_kws={'color':'red'})
    ax.set_title(f'{x_col} vs {y_col}', fontsize=8)
    ax.set_xlabel(x_col, fontsize=7)
    ax.set_ylabel(y_col, fontsize=7)
    ax.tick_params(axis='both', labelsize=6)

for ax in axes[len(pairs):]:
    ax.set_visible(False)

plt.tight_layout(pad=3.0)  # More padding to prevent overlap
plt.show()

# Bar plots for categorical vs numeric columns
pairs = [(cat_col, num_col) for cat_col in categorical_cols for num_col in numeric_cols]
fig, axes = plt.subplots(6, 3, figsize=(18, 28))  # Increased figure size
axes = axes.flatten()

for ax, (cat_col, num_col) in zip(axes, pairs):
    sns.barplot(
        x=df[cat_col], 
        y=df[num_col], 
        estimator='mean', 
        errorbar=None, 
        ax=ax, 
        palette='Set2'  # Colorful bars
    )
    ax.set_title(f'{cat_col} vs {num_col}', fontsize=8)
    ax.set_xlabel(cat_col, fontsize=9)
    ax.set_ylabel(f'Mean {num_col}', fontsize=6)
    
    # Rotate and align x-axis labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=5)
    ax.tick_params(axis='y', labelsize=5)

    # Format numeric-like categorical columns as integers
    if df[cat_col].dtype.kind in 'fi':
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

for ax in axes[len(pairs):]:
    ax.set_visible(False)

# Add extra spacing between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.show()

# Pairplot of subject scores
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df[subject_cols], kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha':0.6, 's':40}})
plt.suptitle('Pairplot of Subject Scores', y=1.02)
plt.show()

# Top 10 students bar chart
top_students = df.nlargest(10, 'average_score')
top_students.plot(x='roll_no', y='average_score', kind='bar', figsize=(12,6), color='green')
plt.title('Top 10 Students by Average Score')
plt.ylabel('Average Score')
plt.show()

# Test preparation effect on all subjects
sns.boxplot(x='test_preparation_course', y='average_score', data=df)
plt.title('Average Score vs Test Preparation Course')
plt.show()

# Lunch type effect on all subjects
sns.boxplot(x='lunch', y='average_score', data=df) 
plt.title('Average Score vs Lunch Type')
plt.show()

# Parental education effect on all subjects
sns.boxplot(x='parental_level_of_education', y='average_score', data=df)
plt.title('Average Score vs Parental Level of Education')  
plt.xticks(rotation=45)
plt.show() 

# Grade distribution
sns.countplot(x='grade', data=df, order=sorted(df['grade'].unique()))
plt.title('Grade Distribution')
plt.show()

# Advanced correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[subject_cols + ['average_score']].corr(), annot=True, cmap='viridis')
plt.title('Advanced Correlation Heatmap')
plt.show()

# Additional Visualizations
# Histograms of subject scores
df[subject_cols].hist(bins=15, figsize=(12,8), layout=(2,2))
plt.suptitle('Subject Score Distributions')
plt.show()

# Boxplots of subject scores by gender
fig, axes = plt.subplots(2, 2, figsize=(12,10))
for ax, col in zip(axes.flatten(), subject_cols):
    sns.boxplot(x='gender', y=col, data=df, ax=ax)
    ax.set_title(f'{col} Scores by Gender')
plt.tight_layout()
plt.show()

# Scatter plots of subject scores
sns.pairplot(df, vars=subject_cols, hue='gender', diag_kind='kde', plot_kws={'alpha':0.6})
plt.suptitle('Scatter Plots of Subject Scores by Gender', y=1.02)
plt.show()

# Test Preparation effect
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x='test_preparation_course', y='average_score', data=df, palette='Set3', ax=ax)
ax.set_title("Average Score vs Test Preparation Course")
plt.tight_layout()
plt.show()

# Z-score anomalies detection
from scipy.stats import zscore

# Ensure average_score column is numeric
df['average_score'] = pd.to_numeric(df['average_score'], errors='coerce')

# Compute Z-scores safely (skip NaN)
df['average_zscore'] = zscore(df['average_score'], nan_policy='omit')

# Define anomaly threshold (commonly >3 SD)
threshold = 3

# Extract anomalies (very high OR very low)
anomalies = df[df['average_zscore'].abs() > threshold].copy()

# Add explanation tag
anomalies['anomaly_type'] = anomalies['average_zscore'].apply(
    lambda x: "Extremely Low Score" if x < 0 else "Extremely High Score"
)

# Print summary
print("Anomalies Detection Summary")
print(f"Total students: {len(df)}")
print(f"Anomalies detected: {len(anomalies)}")
print(f"Percentage: {round((len(anomalies)/len(df))*100, 3)}%")


# Gender differences (t-test)

subjects = ['math_score', 'reading_score', 'writing_score', 'science_score', 'total_score']

male = df[df['gender'] == 'male']
female = df[df['gender'] == 'female']

for subject in subjects:
    t_stat, p_val = ttest_ind(male[subject], female[subject], nan_policy='omit')
    print(f"\nGender vs {subject}: t={t_stat:.3f}, p={p_val:.4f}")
    print("Significant Difference" if p_val < 0.05 else "No Significant Difference")


# Race/ethnicity differences (ANOVA)

ethnic_groups = df['race_ethnicity'].unique()

for subject in subjects:
    groups = [df[df['race_ethnicity'] == g][subject] for g in ethnic_groups]
    f_stat, p_val = f_oneway(*groups)
    print(f"\nRace/Ethnicity vs {subject}: F={f_stat:.3f}, p={p_val:.4f}")
    print("Significant Difference" if p_val < 0.05 else "No Significant Difference")


# Parental education (ANOVA)

par_edu_levels = df['parental_level_of_education'].unique()

for subject in subjects:
    groups = [df[df['parental_level_of_education'] == level][subject] for level in par_edu_levels]
    f_stat, p_val = f_oneway(*groups)
    print(f"\nParental Education vs {subject}: F={f_stat:.3f}, p={p_val:.4f}")
    print("Significant Difference" if p_val < 0.05 else "No Significant Difference")


# Lunch type effect (t-test)

lunch_levels = df['lunch'].unique()

if len(lunch_levels) == 2:
    lunch1 = df[df['lunch'] == lunch_levels[0]]
    lunch2 = df[df['lunch'] == lunch_levels[1]]

    for subject in subjects:
        t_stat, p_val = ttest_ind(lunch1[subject], lunch2[subject], nan_policy='omit')
        print(f"\nLunch vs {subject}: t={t_stat:.3f}, p={p_val:.4f}")
        print("Significant Difference" if p_val < 0.05 else "No Significant Difference")


# Test preparation course effect (t-test)

prep_levels = df['test_preparation_course'].unique()

if len(prep_levels) == 2:
    prep1 = df[df['test_preparation_course'] == prep_levels[0]]
    prep2 = df[df['test_preparation_course'] == prep_levels[1]]

    for subject in subjects:
        t_stat, p_val = ttest_ind(prep1[subject], prep2[subject], nan_policy='omit')
        print(f"\nTest Prep vs {subject}: t={t_stat:.3f}, p={p_val:.4f}")
        print("Significant Difference" if p_val < 0.05 else "No Significant Difference")


# Chi-square test for categorical variables

categorical_pairs = [
    ('gender', 'grade'),
    ('race_ethnicity', 'grade'),
    ('parental_level_of_education', 'grade'),
    ('lunch', 'grade'),
    ('test_preparation_course', 'grade')
]

for var1, var2 in categorical_pairs:
    contingency = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square: {var1} vs {var2}: chi2={chi2:.3f}, p={p:.4f}")
    print("Variables Are Associated" if p < 0.05 else "No Association Between Variables")


# Pearson correlation for score pairs

score_pairs = [
    ('math_score', 'reading_score'),
    ('math_score', 'writing_score'),
    ('math_score', 'science_score'),
    ('reading_score', 'writing_score'),
    ('reading_score', 'science_score'),
    ('writing_score', 'science_score'),
    ('total_score', 'math_score'),
    ('total_score', 'reading_score'),
    ('total_score', 'writing_score'),
    ('total_score', 'science_score')
]

for x, y in score_pairs:
    corr, p_val = pearsonr(df[x], df[y])
    print(f"\nCorrelation {x} vs {y}: r={corr:.3f}, p={p_val:.4f}")
    print("Significant Correlation" if p_val < 0.05 else "No Significant Correlation")


