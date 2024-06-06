# Load libraries
import pandas as pd              # For data manipulation and analysis
import numpy as np               # For numerical computing
from datetime import datetime
import scipy.stats as stats      # For statistical analysis
import math
import matplotlib                # For plotting and visualization
import matplotlib.pyplot as plt  
from pandas.plotting import parallel_coordinates
import seaborn as sns            # For statistical data visualization
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

# For machine learning
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor,GradientBoostingRegressor,VotingRegressor
from sklearn.model_selection import KFold
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

# Load data
df_train = pd.read_csv('/kaggle/input/playground-series-s4e5/train.csv',index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e5/sample_submission.csv',index_col='id')

# EDA
df_train.info()
df_test.info()
df_train.nunique()
df_train.describe().T.style

# Distribution of each features
sns.histplot(data = df_train['FloodProbability'], bins = 20)

fig, axes = plt.subplots(5, 4, figsize=(20, 25))
 
for i, column in enumerate(df_train.columns):
    if column == 'FloodProbability':
        continue
    plt.subplots_adjust(top = 0.85)
    ax = sns.histplot(data = df_train, 
                x = column, 
                bins = df_train[column].nunique(),
                ax = axes[i // 4, i % 4])
    
    ax.set_yticklabels(['{:,.0f}K'.format(ticks / 1000) for ticks in ax.get_yticks()])
fig.tight_layout(h_pad = 2)
plt.subplots_adjust(top = 0.95)
plt.suptitle('Distribution of Flood Predictors', fontsize = 14)
plt.show()

# Correlation Matrix Heatmap
fig, ax = plt.subplots(figsize = (12,10))
corr = df_train.corr()
hm = sns.heatmap(corr,
                annot = True,
                ax = ax,
                cmap = 'Blues',
                fmt = '.2f')
fig.subplots_adjust(top = 0.95)
plt.suptitle('Flood Predictors Correlation Heatmap', fontsize = 14)
plt.show()

# Scatterplots between target and each predictor
fig, axes = plt.subplots(5, 4, figsize=(20, 25))

for i, column in enumerate(df_train.columns):
    if column == 'FloodProbability':
        continue
    temp_df = df_train[['FloodProbability', column]].groupby(column).mean().reset_index()
    plt.subplots_adjust(top = 0.85)
    ax = sns.scatterplot(data = temp_df,
                y = column,
                x = 'FloodProbability',
                ax = axes[i // 4, i % 4])

fig.tight_layout(h_pad = 2)
fig.subplots_adjust(top = 0.97)
plt.suptitle('Linearity between each of Predictors and Flood Probability', fontsize = 14)
plt.show()

# Feature Engineering
BASE_FEATURES = df_test.columns

def add_features(df):
    df['total'] = df[BASE_FEATURES].sum(axis=1)
    df['mean'] = df[BASE_FEATURES].mean(axis=1)
    df['std'] = df[BASE_FEATURES].std(axis=1)
    df['max'] = df[BASE_FEATURES].max(axis=1)
    df['min'] = df[BASE_FEATURES].min(axis=1)
    df['median'] = df[BASE_FEATURES].median(axis=1)
    df['ptp'] = df[BASE_FEATURES].values.ptp(axis=1)
    df['q25'] = df[BASE_FEATURES].quantile(0.25, axis=1)
    df['q75'] = df[BASE_FEATURES].quantile(0.75, axis=1)
    
    df['ClimateImpact'] = df['MonsoonIntensity'] + df['ClimateChange']
    df['AnthropogenicPressure'] = df['Deforestation'] + df['Urbanization'] + df['AgriculturalPractices'] + df['Encroachments']
    df['InfrastructureQuality'] = df['DamsQuality'] + df['DrainageSystems'] + df['DeterioratingInfrastructure']
    df['CoastalVulnerabilityTotal'] = df['CoastalVulnerability'] + df['Landslides']
    df['PreventiveMeasuresEfficiency'] = df['RiverManagement'] + df['IneffectiveDisasterPreparedness'] + df['InadequatePlanning']
    df['EcosystemImpact'] = df['WetlandLoss'] + df['Watersheds']
    df['SocioPoliticalContext'] = df['PopulationScore'] * df['PoliticalFactors']


    df['FloodVulnerabilityIndex'] = (df['AnthropogenicPressure'] + df['InfrastructureQuality'] +
                                     df['CoastalVulnerabilityTotal'] + df['PreventiveMeasuresEfficiency']) / 4
    
    df['PopulationDensityImpact'] = df['PopulationScore'] * (df['Urbanization'] + df['Encroachments'])
    
    df['DeforestationUrbanizationRatio'] = df['Deforestation'] / df['Urbanization']
    
    df['AgriculturalEncroachmentImpact'] = df['AgriculturalPractices'] * df['Encroachments']
    
    df['DamDrainageInteraction'] = df['DamsQuality'] * df['DrainageSystems']
    
    df['LandslideSiltationInteraction'] = df['Landslides'] * df['Siltation']
    
    df['WatershedWetlandRatio'] = df['Watersheds'] / df['WetlandLoss']
    
    df['PoliticalPreparednessInteraction'] = df['PoliticalFactors'] * df['IneffectiveDisasterPreparedness']
    
    
    df['TopographyDrainageSiltation'] = df['TopographyDrainage'] + df['Siltation']
    
    df['ClimateAnthropogenicInteraction'] = df['ClimateImpact'] * df['AnthropogenicPressure']
    
    df['InfrastructurePreventionInteraction'] = df['InfrastructureQuality'] * df['PreventiveMeasuresEfficiency']
    
    df['CoastalEcosystemInteraction'] = df['CoastalVulnerabilityTotal'] * df['EcosystemImpact']

    return df

df_train = add_features(df_train)
df_test = add_features(df_test)

# Data Preparation
target = 'FloodProbability'
features = [col for col in df_train.columns if col != target ]
X = df_train[features]
y = df_train[target]

for column in X.columns:
    X[column].replace([np.inf, -np.inf], np.nan, inplace = True)
    mean = X[column].mean(skipna=True)
    X[column].fillna(mean, inplace = True)
X.isnull().any().sum()

# Multiple Linear Regression
lm = LinearRegression()

# Fit the data(train the model) 
lm.fit(X, y) 

# Predict
y_predicted = lm.predict(X)

# Model evaluation
r2 = r2_score(y, y_predicted) 

# printing values 
print('Slope:' ,lm.coef_) 
print('Intercept:', lm.intercept_) 
print('R2 score: ', r2) 


############################## XGBoost ##############################
# Create XGBoost model
xgb = XGBRegressor(booster = 'gbtree',
                   max_depth = 10,
                   num_leaves = 250,
                   reg_alpha = 0.1,
                   reg_lambda = 3.25,
                   learning_rate = 0.01,
                   n_estimators = 3000,
                   subsample_for_bin= 165700, 
                   min_child_samples= 114, 
                   colsample_bytree= 0.9634,
                   subsample= 0.9592, 
                   random_state = 0)

n_splits = 5
# Create a KFold cross-validator
kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)

scores = []
# Perform K-Fold Cross-Validation
for train_index, val_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
    y_train, y_valid = y[train_index], y[val_index]
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_valid)
    score = r2_score(y_valid, y_pred)
    print(score)
    scores.append(score)

# Output the average R2 score across all folds
print(f'Mean R2 score: {np.mean(scores):.5f}')

#####################################################################

############################## CatBoost #############################
# Create CatBoostRegressor model
catb = CatBoostRegressor(n_estimators = 3000,
                       learning_rate = 0.05,
                       verbose = 0)

n_splits = 5
# Create a KFold cross-validator
kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)

scores = []
# Perform K-Fold Cross-Validation
for train_index, val_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
    y_train, y_valid = y[train_index], y[val_index]
    catb.fit(X_train, y_train)
    y_pred = catb.predict(X_valid)
    score = r2_score(y_valid, y_pred)
    print(score)
    scores.append(score)

# Output the average R2 score across all folds
print(f'Mean R2 score: {np.mean(scores):.5f}')

#####################################################################

############################## LGBM #################################
n_splits = 5
# Create a KFold cross-validator
kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
# Create LGBM model
lgbm = LGBMRegressor(objective = 'regression',
               boosting_type = 'gbdt',
               max_depth = 10,
               num_leaves = 250,
               reg_alpha = 0.1,
               reg_lambda = 3.25,
               learning_rate = 0.01,
               n_estimators = 3000,
               subsample_for_bin= 165700, 
               min_child_samples= 114, 
               colsample_bytree= 0.9634,
               subsample= 0.9592, 
               random_state = 0,
               verbosity = -1)
scores = []
# Perform K-Fold Cross-Validation
for train_index, val_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
    y_train, y_valid = y[train_index], y[val_index]
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_valid)
    score = r2_score(y_valid, y_pred)
    print(score)
    scores.append(score)

# Output the average R2 score across all folds
print(f'Mean R2 score: {np.mean(scores):.5f}')
#####################################################################

# Ensemble
r1 = catb
r2 = xgb
#r3 = lgbm
r4 = HistGradientBoostingRegressor(learning_rate = 0.05,
                                   max_iter = 400)
r5 = GradientBoostingRegressor(learning_rate = 0.05,
                               n_estimators = 400)
r6 = RandomForestRegressor(n_estimators = 400,
                           max_depth = 4)
r7 = LinearRegression()
r8 = SVR(kernel='linear')

stack = StackingCVRegressor(regressors=(r1, r2, r4, r5, r6, r7, r8),
                            meta_regressor = CatBoostRegressor(verbose = 0),
                            cv = KFold(n_splits=10))

stack.fit(X, y)
X_submission = df_test[features]
y_submission_pred = stack.predict(X_submission)
df_test.reset_index(inplace = True)
submission = pd.DataFrame({
    "id": df_test["id"],
    "probability": y_submission_pred,
}).set_index('id')

submission.to_csv("./submission.csv")
