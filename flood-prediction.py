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

# Data Preparation
target = 'FloodProbability'
features = [col for col in df_train.columns if col != target ]
X = df_train[features]
y = df_train[target]

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
n_splits = 5
# Create a KFold cross-validator
kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)

scores = []
xgb_params = {
    'n_jobs': -1,
    'max_depth': 15,
    'max_leaves': 51,
    'n_estimators': 1000,
    'random_state': 42,
    'objective': 'reg:gamma',
    'grow_policy': 'depthwise',
    'gamma': 0.001191175583365525,
    'reg_alpha': 0.4922409840555407,
    'subsample': 0.9043911969552909,
    'reg_lambda': 0.2006103666827618,
    'max_delta_step': 0.5187236006765079,
    'learning_rate': 0.031068537109748533,
    'colsample_bynode': 0.9056076202576685,
    'min_child_weight': 0.1519636306480494,
    'colsample_bytree': 0.8136171314595549,
    'colsample_bylevel': 0.8469915838866402,
}
# Perform K-Fold Cross-Validation
for train_index, val_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
    y_train, y_valid = y[train_index], y[val_index]
    # Create XGBoost model
    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_valid)
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

scores = []
# Perform K-Fold Cross-Validation
for train_index, val_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
    y_train, y_valid = y[train_index], y[val_index]
    # Create XGBoost model
    lgbm = LGBMRegressor(objective = 'regression',
                   boosting_type = 'gbdt',
                   max_depth = 7,
                   reg_alpha = 0.1,
                   reg_lambda = 3.25,
                   learning_rate = 0.05,
                   n_estimators = 3000)
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_valid)
    score = r2_score(y_valid, y_pred)
    print(score)
    scores.append(score)

# Output the average R2 score across all folds
print(f'Mean R2 score: {np.mean(scores):.5f}')
#####################################################################

############################## CatBoost #############################
# Create CatBoostRegressor model
catb = CatBoostRegressor(n_estimators = 1000,
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
    y_pred = xgb.predict(X_valid)
    score = r2_score(y_valid, y_pred)
    print(score)
    scores.append(score)

# Output the average R2 score across all folds
print(f'Mean R2 score: {np.mean(scores):.5f}')

#####################################################################

# Ensemble
X_submission = df_test[features]
X_submission = df_test[features]
y_xgb_pred = xgb.predict(X_submission)
y_lgb_pred = lgb.predict(X_submission)
y_lm_pred = lm.predict(X_submission)
y_cat_pred = catb.predict(X_submission)

y_submission_pred = 0.4*y_xgb_pred + 0.3*y_lgb_pred + 0.3*y_cat_pred
submission = pd.DataFrame({
    "id": df_test["id"],
    "probability": y_submission_pred,
}).set_index('id')

submission.to_csv("./submission.csv")