# # import libraries
# import pandas as pd
# import warnings
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import pickle

# # Suppress Warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# pd.options.mode.chained_assignment = None  

# # read the data file
# df = pd.read_csv('data/energy_complete.csv')

# # see the first few rows of the data
# df.head()


# # ## Data Cleaning
# # Handling missing values
# # Remove rows with missing target values
# df = df.dropna(subset=['RRP'])

# # Fill missing numerical features with median values
# numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
# for col in numeric_cols:
#     df[col].fillna(df[col].median(), inplace=True)

# # Fill missing categorical features with mode values
# categorical_cols = df.select_dtypes(include=['object']).columns
# for col in categorical_cols:
#     df[col].fillna(df[col].mode()[0], inplace=True)

# # Removing duplicates
# df = df.drop_duplicates()

# # convert the date column to datetime
# df['month'] = pd.to_datetime(df['date']).dt.month

# # ## Machine Learning Model to Predict RRP
# # Feature Selection and Preprocessing
# features = ['demand', 'demand_pos_RRP', 'demand_neg_RRP', 'min_temperature', 'max_temperature', 'solar_exposure', 'rainfall', 'frac_at_neg_RRP', 'month', 'school_day']
# X = df[features]
# y = df['RRP']

# # Handling categorical variables and missing values
# X['school_day'] = X['school_day'].map({'Y': 1, 'N': 0})

# # Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Data preprocessing pipeline
# numeric_features = ['demand', 'demand_pos_RRP', 'demand_neg_RRP', 'min_temperature', 'max_temperature', 'solar_exposure', 'rainfall', 'frac_at_neg_RRP', 'month']
# categorical_features = ['school_day']

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numeric_features),
#         ('cat', 'passthrough', categorical_features)
#     ]
# )

# # Define models to train
# models = {
#     'Linear Regression': LinearRegression(),
#     'Random Forest': RandomForestRegressor(random_state=42),
#     'Support Vector Regressor': SVR()
# }

# # Cross-validation and hyperparameter tuning
# best_model = None
# best_score = -np.inf

# for model_name, model in models.items():
#     # Create pipeline with preprocessing and model
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                ('model', model)])
    
#     # Performing cross-validation
#     cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
#     print(f"{model_name} Cross-Validation R^2 Scores: {cv_scores}")
#     print(f"{model_name} Mean R^2 Score: {cv_scores.mean()}\n")
    
#     # Hyperparameter tuning using GridSearchCV (example for RandomForest and SVR)
#     if model_name == 'Random Forest':
#         param_grid = {'model__n_estimators': [50, 100, 150], 'model__max_depth': [None, 10, 20]}
#         grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
#         grid_search.fit(X_train, y_train)
#         model = grid_search.best_estimator_
#         print(f"Best Parameters for Random Forest: {grid_search.best_params_}")
#     elif model_name == 'Support Vector Regressor':
#         param_grid = {'model__C': [0.1, 1, 10], 'model__kernel': ['rbf', 'linear']}
#         grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
#         grid_search.fit(X_train, y_train)
#         model = grid_search.best_estimator_
#         print(f"Best Parameters for SVR: {grid_search.best_params_}")
#     else:
#         pipeline.fit(X_train, y_train)

#      # Train the model with the best parameters
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     score = r2_score(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     print(f"{model_name} Test R^2 Score: {score}")
#     print(f"{model_name} Test RMSE: {rmse}\n")
    
#     # Track the best model
#     if score > best_score:
#         best_score = score
#         best_model = pipeline

# print(f"The best model is: {best_model.named_steps['model']} with R^2 Score: {best_score}")

# # Final Model Evaluation
# y_final_pred = best_model.predict(X_test)
# mse = mean_squared_error(y_test, y_final_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_final_pred)

# print(f"Final Model Mean Squared Error: {mse}")
# print(f"Final Model RMSE: {rmse}")
# print(f"Final Model R^2 Score: {r2}")

# # Save both the model and the preprocessor to .bin files using pickle
# with open('model.bin', 'wb') as model_file:
#     pickle.dump(model, model_file)
#     print("\nModel has been saved to 'model.bin'")

# with open('preprocessor.bin', 'wb') as preprocessor_file:
#     pickle.dump(preprocessor, preprocessor_file)
#     print("Preprocessor has been saved to 'preprocessor.bin'")


# import libraries
import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Suppress Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  

# Read the data file
df = pd.read_csv('data/energy_complete.csv')

# Data Cleaning
# Handling missing values
# Remove rows with missing target values
df = df.dropna(subset=['RRP'])

# Fill missing numerical features with median values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))

# Fill missing categorical features with mode values
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Removing duplicates
df = df.drop_duplicates()

# Convert the date column to datetime
df['month'] = pd.to_datetime(df['date']).dt.month

# Feature Selection and Preprocessing
features = ['demand', 'demand_pos_RRP', 'demand_neg_RRP', 'min_temperature', 'max_temperature', 'solar_exposure', 'rainfall', 'frac_at_neg_RRP', 'month', 'school_day']
X = df[features]
y = df['RRP']

# Handling categorical variables and missing values
X = X.copy()  # Prevent SettingWithCopyWarning
X['school_day'] = X['school_day'].map({'Y': 1, 'N': 0})

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing and Modeling
scaler = StandardScaler()
numeric_features = ['demand', 'demand_pos_RRP', 'demand_neg_RRP', 'min_temperature', 'max_temperature', 'solar_exposure', 'rainfall', 'frac_at_neg_RRP']
preprocessor = ColumnTransformer([('scaler', scaler, numeric_features)], remainder='passthrough')

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Support Vector Regressor': SVR()
}

best_model = None
best_score = -np.inf

for model_name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    if model_name == 'Random Forest':
        param_grid = {'model__n_estimators': [50, 100, 150], 'model__max_depth': [None, 10, 20]}
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', error_score='raise')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best Parameters for Random Forest: {grid_search.best_params_}")
    elif model_name == 'Support Vector Regressor':
        param_grid = {'model__C': [0.1, 1, 10], 'model__kernel': ['rbf', 'linear']}
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', error_score='raise')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best Parameters for SVR: {grid_search.best_params_}")
    else:
        pipeline.fit(X_train, y_train)
        model = pipeline

    # Evaluate the model
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{model_name} Test R^2 Score: {score}")
    print(f"{model_name} Test RMSE: {rmse}\n")
    
    # Track the best model
    if score > best_score:
        best_score = score
        best_model = model

print(f"The best model is: {best_model.named_steps['model']} with R^2 Score: {best_score}")

# Final Model Evaluation
y_final_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_final_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_final_pred)

print(f"Final Model Mean Squared Error: {mse}")
print(f"Final Model RMSE: {rmse}")
print(f"Final Model R^2 Score: {r2}")

# Save the best_model pipeline to a .bin file using pickle
with open('best_model.bin', 'wb') as model_file:
    pickle.dump(best_model, model_file)
    print("\nBest Model has been saved to 'best_model.bin'")
