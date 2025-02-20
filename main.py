import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv("2022_Test_ML.csv")
df = df.drop(columns=['ind'])

# Display dataset information
print(df.info())
print(df.describe())

# Visualize feature distributions
sns.pairplot(df)
plt.show()

# Plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Split into features and target variables
X = df[['s_mt', 's_mq', 'd', 'h_p']]
y = df[['QW', 'DP']]

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

# Fit GridSearchCV to the training data
grid_search.fit(X_train_scaled, y_train)

# Display the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best R² Score:", grid_search.best_score_)

# Train the model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)

# Evaluate the model
r2_train = r2_score(y_train, y_pred_train, multioutput='uniform_average')
r2_test = r2_score(y_test, y_pred_test, multioutput='uniform_average')
print(f'R² (Train): {r2_train:.3f}')
print(f'R² (Test): {r2_test:.3f}')

# Analyze how R² changes with the number of training points
train_sizes = np.linspace(0.1, 0.99, 10)
r2_scores = []

for size in train_sizes:
    X_partial, _, y_partial, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
    X_partial_scaled = scaler.transform(X_partial)
    best_model.fit(X_partial_scaled, y_partial)
    y_pred_partial = best_model.predict(X_test_scaled)
    r2_scores.append(r2_score(y_test, y_pred_partial, multioutput='uniform_average'))

# Visualize the results
plt.plot(train_sizes * len(X_train), r2_scores, marker='o')
plt.xlabel("Number of Training Points")
plt.ylabel("R² on Test Set")
plt.title("R² vs. Number of Training Points")
plt.show()