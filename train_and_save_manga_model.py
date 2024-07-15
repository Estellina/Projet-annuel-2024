import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import RandomizedSearchCV
import pickle
from scipy.stats import uniform

# Load the ratings dataset
ratings = pd.read_csv('manga_ratinga.csv')

# Data preparation
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['user', 'manga_id', 'score']], reader)

# Define the parameter distribution
param_dist = {
    'n_factors': [20, 50, 100],
    'n_epochs': [10, 20, 30],
    'lr_all': uniform(0.002, 0.01),  # uniform distribution between 0.002 and 0.012
    'reg_all': uniform(0.02, 0.1)    # uniform distribution between 0.02 and 0.12
}

# Perform randomized search
rs = RandomizedSearchCV(SVD, param_dist, measures=['rmse'], cv=3, n_iter=20, n_jobs=-1, random_state=42)
rs.fit(data)

# Best model
best_model = rs.best_estimator['rmse']

# Train the best model on the entire dataset
trainset = data.build_full_trainset()
best_model.fit(trainset)

# Save the best model to a file
with open('best_svd_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Model training and saving completed.")

