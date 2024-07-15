import pandas as pd
import pickle
from surprise import Dataset, Reader


with open('best_svd_model.pkl', 'rb') as f:
    best_model = pickle.load(f)


manga_titles = pd.read_csv('manga_cleaned.csv')[['id', 'title']]
manga_dict = pd.Series(manga_titles.title.values, index=manga_titles.id).to_dict()

def get_manga_recommendations_collab(user_id, model=best_model, n_recommendations=10):

    manga_ids = manga_titles['id'].tolist()

    user_ratings = [(manga_id, model.predict(user_id, manga_id).est) for manga_id in manga_ids]


    user_ratings.sort(key=lambda x: x[1], reverse=True)

    # Get the top n recommendations
    top_recommendations = user_ratings[:n_recommendations]
    recommended_titles = [manga_dict[manga_id] for manga_id, _ in top_recommendations]

    return recommended_titles

