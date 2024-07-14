import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
import pickle

def train_and_save_model():
    # Charger les données de notation et de crédits des films
    ratings = pd.read_csv('dataset_fillm/ratings.csv')  # Utiliser le dataset plus grand
    credits = pd.read_csv('dataset_fillm/tmdb_5000_credits.csv')
    movies = pd.read_csv('dataset_fillm/tmdb_5000_movies.csv')

    # Convertir les titres des films en minuscules pour la recherche insensible à la casse
    credits['title_lower'] = credits['title'].str.lower()
    movies['title_lower'] = movies['title'].str.lower()

    # Fusionner les données pour obtenir le titre et le genre des films
    data = pd.merge(ratings, credits[['movie_id', 'title', 'title_lower']], left_on='movieId', right_on='movie_id')
    movies = movies[['id', 'title', 'genres', 'title_lower']]

    # Extraire les genres des films
    movies['genres'] = movies['genres'].apply(lambda x: [d['name'] for d in eval(x)])

    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

    # Split data into train and test sets
    trainset, testset = train_test_split(data, test_size=0.2)

    svd = SVD()
    svd.fit(trainset)

    # Sauvegarder le modèle et les données
    with open('svd_model.pkl', 'wb') as model_file:
        pickle.dump(svd, model_file)
    with open('movies.pkl', 'wb') as movies_file:
        pickle.dump(movies, movies_file)
    with open('ratings.pkl', 'wb') as ratings_file:
        pickle.dump(ratings, ratings_file)
    with open('testset.pkl', 'wb') as testset_file:
        pickle.dump(testset, testset_file)

if __name__ == "__main__":
    train_and_save_model()
