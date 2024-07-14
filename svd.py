import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Charger les données
tmdb_5000_credits = pd.read_csv ('dataset_fillm/tmdb_5000_credits.csv')
tmdb_5000_movies = pd.read_csv ('dataset_fillm/tmdb_5000_movies.csv')
movies_metadata = pd.read_csv ('dataset_fillm/movies_metadata.csv', low_memory=False)
ratings = pd.read_csv ('dataset_fillm/ratings.csv')

# Prétraiter les données
tmdb_5000_movies['genres'] = tmdb_5000_movies['genres'].apply (lambda x: ' '.join ([i['name'] for i in eval (x)]))
tmdb_5000_movies['keywords'] = tmdb_5000_movies['keywords'].apply (lambda x: ' '.join ([i['name'] for i in eval (x)]))
tmdb_5000_movies['cast'] = tmdb_5000_credits['cast'].apply (lambda x: ' '.join ([i['name'] for i in eval (x)]))
tmdb_5000_movies['crew'] = tmdb_5000_credits['crew'].apply (
    lambda x: ' '.join ([i['name'] for i in eval (x) if i['job'] == 'Director']))
tmdb_5000_movies['combined_features'] = (
        tmdb_5000_movies['title'] + ' ' +
        tmdb_5000_movies['overview'].fillna ('') + ' ' +
        tmdb_5000_movies['genres'] + ' ' +
        tmdb_5000_movies['keywords'] + ' ' +
        tmdb_5000_movies['cast'] + ' ' +
        tmdb_5000_movies['crew']
)
movies_metadata['genres'] = movies_metadata['genres'].apply (
    lambda x: ' '.join ([i['name'] for i in eval (x)]) if pd.notnull (x) else '')
movies_metadata['combined_features'] = (
        movies_metadata['title'].fillna ('') + ' ' +
        movies_metadata['overview'].fillna ('') + ' ' +
        movies_metadata['genres'] + ' ' +
        movies_metadata['tagline'].fillna ('')
)
all_movies = pd.concat (
    [tmdb_5000_movies[['id', 'title', 'combined_features']], movies_metadata[['id', 'title', 'combined_features']]],
    ignore_index=True).drop_duplicates (subset='id')

# TF-IDF Vectorizer
tfidf = TfidfVectorizer (stop_words='english')
all_movies['combined_features'] = all_movies['combined_features'].fillna ('')
tfidf_matrix = tfidf.fit_transform (all_movies['combined_features'])

# Sauvegarder les modèles
with open ('tfidf_matrix.pkl', 'wb') as tfidf_file:
    pickle.dump (tfidf_matrix, tfidf_file)


def recommend_hybrid(user_id, movie_title, svd, tfidf_matrix, movies, top_n=10):
    # Recommandations basées sur le contenu
    movie_data = movies[movies['title'].str.lower () == movie_title.lower ()]
    if movie_data.empty:
        print (f"Le film '{movie_title}' n'a pas été trouvé dans la base de données.")
        return []

    movie_idx = movie_data.index[0]
    cosine_sim = linear_kernel (tfidf_matrix[movie_idx], tfidf_matrix).flatten ()
    content_recs = cosine_sim.argsort ()[-top_n:][::-1]

    # Recommandations de filtrage collaboratif
    collaborative_recs = []
    user_ratings = ratings[ratings['userId'] == int (user_id)]
    if not user_ratings.empty:
        for movie_id in user_ratings['movieId']:
            predicted_rating = svd.predict (int (user_id), int (movie_id)).est
            collaborative_recs.append ((movie_id, predicted_rating))
        collaborative_recs = sorted (collaborative_recs, key=lambda x: x[1], reverse=True)[:top_n]

    # Fusionner les recommandations
    hybrid_recs = []
    for idx in content_recs:
        if idx < len (movies):
            movie_id = movies.iloc[idx]['id']
            title = movies.iloc[idx]['title']
            rating = svd.predict (int (user_id), int (movie_id)).est
            hybrid_recs.append ((title, rating))

    hybrid_recs = sorted (hybrid_recs, key=lambda x: x[1], reverse=True)[:top_n]
    return hybrid_recs
