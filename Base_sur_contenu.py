import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the datasets

tmdb_5000_credits = pd.read_csv('dataset_fillm/tmdb_5000_credits.csv')
tmdb_5000_movies = pd.read_csv('dataset_fillm/tmdb_5000_movies.csv')
movies_metadata = pd.read_csv('dataset_fillm/movies_metadata.csv', low_memory=False)

# Preprocess the data
movies_metadata['genres'] = movies_metadata['genres'].apply(lambda x: ' '.join([i['name'] for i in eval(x)]) if pd.notnull(x) else '')
tmdb_5000_movies['genres'] = tmdb_5000_movies['genres'].apply(lambda x: ' '.join([i['name'] for i in eval(x)]))
tmdb_5000_movies['keywords'] = tmdb_5000_movies['keywords'].apply(lambda x: ' '.join([i['name'] for i in eval(x)]))
tmdb_5000_movies['cast'] = tmdb_5000_credits['cast'].apply(lambda x: ' '.join([i['name'] for i in eval(x)]))
tmdb_5000_movies['crew'] = tmdb_5000_credits['crew'].apply(lambda x: ' '.join([i['name'] for i in eval(x) if i['job'] == 'Director']))

# Combine relevant features into a single text string for each movie
tmdb_5000_movies['combined_features'] = (
    tmdb_5000_movies['title'] + ' ' +
    tmdb_5000_movies['overview'].fillna('') + ' ' +
    tmdb_5000_movies['genres'] + ' ' +
    tmdb_5000_movies['keywords'] + ' ' +
    tmdb_5000_movies['cast'] + ' ' +
    tmdb_5000_movies['crew']
)

# Extract relevant features from movies_metadata and combine them
movies_metadata['combined_features'] = (
    movies_metadata['title'].fillna('') + ' ' +
    movies_metadata['overview'].fillna('') + ' ' +
    movies_metadata['genres'] + ' ' +
    movies_metadata['tagline'].fillna('')
)

# Concatenate the combined features from both datasets
all_movies = pd.concat([tmdb_5000_movies[['id', 'title', 'combined_features']], movies_metadata[['id', 'title', 'combined_features']]], ignore_index=True).drop_duplicates(subset='id')

# Define a TF-IDF Vectorizer Object. Remove all English stop words

tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
all_movies['combined_features'] = all_movies['combined_features'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(all_movies['combined_features'])




def recommend_movies_based_on_prompt(prompt, tfidf_matrix=tfidf_matrix, movies_df=all_movies, top_n=10):
    # Vectorize the user prompt
    user_tfidf = tfidf.transform([prompt])

    # Compute the cosine similarity between the user prompt and all movies
    cosine_sim = linear_kernel(user_tfidf, tfidf_matrix).flatten()

    # Get the top n movie indices based on the similarity scores
    top_movie_indices = cosine_sim.argsort()[-top_n:][::-1]

    # Get the top n recommended movies
    recommended_movies = movies_df.iloc[top_movie_indices][['title', 'combined_features']]

    return recommended_movies


