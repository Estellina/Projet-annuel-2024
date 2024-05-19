import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Creation de dataframes pour les fichiers csv
credits = pd.read_csv('tmdb_5000_credits.csv')
movies_metadata = pd.read_csv('tmdb_5000_movies.csv')

# Jointure des dataframe sur le champ ID
movies = movies_metadata.merge(credits, left_on='id', right_on='movie_id')
# Observation des 5 premières lignes
#print(movies.head(5))

# Premier Filtre - Content Based

def combine_features(row):
    try:
        return row['cast'] + " " + row['crew'] + " " + row['keywords'] + " " + row['genres']
    except:
        return " "

# Prendre les champs important et enlevés les valeurs vides
for feature in ['cast', 'crew', 'keywords', 'genres']:
    movies[feature] = movies[feature].fillna('')

# Faire une nouvelle colonne pour les infos combiné
movies['combined_features'] = movies.apply(combine_features, axis=1)


# Transformations des données textes en vecteurs pour analyses
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies['combined_features'])

# Calculer le taux de similarité
cosine_sim = cosine_similarity(count_matrix)


# Pour obtenir le titre de l'index donné
def get_title_from_index(index):
    return movies.iloc[index]['original_title']


def get_index_from_title(title):
    return movies[movies.original_title == title].index[0]

# Fonction de recommendation selon un film similaire
def recommend_movies(movie_title, num_recommendations=1):
    movie_index = get_index_from_title(movie_title)
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]

    print(f"Top {num_recommendations} movies similar to '{movie_title}':\n")
    for i, element in sorted_similar_movies:
        print(get_title_from_index(i))



recommend_movies('The Dark Knight', 10)


