import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split


def train_model():
    # Charger les données de notation et de crédits des films
    ratings = pd.read_csv ('dataset_fillm/ratings.csv')  # Utiliser le dataset plus grand
    credits = pd.read_csv ('dataset_fillm/tmdb_5000_credits.csv')
    movies = pd.read_csv ('dataset_fillm/tmdb_5000_movies.csv')

    # Convertir les titres des films en minuscules pour la recherche insensible à la casse
    credits['title_lower'] = credits['title'].str.lower ()
    movies['title_lower'] = movies['title'].str.lower ()

    # Fusionner les données pour obtenir le titre et le genre des films
    data = pd.merge (ratings, credits[['movie_id', 'title', 'title_lower']], left_on='movieId', right_on='movie_id')
    movies = movies[['id', 'title', 'genres', 'title_lower']]

    # Extraire les genres des films
    movies['genres'] = movies['genres'].apply (lambda x: [d['name'] for d in eval (x)])

    reader = Reader (rating_scale=(0.5, 5))
    data = Dataset.load_from_df (data[['userId', 'movieId', 'rating']], reader)

    # Split data into train and test sets
    trainset, testset = train_test_split (data, test_size=0.2)

    svd = SVD ()
    svd.fit (trainset)

    return svd, movies, ratings, testset


def recommend_movies(svd, movies, ratings, user_id, movie_title):
    # Vérifier si l'utilisateur a déjà noté le film demandé
    movie_data = movies[movies['title_lower'] == movie_title.lower ()]

    if movie_data.empty:
        print (f"Le film '{movie_title}' n'a pas été trouvé dans la base de données.")
        return

    movie_id = movie_data['id'].iloc[0]
    user_ratings = ratings[ratings['userId'] == int (user_id)]

    if movie_id in user_ratings['movieId'].values:
        print (f"Vous avez déjà noté le film '{movie_title}'.")
        return

    # Prédire la note que l'utilisateur donnerait au film
    predicted_rating = svd.predict (int (user_id), int (movie_id)).est
    print (
        f"L'utilisateur avec l'ID {user_id} donnerait probablement une note de {predicted_rating:.2f} /5 au film '{movie_title}'.")

    # Recommander des films similaires en genre
    movie_genres = movie_data['genres'].iloc[0]
    similar_movies = movies[movies['genres'].apply (lambda x: set (movie_genres).issubset (set (x)))]

    # Calculer les notes prédites pour les films similaires et trier par note prédite
    recommendations = []
    for idx, row in similar_movies.iterrows ():
        similar_movie_id = row['id']
        similar_movie_title = row['title']
        similar_movie_rating = svd.predict (int (user_id), int (similar_movie_id)).est
        recommendations.append ((similar_movie_title, similar_movie_rating))

    # Trier les recommandations par note prédite (descendante) et afficher les 5 meilleures
    recommendations.sort (key=lambda x: x[1], reverse=True)
    top_5_recommendations = recommendations[:5]

    print ("Top 5 des films similaires  :")
    for title, rating in top_5_recommendations:
        print (f"Film: {title}, Note prédite: {rating:.2f}")


if __name__ == "__main__":
    svd, movies, ratings, testset = train_model ()
    user_registered = input ("Êtes-vous un utilisateur enregistré ? (oui/non) : ").strip ().lower ()
    if user_registered == 'oui':
        user_id = input ("Entrez l'ID de l'utilisateur : ").strip ()
        movie_title = input ("Entrez le nom du film : ").strip ()
        recommend_movies (svd, movies, ratings, user_id, movie_title)
    else:
        print ("Vous devez être un utilisateur enregistré pour utiliser ce système de recommandation.")
