import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Recharger le dataset
data_path = 'MAL-anime.csv/MAL-anime.csv'
anime_data = pd.read_csv (data_path)

# Préparation des données
anime_data['combined_features'] = anime_data.apply (lambda row: f"{row['Title']} {row['Type']} {row['Aired']}", axis=1)

# Utilisation du TF-IDF Vectorizer pour convertir les caractéristiques textuelles en une matrice de caractéristiques
tfidf = TfidfVectorizer (stop_words='english')
tfidf_matrix = tfidf.fit_transform (anime_data['combined_features'])

# Calcul de la similarité cosinus entre tous les animes
cosine_sim = cosine_similarity (tfidf_matrix, tfidf_matrix)


# Fonction pour obtenir des recommandations basées sur le contenu
def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        # Trouver l'index de l'anime qui correspond au titre
        idx = anime_data[anime_data['Title'].str.contains (title, case=False)].index[0]

        # Obtenir les scores de similarité pour cet anime avec tous les autres animes
        sim_scores = list (enumerate (cosine_sim[idx]))

        # Trier les animes en fonction des scores de similarité
        sim_scores = sorted (sim_scores, key=lambda x: x[1], reverse=True)

        # Obtenir les scores des 10 animes les plus similaires
        sim_scores = sim_scores[1:11]

        # Obtenir les index des animes recommandés
        anime_indices = [i[0] for i in sim_scores]

        # Retourner les titres et les scores des animes recommandés
        return [(anime_data['Title'].iloc[i], anime_data['Score'].iloc[i]) for i in anime_indices]
    except IndexError:
        return [("Anime not found. Please try with a different title.", 0)]


# Fonctions d'évaluation
def evaluate_recommendations(recommended, relevant):
    y_true = [1 if item in relevant else 0 for item in recommended]
    y_score = [1 for _ in recommended]

    precision = precision_score (y_true, y_score, zero_division=0)
    recall = recall_score (y_true, y_score, zero_division=0)
    f1 = f1_score (y_true, y_score, zero_division=0)

    return precision, recall, f1




# Fonction interactive pour tester les recommandations et afficher les meilleurs scores
def interactive_recommendation_system():
    while True:
        user_input = input ("Enter an anime title (or 'exit' to quit): ")
        if user_input.lower () == 'exit':
            break
        else:
            recommendations = get_recommendations (user_input)
            print ("Recommendations:")
            recommended_titles = [rec[0] for rec in recommendations]
            for rec in recommendations:
                print (f"{rec[0]} - Score: {rec[1]}")

            # Liste d'animes pertinents pour 'conan' (exemple)
            if user_input.lower () == 'conan':
                relevant_anime = ['Detective Conan', 'Detective Conan vs. Wooo',
                                  'Detective Conan Magic File 5: Niigata - Tokyo Omiyage Capriccio']
            else:
                relevant_anime = ['Free!','Gambo']  # À définir pour d'autres titres

            # Calculer et afficher les métriques d'évaluation
            precision, recall, f1 = evaluate_recommendations (recommended_titles, relevant_anime)

            print (f"\nÉvaluation des recommandations pour '{user_input}':")
            print (f"Précision: {precision:.2f}")
            print (f"Rappel: {recall:.2f}")
            print (f"F1 Score: {f1:.2f}")

        print ("\n")


# Appel de la fonction interactive
#interactive_recommendation_system ()
