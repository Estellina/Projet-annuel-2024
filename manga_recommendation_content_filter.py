import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


manga_cleaned = pd.read_csv('manga_cleaned.csv')

# Préparation des données
manga_cleaned = manga_cleaned[['id', 'title', 'synopsis']].dropna().reset_index(drop=True)

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(manga_cleaned['synopsis'])

# Calcul de similarité
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Helper function
indices = pd.Series(manga_cleaned.index, index=manga_cleaned['title']).drop_duplicates()

def get_recommendations_from_prompt(prompt, n_recommendations=10):
    # Vérification du prmpt
    if prompt in indices.index:
        idx = indices[prompt]
        sim_scores = list(enumerate(cosine_sim[idx]))
    else:
        # Vectorisation
        prompt_tfidf = tfidf.transform([prompt])
        sim_scores = list(enumerate(linear_kernel(prompt_tfidf, tfidf_matrix)[0]))

    # Assemblé et trié les score de similitudes
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:n_recommendations]
    manga_indices = [i[0] for i in sim_scores]
    return manga_cleaned['title'].iloc[manga_indices]


