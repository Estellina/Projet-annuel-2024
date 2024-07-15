import pandas as pd
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Préparation du dataset
manga_cleaned = pd.read_csv('manga_cleaned.csv')


manga_cleaned = manga_cleaned[['id', 'title', 'synopsis']].dropna().reset_index(drop=True)


# BERT embeddings
def get_bert_embeddings(text_list):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []

    for text in text_list:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    return np.array(embeddings)



synopsis_embeddings = get_bert_embeddings(manga_cleaned['synopsis'].tolist())

# Calcul de similitude avec les KMEANS
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(synopsis_embeddings)

# Helper function
indices = pd.Series(manga_cleaned.index, index=manga_cleaned['title']).drop_duplicates()


def get_recommendations_BERT(title, model_knn=model_knn, n_recommendations=10):
    idx = indices[title]
    distances, indices_knn = model_knn.kneighbors([synopsis_embeddings[idx]], n_neighbors=n_recommendations + 1)
    recommended_titles = manga_cleaned['title'].iloc[indices_knn[0][1:]]
    return recommended_titles




