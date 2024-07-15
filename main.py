import pickle
import torch
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from Base_sur_contenu import recommend_movies_based_on_prompt
from filtrage_collaboratif import recommend_movies
from manga_recommendation_BERT import get_recommendations_BERT
from manga_recommendation_collab_filtering import get_manga_recommendations_collab
from manga_recommendation_content_filter import get_recommendations_from_prompt
from sklearn.neighbors import NearestNeighbors
from transformers import BertTokenizer, BertModel

from svd import recommend_hybrid

# Importer les fonctions de recommandation d'anime
from anime_BC import get_recommendations

console = Console()

def load_models():
    with open('svd_model.pkl', 'rb') as model_file:
        svd = pickle.load(model_file)
    with open('movies.pkl', 'rb') as movies_file:
        movies = pickle.load(movies_file)
    with open('ratings.pkl', 'rb') as ratings_file:
        ratings = pickle.load(ratings_file)
    with open('tfidf_matrix.pkl', 'rb') as tfidf_file:
        tfidf_matrix = pickle.load(tfidf_file)
    with open('best_svd_model.pkl', 'rb') as manga_file:
        manga_svd = pickle.load(manga_file)
    return svd, movies, ratings, tfidf_matrix, manga_svd

def display_recommendations(recommendations, title="Recommandations"):
    table = Table(title=f"[bold blue]{title}[/bold blue]")
    table.add_column("Titre", justify="left", style="cyan", no_wrap=True)
    table.add_column("Note ", justify="left", style="magenta")

    for rec in recommendations:
        if isinstance(rec[1], float):
            table.add_row(rec[0], f"{rec[1]:.2f}")
        else:
            table.add_row(rec[0], str(rec[1]))

    console.print(table)

def main():
    console.print("[bold green]Bienvenue au système de recommandation de films et d'animes ![/bold green]")

    svd, movies, ratings, tfidf_matrix, manga_svd = load_models()

    while True:
        category = Prompt.ask(
            "[bold yellow]Voulez-vous une recommandation de film ou d'anime ?[/bold yellow]",
            choices=["film", "anime", "quitter"],
            default="quitter"
        )

        if category == "quitter":
            console.print("[bold red]Merci d'avoir utilisé notre système de recommandation ![/bold red]")
            break

        if category == "film":
            option = Prompt.ask(
                "[bold yellow]Quelle méthode de recommandation souhaitez-vous utiliser ?[/bold yellow]",
                choices=["filtrage_collaboratif", "base_sur_le_contenu", "hybride", "retour"],
                default="retour"
            )

            if option == "retour":
                continue

            if option == "filtrage_collaboratif":
                user_registered = Prompt.ask("[bold yellow]Êtes-vous un utilisateur enregistré ?[/bold yellow]", choices=["oui", "non"], default="non")
                if user_registered == 'oui':
                    user_id = Prompt.ask("[bold yellow]Entrez l'ID de l'utilisateur :[/bold yellow] ")
                    movie_title = Prompt.ask("[bold yellow]Entrez le nom du film :[/bold yellow] ")
                    try:
                        recommendations = recommend_movies(svd, movies, ratings, user_id, movie_title)
                        display_recommendations(recommendations, title="Recommandations de Films")
                    except Exception as e:
                        console.print(f"[bold red]Erreur: {e}[/bold red]")
                else:
                    console.print("[bold red]Vous devez être un utilisateur enregistré pour utiliser ce système de recommandation.[/bold red]")

            elif option == "base_sur_le_contenu":
                prompt_text = Prompt.ask("[bold yellow]Entrez une description ou un prompt pour les recommandations :[/bold yellow] ")
                try:
                    recommendations = recommend_movies_based_on_prompt(prompt_text).values.tolist()
                    display_recommendations(recommendations, title="Recommandations de Films")
                except Exception as e:
                    console.print(f"[bold red]Erreur: {e}[/bold red]")

            elif option == "hybride":
                user_id = Prompt.ask("[bold yellow]Entrez l'ID de l'utilisateur :[/bold yellow] ")
                movie_title = Prompt.ask("[bold yellow]Entrez le nom du film :[/bold yellow] ")
                try:
                    recommendations = recommend_hybrid(user_id, movie_title, svd, tfidf_matrix, movies)
                    display_recommendations(recommendations, title="Recommandations de Films")
                except Exception as e:
                    console.print(f"[bold red]Erreur: {e}[/bold red]")

        elif category == "anime":
            anime_title = Prompt.ask("[bold yellow]Entrez le titre de l'anime :[/bold yellow] ")
            try:
                recommendations = get_recommendations(anime_title)
                display_recommendations(recommendations, title="Recommandations d'Animes")
            except Exception as e:
                console.print(f"[bold red]Erreur: {e}[/bold red]")
        elif category == "manga":
            if option == "base_sur_le_contenu":
                prompt_text = Prompt.ask(
                    "[bold yellow]Entrez une description ou un prompt pour les recommandations :[/bold yellow] ")
                try:
                    recommendations = get_recommendations_from_prompt(prompt_text).values.tolist()
                    display_recommendations(recommendations, title="Recommandations de mangas")
                except Exception as e:
                    console.print(f"[bold red]Erreur: {e}[/bold red]")
            elif option == "filtrage_collaboratif":
                user_registered = Prompt.ask("[bold yellow]Êtes-vous un utilisateur enregistré ?[/bold yellow]",
                                             choices=["oui", "non"], default="non")
                if user_registered == 'oui':
                    manga_user_id = Prompt.ask("[bold yellow]Entrez l'ID de l'utilisateur :[/bold yellow] ")
                    try:
                        recommendations = get_manga_recommendations_collab(manga_user_id, manga_svd)
                        display_recommendations(recommendations, title="Recommandations de Films")
                    except Exception as e:
                        console.print(f"[bold red]Erreur: {e}[/bold red]")
                else:
                    console.print(
                        "[bold red]Vous devez être un utilisateur enregistré pour utiliser ce système de recommandation.[/bold red]")


if __name__ == "__main__":
    main()