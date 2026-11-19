"""Comparaison de plusieurs modèles sur les mêmes données et choix du meilleur."""

import pandas as pd

from src.modele_v2.modeles import ModeleBase


def comparer_modeles(candidats: list[ModeleBase], X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Entraîne et évalue chaque candidat sur la même séparation.

    Renvoie : un tableau (une ligne par modèle, une colonne par métrique), dans l'ordre reçu.
    Échoue : ValueError si la liste de candidats est vide.
    """
    if not candidats:
        raise ValueError("Il faut au moins un modèle à comparer.")
    lignes = {}
    for modele in candidats:
        modele.entrainer(X_train, y_train)
        lignes[modele.nom] = modele.evaluer(X_test, y_test, X_train, y_train)
    return pd.DataFrame.from_dict(lignes, orient="index")


def choisir_meilleur(candidats: list[ModeleBase], tableau: pd.DataFrame,
                     metrique: str = "log_loss") -> ModeleBase:
    """Renvoie le candidat qui a la plus petite valeur de la métrique (ex. log loss).

    Échoue : KeyError si la métrique n'est pas dans le tableau.
    """
    meilleur_nom = tableau[metrique].idxmin()
    return next(modele for modele in candidats if modele.nom == meilleur_nom)
