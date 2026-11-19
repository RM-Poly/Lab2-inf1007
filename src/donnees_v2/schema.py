"""Schéma du jeu de données : colonnes, types et bornes physiquement possibles."""

COLONNES_NUMERIQUES: list[str] = ["score_joueur", "score_croupier", "nb_dix", "nb_cartes"]
COLONNES_CATEGORIELLES: list[str] = ["as_souple", "action"]
CARACTERISTIQUES: list[str] = COLONNES_NUMERIQUES + COLONNES_CATEGORIELLES
CIBLE: str = "resultat"
VALEURS_CIBLE: set[str] = {"victoire", "defaite", "egalite"}

# Valeurs autorisées des colonnes catégorielles.
VALEURS_CATEGORIELLES: dict[str, set[str]] = {
    "as_souple": {"oui", "non"},
    "action": {"tirer", "rester"},
}

# Bornes physiquement possibles au blackjack ; tout ce qui sort est aberrant.
BORNES: dict[str, tuple[int, int]] = {
    "score_joueur": (4, 21),
    "score_croupier": (2, 11),
    "nb_dix": (0, 8),
    "nb_cartes": (2, 11),
}
