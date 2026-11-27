# Auteurs : Claude (Claude Code), Wproulx09
"""Tests de la comparaison avec la stratégie de base (outils/comparer_strategie_base.py)."""

import pytest

from outils.comparer_strategie_base import CARTES_CROUPIER, SCORES, comparer, strategie_base
from src.donnees import JeuDeDonnees
from src.modele import ForetAleatoire


@pytest.mark.parametrize("score, croupier, attendu", [
    (11, 10, "tirer"),    # impossible de dépasser 21 : on tire
    (12, 3, "tirer"),     # cas limite : 12 contre 3 → tirer
    (12, 4, "rester"),    # cas limite : 12 contre 4 → rester
    (16, 6, "rester"),    # le croupier risque de dépasser : on reste
    (16, 7, "tirer"),     # cas limite : à partir de 7, on tire
    (17, 11, "rester"),   # 17 et plus : on reste toujours
])
def test_strategie_base(score, croupier, attendu):
    assert strategie_base(score, croupier) == attendu


@pytest.mark.parametrize("croupier", [1, 12])
def test_carte_du_croupier_invalide(croupier):
    with pytest.raises(ValueError):
        strategie_base(16, croupier)


def test_comparer_couvre_toute_la_grille():
    """Une ligne par situation (score × carte du croupier), avec un poids positif ou nul."""
    jeu = JeuDeDonnees()
    jeu.charger()
    jeu.nettoyer()
    X_train, _, y_train, _ = jeu.separer()
    modele = ForetAleatoire(nb_arbres=10)
    modele.entrainer(X_train, y_train)

    tableau = comparer(modele, jeu.df)
    assert len(tableau) == len(SCORES) * len(CARTES_CROUPIER)
    assert set(tableau["modele"]) <= {"tirer", "rester"}
    assert (tableau["poids"] >= 0).all()
    # Les poids comptent chaque main dure de la grille exactement une fois.
    dures = jeu.df[(jeu.df["as_souple"] == "non") & jeu.df["score_joueur"].isin(SCORES)]
    assert tableau["poids"].sum() == len(dures)
