"""Couche données : schéma du CSV, chargement, nettoyage et séparation."""

from src.donnees_v2.jeu_de_donnees import JeuDeDonnees, lire_par_blocs
from src.donnees_v2.schema import (
    CARACTERISTIQUES,
    CIBLE,
    COLONNES_CATEGORIELLES,
    COLONNES_NUMERIQUES,
)

__all__ = [
    "JeuDeDonnees", "lire_par_blocs",
    "CARACTERISTIQUES", "CIBLE", "COLONNES_CATEGORIELLES", "COLONNES_NUMERIQUES",
]
