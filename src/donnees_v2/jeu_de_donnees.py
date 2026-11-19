"""Chargement, nettoyage et séparation du jeu de données de décisions de blackjack."""

import os
from collections.abc import Iterator

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import GRAINE, TAILLE_TEST
from src.donnees_v2.schema import BORNES, CARACTERISTIQUES, CIBLE, VALEURS_CATEGORIELLES, VALEURS_CIBLE
from src.exceptions import DonneeInvalideError, FichierIntrouvableError


def lire_par_blocs(chemin: str, taille_bloc: int = 5000) -> Iterator[pd.DataFrame]:
    """Générateur qui lit un CSV par blocs pour ne pas tout charger d'un coup.

    Reçoit : le chemin du CSV et le nombre de lignes par bloc.
    Renvoie : un itérateur de DataFrame.
    Échoue : FichierIntrouvableError si le fichier n'existe pas.
    """
    if not os.path.isfile(chemin):
        raise FichierIntrouvableError(f"Fichier introuvable : {chemin}")
    with open(chemin, encoding="utf-8") as fichier:
        for bloc in pd.read_csv(fichier, chunksize=taille_bloc):
            yield bloc


class JeuDeDonnees:
    """Encapsule le DataFrame des décisions et les étapes de préparation."""

    def __init__(self, chemin: str) -> None:
        self._chemin = chemin
        self._df: pd.DataFrame | None = None
        self._nb_lignes_retirees = 0

    @property
    def df(self) -> pd.DataFrame:
        """Le DataFrame chargé. Échoue avec DonneeInvalideError s'il n'est pas chargé."""
        if self._df is None:
            raise DonneeInvalideError("Le jeu de données n'est pas encore chargé.")
        return self._df

    @property
    def nb_lignes_retirees(self) -> int:
        """Nombre de lignes retirées par le nettoyage."""
        return self._nb_lignes_retirees

    def charger(self) -> pd.DataFrame:
        """Lit le CSV (par blocs) et vérifie que les colonnes attendues existent.

        Renvoie : le DataFrame brut.
        Échoue : FichierIntrouvableError ou DonneeInvalideError (colonnes manquantes).
        """
        df = pd.concat(lire_par_blocs(self._chemin), ignore_index=True)
        manquantes = set(CARACTERISTIQUES + [CIBLE]) - set(df.columns)
        if manquantes:
            raise DonneeInvalideError(f"Colonnes manquantes : {sorted(manquantes)}")
        self._df = df
        return df

    def nettoyer(self) -> pd.DataFrame:
        """Retire les valeurs manquantes, les valeurs hors bornes et les catégories inconnues.

        Les doublons sont conservés : deux mains identiques sont des observations
        distinctes et légitimes au blackjack.

        Renvoie : le DataFrame nettoyé.
        """
        df = self.df
        avant = len(df)
        df = df.dropna(subset=CARACTERISTIQUES + [CIBLE])
        for colonne, (bas, haut) in BORNES.items():
            df = df[df[colonne].between(bas, haut)]
        for colonne, valeurs in VALEURS_CATEGORIELLES.items():
            df = df[df[colonne].isin(valeurs)]
        df = df[df[CIBLE].isin(VALEURS_CIBLE)]
        self._nb_lignes_retirees = avant - len(df)
        self._df = df.reset_index(drop=True)
        return self._df

    def separer(self, taille_test: float = TAILLE_TEST, graine: int = GRAINE):
        """Sépare en ensembles d'entraînement et de test (stratifié sur la cible).

        Renvoie : (X_train, X_test, y_train, y_test).
        """
        X = self.df[CARACTERISTIQUES]
        y = self.df[CIBLE]
        return train_test_split(X, y, test_size=taille_test, random_state=graine, stratify=y)

    def __len__(self) -> int:
        return 0 if self._df is None else len(self._df)

    def __repr__(self) -> str:
        return f"JeuDeDonnees(chemin={self._chemin!r}, lignes={len(self)})"
