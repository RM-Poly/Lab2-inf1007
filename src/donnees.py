# Auteurs : Claude (Claude Code), Wproulx09
"""Lecture, nettoyage, séparation du jeu de données et classes de variables.

Usage (depuis la racine du dépôt) : python -m src.donnees
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exceptions import DonneeInvalideError, FichierIntrouvableError

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN_DONNEES = os.path.join(RACINE, "donnees", "blackjack_decisions.csv")
GRAINE = 42
TAILLE_TEST = 0.2

# Schéma : variables utilisées par le modèle, cible et valeurs permises.
COLONNES_NUMERIQUES: list[str] = ["score_joueur", "score_croupier", "nb_dix"]
COLONNES_CATEGORIELLES: list[str] = ["as_souple", "action"]
CARACTERISTIQUES: list[str] = COLONNES_NUMERIQUES + COLONNES_CATEGORIELLES
CIBLE: str = "resultat"
VALEURS_CIBLE: set[str] = {"victoire", "defaite", "egalite"}
VALEURS_CATEGORIELLES: dict[str, set[str]] = {
    "as_souple": {"oui", "non"},
    "action": {"tirer", "rester"},
}
# Bornes physiquement possibles au blackjack ; tout ce qui sort est aberrant.
BORNES: dict[str, tuple[int, int]] = {
    "score_joueur": (4, 21),
    "score_croupier": (2, 11),
    "nb_dix": (0, 8),
}
# Libellés affichés à l'utilisateur (dans l'interface et les messages d'erreur).
LIBELLES: dict[str, str] = {
    "score_joueur": "Score de votre main",
    "score_croupier": "Carte visible du croupier (As = 11)",
    "nb_dix": "Cartes de valeur 10 visibles",
    "as_souple": "As compté comme 11 dans votre main",
    "action": "Action envisagée",
}


class Variable(ABC):
    """Une caractéristique du modèle : son nom, son libellé et sa règle de validation."""

    def __init__(self, nom: str, libelle: str) -> None:
        """Crée la variable.

        Reçoit : le nom de la colonne et le libellé affiché à l'utilisateur.
        Renvoie : rien.
        Échoue : jamais.
        """
        self._nom = nom
        self._libelle = libelle

    @property
    def nom(self) -> str:
        """Nom de la colonne dans le jeu de données (lecture seule)."""
        return self._nom

    @property
    def libelle(self) -> str:
        """Libellé lisible pour l'utilisateur (lecture seule)."""
        return self._libelle

    @abstractmethod
    def valider(self, saisie: str) -> int | str:
        """Transforme une saisie brute en valeur utilisable par le modèle.

        Reçoit : le texte saisi par l'utilisateur.
        Renvoie : la valeur nettoyée.
        Échoue : DonneeInvalideError avec un message clair si la saisie est invalide.
        """

    def _verifier_non_vide(self, saisie: str) -> str:
        """Retire les espaces autour de la saisie et refuse une saisie vide.

        Reçoit : le texte saisi.
        Renvoie : le texte sans espaces autour.
        Échoue : DonneeInvalideError si la saisie est vide.
        """
        texte = str(saisie).strip()
        if not texte:
            raise DonneeInvalideError(f"« {self.libelle} » est obligatoire.")
        return texte


class VariableNumerique(Variable):
    """Variable entière, comprise entre un minimum et un maximum inclus."""

    def __init__(self, nom: str, libelle: str, minimum: int, maximum: int) -> None:
        """Crée la variable numérique.

        Reçoit : le nom, le libellé et les bornes permises (incluses).
        Renvoie : rien.
        Échoue : ValueError si le minimum dépasse le maximum.
        """
        if minimum > maximum:
            raise ValueError(f"Bornes incohérentes pour {nom} : {minimum} > {maximum}")
        super().__init__(nom, libelle)
        self._minimum = minimum
        self._maximum = maximum

    @property
    def minimum(self) -> int:
        """Plus petite valeur permise."""
        return self._minimum

    @property
    def maximum(self) -> int:
        """Plus grande valeur permise."""
        return self._maximum

    def valider(self, saisie: str) -> int:
        """Convertit la saisie en entier et vérifie les bornes.

        Reçoit : le texte saisi (ex. " 16 ").
        Renvoie : l'entier (ex. 16).
        Échoue : DonneeInvalideError si la saisie est vide, n'est pas un nombre
                 entier ou sort des bornes.
        """
        texte = self._verifier_non_vide(saisie)
        try:
            valeur = int(texte)
        except ValueError:
            raise DonneeInvalideError(
                f"« {self.libelle} » doit être un nombre entier (reçu : {texte!r})."
            ) from None
        if not self._minimum <= valeur <= self._maximum:
            raise DonneeInvalideError(
                f"« {self.libelle} » doit être entre {self._minimum} et {self._maximum} "
                f"(reçu : {valeur})."
            )
        return valeur

    def __repr__(self) -> str:
        """Représentation lisible, ex. VariableNumerique('score_joueur', 4–21)."""
        return f"VariableNumerique({self.nom!r}, {self._minimum}–{self._maximum})"


class VariableCategorielle(Variable):
    """Variable dont la valeur doit faire partie d'une liste de choix."""

    def __init__(self, nom: str, libelle: str, valeurs: set[str]) -> None:
        """Crée la variable catégorielle.

        Reçoit : le nom, le libellé et l'ensemble des valeurs permises.
        Renvoie : rien.
        Échoue : ValueError si aucune valeur n'est permise.
        """
        if not valeurs:
            raise ValueError(f"Aucune valeur permise pour {nom}.")
        super().__init__(nom, libelle)
        self._valeurs = frozenset(valeurs)

    @property
    def valeurs(self) -> list[str]:
        """Valeurs permises, triées (pour une liste déroulante)."""
        return sorted(self._valeurs)

    def valider(self, saisie: str) -> str:
        """Vérifie que la saisie est un des choix permis (sans tenir compte de la casse).

        Reçoit : le texte saisi (ex. " Oui ").
        Renvoie : la valeur normalisée (ex. "oui").
        Échoue : DonneeInvalideError si la saisie est vide ou n'est pas un choix permis.
        """
        texte = self._verifier_non_vide(saisie).lower()
        if texte not in self._valeurs:
            raise DonneeInvalideError(
                f"« {self.libelle} » doit être parmi : {', '.join(self.valeurs)} (reçu : {texte!r})."
            )
        return texte

    def __repr__(self) -> str:
        """Représentation lisible, ex. VariableCategorielle('action', ['rester', 'tirer'])."""
        return f"VariableCategorielle({self.nom!r}, {self.valeurs})"


def creer_variables() -> list[Variable]:
    """Construit une variable par caractéristique du modèle, dans l'ordre de CARACTERISTIQUES.

    Reçoit : rien (utilise BORNES, VALEURS_CATEGORIELLES et LIBELLES).
    Renvoie : la liste des variables.
    Échoue : jamais.
    """
    variables: list[Variable] = []
    for nom in CARACTERISTIQUES:
        if nom in BORNES:
            variables.append(VariableNumerique(nom, LIBELLES[nom], *BORNES[nom]))
        else:
            variables.append(VariableCategorielle(nom, LIBELLES[nom], VALEURS_CATEGORIELLES[nom]))
    return variables


def valider_saisies(saisies: dict[str, str]) -> dict[str, int | str]:
    """Valide toutes les saisies de l'utilisateur en une fois.

    Reçoit : {nom de la variable: texte saisi}.
    Renvoie : {nom de la variable: valeur validée}, prêt pour le modèle.
    Échoue : DonneeInvalideError qui liste toutes les erreurs trouvées (une par ligne).
    """
    valeurs: dict[str, int | str] = {}
    erreurs: list[str] = []
    for variable in creer_variables():
        try:
            valeurs[variable.nom] = variable.valider(saisies.get(variable.nom, ""))
        except DonneeInvalideError as erreur:
            erreurs.append(str(erreur))
    if erreurs:
        raise DonneeInvalideError("\n".join(erreurs))
    return valeurs


def lire_par_blocs(chemin: str, taille_bloc: int = 1000) -> Iterator[pd.DataFrame]:
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

    def __init__(self, chemin: str = CHEMIN_DONNEES) -> None:
        """Prépare le jeu de données sans encore lire le fichier.

        Reçoit : le chemin du CSV.
        Renvoie : rien.
        Échoue : jamais.
        """
        self._chemin = chemin
        self._df: pd.DataFrame | None = None
        self._nb_lignes_retirees = 0

    @property
    def df(self) -> pd.DataFrame:
        """Le DataFrame chargé.

        Reçoit : rien.
        Renvoie : le DataFrame (brut ou nettoyé selon l'étape).
        Échoue : DonneeInvalideError si charger() n'a pas encore été appelé.
        """
        if self._df is None:
            raise DonneeInvalideError("Le jeu de données n'est pas encore chargé.")
        return self._df

    @property
    def nb_lignes_retirees(self) -> int:
        """Nombre de lignes retirées par le dernier nettoyage.

        Reçoit : rien.
        Renvoie : un entier (0 si nettoyer() n'a pas été appelé).
        Échoue : jamais.
        """
        return self._nb_lignes_retirees

    def charger(self) -> pd.DataFrame:
        """Lit le CSV (par blocs) et vérifie que les colonnes attendues existent.

        Reçoit : rien (utilise le chemin donné au constructeur).
        Renvoie : le DataFrame brut.
        Échoue : FichierIntrouvableError si le fichier n'existe pas ;
                 DonneeInvalideError s'il manque des colonnes.
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

        Reçoit : rien.
        Renvoie : le DataFrame nettoyé.
        Échoue : DonneeInvalideError si charger() n'a pas encore été appelé.
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

    def separer(self, taille_test: float = TAILLE_TEST,
                graine: int = GRAINE) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Sépare en ensembles d'entraînement et de test, stratifiés sur la cible.

        Reçoit : la proportion de test et la graine aléatoire.
        Renvoie : (X_train, X_test, y_train, y_test).
        Échoue : DonneeInvalideError si charger() n'a pas encore été appelé ;
                 ValueError si une classe a trop peu de lignes pour être stratifiée.
        """
        X = self.df[CARACTERISTIQUES]
        y = self.df[CIBLE]
        return train_test_split(X, y, test_size=taille_test, random_state=graine, stratify=y)

    def __len__(self) -> int:
        """Nombre de lignes chargées (0 si rien n'est chargé)."""
        return 0 if self._df is None else len(self._df)

    def __repr__(self) -> str:
        """Représentation lisible, ex. JeuDeDonnees(chemin='...', lignes=4178)."""
        return f"JeuDeDonnees(chemin={self._chemin!r}, lignes={len(self)})"


if __name__ == "__main__":
    jeu = JeuDeDonnees()
    jeu.charger()
    print(f"Chargé : {len(jeu)} lignes")
    jeu.nettoyer()
    print(f"Nettoyé : {jeu.nb_lignes_retirees} lignes retirées, {len(jeu)} restantes")
    print("Répartition de la cible :")
    print(jeu.df[CIBLE].value_counts(normalize=True).round(3).to_string())
    X_train, X_test, _, _ = jeu.separer()
    print(f"Entraînement : {len(X_train)} lignes ; test : {len(X_test)} lignes")
