# Auteurs : Claude (Claude Code), Wproulx09
"""Modèles de classification : entraînement, évaluation honnête, sauvegarde et rechargement.

Le modèle prédit l'issue d'une main (victoire, défaite, égalité) à partir de la situation
et de l'action choisie. L'application l'interroge pour « tirer » et pour « rester »,
puis conseille l'action la plus favorable.

Usage (depuis la racine du dépôt) : python -m src.modele
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.donnees import (CARACTERISTIQUES, COLONNES_CATEGORIELLES, COLONNES_NUMERIQUES, GRAINE,
                         RACINE)
from src.exceptions import DonneeInvalideError, FichierIntrouvableError

DOSSIER_MODELES = os.path.join(RACINE, "modeles")
FICHIER_MODELE = "modele.joblib"
FICHIER_METADONNEES = "metadonnees.json"
ACTIONS: tuple[str, str] = ("tirer", "rester")

journal = logging.getLogger(__name__)


# ---------------------------------------------------------------- métriques (NumPy)

def _verifier_tableaux(y_vrai: Any, y_predit: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convertit en tableaux NumPy et vérifie qu'ils sont comparables.

    Reçoit : les vraies classes et les classes prédites (listes, Series ou tableaux).
    Renvoie : les deux tableaux NumPy.
    Échoue : ValueError si les tailles diffèrent ou si les tableaux sont vides.
    """
    y_vrai, y_predit = np.asarray(y_vrai), np.asarray(y_predit)
    if y_vrai.shape != y_predit.shape or y_vrai.size == 0:
        raise ValueError("Les tableaux doivent être non vides et de même taille.")
    return y_vrai, y_predit


def exactitude(y_vrai: Any, y_predit: Any) -> float:
    """Proportion de prédictions correctes (calcul vectorisé, sans boucle).

    Reçoit : les vraies classes et les classes prédites.
    Renvoie : un flottant entre 0 et 1.
    Échoue : ValueError si les tailles diffèrent ou si les tableaux sont vides.
    """
    y_vrai, y_predit = _verifier_tableaux(y_vrai, y_predit)
    return float(np.mean(y_vrai == y_predit))


def f1_macro(y_vrai: Any, y_predit: Any) -> float:
    """Moyenne des scores F1 de chaque classe (chaque classe compte autant, même rare).

    Calcul vectorisé : une matrice booléenne (classes × observations) remplace la boucle
    sur les classes.

    Reçoit : les vraies classes et les classes prédites.
    Renvoie : un flottant entre 0 et 1.
    Échoue : ValueError si les tailles diffèrent ou si les tableaux sont vides.
    """
    y_vrai, y_predit = _verifier_tableaux(y_vrai, y_predit)
    classes = np.union1d(y_vrai, y_predit)[:, None]       # forme (k, 1)
    est_vrai = y_vrai[None, :] == classes                  # forme (k, n)
    est_predit = y_predit[None, :] == classes
    vrais_positifs = np.sum(est_vrai & est_predit, axis=1)
    faux_positifs = np.sum(~est_vrai & est_predit, axis=1)
    faux_negatifs = np.sum(est_vrai & ~est_predit, axis=1)
    denominateur = 2 * vrais_positifs + faux_positifs + faux_negatifs
    f1 = np.divide(2 * vrais_positifs, denominateur,
                   out=np.zeros(len(denominateur)), where=denominateur > 0)
    return float(np.mean(f1))


def chronometrer(fonction: Callable) -> Callable:
    """Décorateur qui mesure la durée d'une fonction et l'écrit dans le journal (logging).

    Reçoit : la fonction à mesurer.
    Renvoie : une fonction qui fait la même chose, en mesurant sa durée.
    Échoue : jamais (les erreurs de la fonction mesurée sont transmises telles quelles).
    """
    @wraps(fonction)
    def enveloppe(*args: Any, **kwargs: Any) -> Any:
        """Appelle la fonction mesurée avec les mêmes arguments et renvoie son résultat."""
        debut = time.perf_counter()
        resultat = fonction(*args, **kwargs)
        journal.info("%s exécuté en %.2f s", fonction.__name__, time.perf_counter() - debut)
        return resultat

    return enveloppe


# ---------------------------------------------------------------- modèles

class ModeleBase(ABC):
    """Classe parente abstraite : déroulement commun à tous les modèles.

    Chaque classe enfant ne redéfinit que son estimateur scikit-learn (et, au besoin,
    son prétraitement). Le reste (entraîner, évaluer, prédire, sauvegarder) est partagé.
    """

    code: str = ""   # identifiant court, enregistré dans les métadonnées
    nom: str = ""    # nom lisible, affiché à l'utilisateur

    def __init__(self) -> None:
        """Assemble le pipeline : prétraitement puis estimateur (composition).

        Reçoit : rien.
        Renvoie : rien.
        Échoue : jamais.
        """
        self._pipeline = Pipeline([
            ("pretraitement", self._creer_pretraitement()),
            ("estimateur", self._creer_estimateur()),
        ])
        self._est_entraine = False
        self._metadonnees: dict = {}

    @abstractmethod
    def _creer_estimateur(self) -> Any:
        """Renvoie l'estimateur scikit-learn, non entraîné (redéfini par chaque enfant)."""

    def _creer_pretraitement(self) -> ColumnTransformer:
        """Encode les variables catégorielles en colonnes 0/1 ; les numériques passent telles quelles.

        Reçoit : rien.
        Renvoie : le ColumnTransformer.
        Échoue : jamais.
        """
        return ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), COLONNES_CATEGORIELLES)],
            remainder="passthrough",
        )

    @property
    def est_entraine(self) -> bool:
        """Vrai si le modèle a été entraîné ou chargé."""
        return self._est_entraine

    @property
    def metadonnees(self) -> dict:
        """Copie des métadonnées (variables, bornes, classes, métriques)."""
        return dict(self._metadonnees)

    @property
    def classes(self) -> list[str]:
        """Les issues possibles, dans l'ordre des probabilités.

        Échoue : DonneeInvalideError si le modèle n'est pas entraîné.
        """
        self._verifier_entraine()
        return list(self._pipeline.classes_)

    def _verifier_entraine(self) -> None:
        """Échoue avec DonneeInvalideError si le modèle n'est ni entraîné ni chargé."""
        if not self._est_entraine:
            raise DonneeInvalideError("Le modèle doit être entraîné ou chargé avant usage.")

    @chronometrer
    def entrainer(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Entraîne le modèle et mémorise les métadonnées des variables.

        Reçoit : les caractéristiques et la cible de l'ensemble d'entraînement.
        Renvoie : rien.
        Échoue : KeyError s'il manque une caractéristique dans X_train.
        """
        self._pipeline.fit(X_train[CARACTERISTIQUES], y_train)
        self._est_entraine = True
        self._metadonnees = {
            "type": self.code,
            "nom": self.nom,
            "caracteristiques": CARACTERISTIQUES,
            "numeriques": {col: {"min": int(X_train[col].min()), "max": int(X_train[col].max())}
                           for col in COLONNES_NUMERIQUES},
            "categorielles": {col: sorted(X_train[col].unique().tolist())
                              for col in COLONNES_CATEGORIELLES},
            "classes": self.classes,
            "graine": GRAINE,
            "nb_lignes_entrainement": len(X_train),
        }

    def evaluer(self, X_test: pd.DataFrame, y_test: pd.Series,
                X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
        """Mesure le modèle sur l'ensemble de test et le compare à une référence naïve.

        La référence prédit toujours l'issue la plus fréquente (« défaite »), avec comme
        probabilités la fréquence de chaque issue dans l'entraînement.

        Reçoit : l'ensemble de test (jamais vu à l'entraînement) et celui d'entraînement
                 (pour construire la référence).
        Renvoie : {métrique: valeur} pour le modèle et pour la référence.
        Échoue : DonneeInvalideError si le modèle n'est pas entraîné.
        """
        self._verifier_entraine()
        X = X_test[CARACTERISTIQUES]
        reference = DummyClassifier(strategy="prior").fit(X_train[CARACTERISTIQUES], y_train)
        y_predit, y_reference = self._pipeline.predict(X), reference.predict(X)
        resultats = {
            "f1_macro": f1_macro(y_test, y_predit),
            "exactitude": exactitude(y_test, y_predit),
            "log_loss": log_loss(y_test, self._pipeline.predict_proba(X), labels=self.classes),
            "f1_macro_reference": f1_macro(y_test, y_reference),
            "exactitude_reference": exactitude(y_test, y_reference),
            "log_loss_reference": log_loss(y_test, reference.predict_proba(X),
                                           labels=list(reference.classes_)),
        }
        self._metadonnees["metriques"] = {cle: round(val, 4) for cle, val in resultats.items()}
        self._metadonnees["nb_lignes_test"] = len(X_test)
        return resultats

    def predire(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit l'issue la plus probable pour chaque ligne.

        Reçoit : un DataFrame contenant les caractéristiques.
        Renvoie : un tableau d'issues.
        Échoue : DonneeInvalideError si le modèle n'est pas entraîné.
        """
        self._verifier_entraine()
        return self._pipeline.predict(X[CARACTERISTIQUES])

    def probabilites(self, situation: dict) -> dict[str, float]:
        """Probabilité de chaque issue pour UNE situation (action comprise).

        Reçoit : {caractéristique: valeur déjà validée}.
        Renvoie : {issue: probabilité}, dont la somme vaut 1.
        Échoue : DonneeInvalideError si le modèle n'est pas entraîné ;
                 KeyError s'il manque une caractéristique.
        """
        self._verifier_entraine()
        ligne = pd.DataFrame([situation])[CARACTERISTIQUES]
        return dict(zip(self.classes, map(float, self._pipeline.predict_proba(ligne)[0])))

    def comparer_actions(self, situation: dict) -> dict[str, dict[str, float]]:
        """Probabilités des issues pour « tirer » et pour « rester » dans la même situation.

        Reçoit : la situation sans l'action (score, carte du croupier, etc.).
        Renvoie : {action: {issue: probabilité}}.
        Échoue : comme probabilites().
        """
        return {action: self.probabilites({**situation, "action": action}) for action in ACTIONS}

    def sauvegarder(self, dossier: str = DOSSIER_MODELES) -> None:
        """Écrit le modèle (joblib) et ses métadonnées (JSON) dans le dossier.

        Reçoit : le dossier de destination (créé au besoin).
        Renvoie : rien.
        Échoue : DonneeInvalideError si le modèle n'est pas entraîné ; OSError si l'écriture échoue.
        """
        self._verifier_entraine()
        os.makedirs(dossier, exist_ok=True)
        joblib.dump(self._pipeline, os.path.join(dossier, FICHIER_MODELE))
        with open(os.path.join(dossier, FICHIER_METADONNEES), "w", encoding="utf-8") as fichier:
            json.dump(self._metadonnees, fichier, ensure_ascii=False, indent=2)

    @staticmethod
    def charger(dossier: str = DOSSIER_MODELES) -> "ModeleBase":
        """Recharge un modèle sauvegardé, avec la bonne classe enfant, sans réentraîner.

        Reçoit : le dossier qui contient modele.joblib et metadonnees.json.
        Renvoie : le modèle prêt à prédire.
        Échoue : FichierIntrouvableError si un des deux fichiers manque ;
                 DonneeInvalideError si le type de modèle est inconnu.
        """
        chemin_modele = os.path.join(dossier, FICHIER_MODELE)
        chemin_meta = os.path.join(dossier, FICHIER_METADONNEES)
        for chemin in (chemin_modele, chemin_meta):
            if not os.path.isfile(chemin):
                raise FichierIntrouvableError(
                    f"Fichier introuvable : {chemin}. Lancez d'abord « python -m src.modele »."
                )
        with open(chemin_meta, encoding="utf-8") as fichier:
            metadonnees = json.load(fichier)
        classe = MODELES.get(metadonnees.get("type"))
        if classe is None:
            raise DonneeInvalideError(f"Type de modèle inconnu : {metadonnees.get('type')!r}")
        modele = classe()
        modele._pipeline = joblib.load(chemin_modele)
        modele._metadonnees = metadonnees
        modele._est_entraine = True
        return modele

    def __repr__(self) -> str:
        """Représentation lisible, ex. ForetAleatoire(entraîné)."""
        etat = "entraîné" if self._est_entraine else "non entraîné"
        return f"{type(self).__name__}({etat})"


class ForetAleatoire(ModeleBase):
    """Forêt aléatoire : capte les effets non linéaires (rester à 16 est mauvais, à 20 excellent)."""

    code = "foret"
    nom = "Forêt aléatoire"

    def __init__(self, nb_arbres: int = 200, profondeur_max: int = 8,
                 min_feuille: int = 20) -> None:
        """Crée la forêt.

        Reçoit : le nombre d'arbres, leur profondeur maximale et le nombre minimal de mains
                 par feuille (évite d'apprendre par cœur les situations rares).
        Renvoie : rien.
        Échoue : jamais.
        """
        self._nb_arbres = nb_arbres
        self._profondeur_max = profondeur_max
        self._min_feuille = min_feuille
        super().__init__()

    def _creer_estimateur(self) -> RandomForestClassifier:
        """Renvoie la forêt scikit-learn, avec la graine fixe pour des résultats reproductibles."""
        return RandomForestClassifier(
            n_estimators=self._nb_arbres, max_depth=self._profondeur_max,
            min_samples_leaf=self._min_feuille, random_state=GRAINE,
        )


class RegressionLogistique(ModeleBase):
    """Régression logistique : modèle linéaire simple, sert de point de comparaison."""

    code = "logistique"
    nom = "Régression logistique"

    def _creer_pretraitement(self) -> ColumnTransformer:
        """Encode les catégorielles et met les numériques à la même échelle (requis pour un modèle linéaire).

        Reçoit : rien.
        Renvoie : le ColumnTransformer.
        Échoue : jamais.
        """
        return ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), COLONNES_CATEGORIELLES),
            ("num", StandardScaler(), COLONNES_NUMERIQUES),
        ])

    def _creer_estimateur(self) -> LogisticRegression:
        """Renvoie la régression logistique scikit-learn."""
        return LogisticRegression(max_iter=1000)


# Registre utilisé par ModeleBase.charger pour recréer la bonne classe enfant.
MODELES: dict[str, type[ModeleBase]] = {
    classe.code: classe for classe in (ForetAleatoire, RegressionLogistique)
}


# ---------------------------------------------------------------- comparaison

def comparer_modeles(candidats: list[ModeleBase], X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Entraîne et évalue chaque modèle sur la même séparation.

    Reçoit : les modèles candidats (non entraînés) et les ensembles d'entraînement et de test.
    Renvoie : un tableau, une ligne par modèle (indexée par son nom), une colonne par métrique.
    Échoue : ValueError si la liste de candidats est vide.
    """
    if not candidats:
        raise ValueError("Il faut au moins un modèle à comparer.")
    lignes = {}
    for modele in candidats:
        modele.entrainer(X_train, y_train)
        lignes[modele.nom] = modele.evaluer(X_test, y_test, X_train, y_train)
    return pd.DataFrame(lignes).T


def choisir_meilleur(candidats: list[ModeleBase], tableau: pd.DataFrame,
                     metrique: str = "log_loss") -> ModeleBase:
    """Renvoie le modèle qui a la plus petite valeur de la métrique (log loss : plus bas = mieux).

    Reçoit : les modèles comparés, le tableau de comparer_modeles et la métrique à minimiser.
    Renvoie : le meilleur modèle.
    Échoue : KeyError si la métrique n'est pas dans le tableau.
    """
    meilleur_nom = tableau[metrique].idxmin()
    return next(modele for modele in candidats if modele.nom == meilleur_nom)


if __name__ == "__main__":
    from src.donnees import JeuDeDonnees

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    jeu = JeuDeDonnees()
    jeu.charger()
    jeu.nettoyer()
    X_train, X_test, y_train, y_test = jeu.separer()
    print(f"Entraînement : {len(X_train)} lignes ; test : {len(X_test)} lignes\n")

    candidats: list[ModeleBase] = [ForetAleatoire(), RegressionLogistique()]
    tableau = comparer_modeles(candidats, X_train, X_test, y_train, y_test)
    print("=== Résultats sur l'ensemble de test ===")
    print(tableau.round(3).to_string(), "\n")

    meilleur = choisir_meilleur(candidats, tableau)
    meilleur.sauvegarder()
    print(f"Modèle retenu (plus petite log loss) : {meilleur.nom}")
    print(f"Enregistré dans {DOSSIER_MODELES}")
