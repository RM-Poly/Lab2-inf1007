"""Modèles de classification : une classe parente abstraite et deux modèles concrets.

ModeleBase fixe le déroulement commun (entraîner, évaluer, prédire, sauvegarder,
charger). Chaque classe enfant ne redéfinit que ce qui change : l'estimateur
scikit-learn, son prétraitement et la façon de mesurer l'importance des variables
(polymorphisme).
"""

import json
import os
from abc import ABC, abstractmethod

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

from src.config import GRAINE
from src.donnees_v2.schema import CARACTERISTIQUES, COLONNES_CATEGORIELLES, COLONNES_NUMERIQUES
from src.exceptions import DonneeInvalideError, FichierIntrouvableError, ModeleNonEntraineError
from src.modele_v2.metriques import chronometrer, exactitude, f1_macro

FICHIER_MODELE = "modele.joblib"
FICHIER_METADONNEES = "metadonnees.json"
ACTIONS: tuple[str, str] = ("tirer", "rester")


class ModeleBase(ABC):
    """Classe parente abstraite : prédit l'issue d'une main (victoire, défaite, égalité)."""

    code: str = ""  # identifiant court enregistré dans les métadonnées
    nom: str = ""   # nom lisible affiché à l'utilisateur

    def __init__(self) -> None:
        self._pipeline = Pipeline([
            ("pretraitement", self._creer_pretraitement()),
            ("estimateur", self._creer_estimateur()),
        ])
        self._est_entraine = False
        self._metadonnees: dict = {}

    # ---- ce que chaque modèle redéfinit ----

    @abstractmethod
    def _creer_estimateur(self):
        """Renvoie l'estimateur scikit-learn (non entraîné)."""

    def _creer_pretraitement(self) -> ColumnTransformer:
        """Encode les variables catégorielles ; les numériques passent telles quelles."""
        return ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), COLONNES_CATEGORIELLES)],
            remainder="passthrough", verbose_feature_names_out=False,
        )

    @abstractmethod
    def _importances_brutes(self) -> np.ndarray:
        """Une valeur d'importance par variable transformée."""

    # ---- déroulement commun ----

    @property
    def metadonnees(self) -> dict:
        return dict(self._metadonnees)

    @property
    def est_entraine(self) -> bool:
        return self._est_entraine

    @property
    def classes(self) -> list[str]:
        """Les classes de la cible, dans l'ordre des probabilités."""
        self._verifier_entraine()
        return list(self._pipeline.classes_)

    def _verifier_entraine(self) -> None:
        if not self._est_entraine:
            raise ModeleNonEntraineError("Le modèle doit être entraîné ou chargé avant usage.")

    @chronometrer
    def entrainer(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Entraîne le pipeline et mémorise les métadonnées des variables."""
        self._pipeline.fit(X_train[CARACTERISTIQUES], y_train)
        self._est_entraine = True
        self._metadonnees = {
            "type": self.code,
            "nom": self.nom,
            "caracteristiques": CARACTERISTIQUES,
            "numeriques": {
                col: {"min": int(X_train[col].min()), "max": int(X_train[col].max())}
                for col in COLONNES_NUMERIQUES
            },
            "categorielles": {
                col: sorted(X_train[col].unique().tolist()) for col in COLONNES_CATEGORIELLES
            },
            "classes": self.classes,
        }

    def evaluer(self, X_test: pd.DataFrame, y_test: pd.Series,
                X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
        """Compare le modèle à une référence triviale sur l'ensemble de test.

        La référence prédit toujours la classe majoritaire et, comme probabilités,
        la fréquence de chaque classe dans l'entraînement (DummyClassifier « prior »).
        La perte logarithmique (log loss, plus bas = mieux) juge la qualité des
        probabilités affichées à l'utilisateur.

        Renvoie : un dictionnaire de métriques (modèle et référence).
        """
        self._verifier_entraine()
        reference = DummyClassifier(strategy="prior").fit(X_train, y_train)
        X = X_test[CARACTERISTIQUES]
        y_pred = self._pipeline.predict(X)
        y_ref = reference.predict(X)
        resultats = {
            "f1_macro": f1_macro(y_test, y_pred),
            "exactitude": exactitude(y_test, y_pred),
            "log_loss": log_loss(y_test, self._pipeline.predict_proba(X), labels=self.classes),
            "f1_macro_reference": f1_macro(y_test, y_ref),
            "exactitude_reference": exactitude(y_test, y_ref),
            "log_loss_reference": log_loss(y_test, reference.predict_proba(X),
                                           labels=list(reference.classes_)),
        }
        self._metadonnees["metriques"] = {k: round(v, 4) for k, v in resultats.items()}
        return resultats

    def predire(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit la classe pour chaque ligne de X."""
        self._verifier_entraine()
        return self._pipeline.predict(X[CARACTERISTIQUES])

    def probabilites(self, situation: dict) -> dict[str, float]:
        """Probabilité de chaque issue pour UNE situation (avec l'action).

        Reçoit : un dictionnaire {caractéristique: valeur déjà validée}.
        Renvoie : {classe: probabilité}.
        """
        self._verifier_entraine()
        ligne = pd.DataFrame([situation])[CARACTERISTIQUES]
        probas = self._pipeline.predict_proba(ligne)[0]
        return dict(zip(self.classes, map(float, probas)))

    def comparer_actions(self, situation: dict) -> dict[str, dict[str, float]]:
        """Probabilités des issues pour « tirer » et pour « rester » dans la même situation."""
        return {action: self.probabilites({**situation, "action": action}) for action in ACTIONS}

    def importances(self) -> pd.Series:
        """Importance relative de chaque variable d'origine (somme = 1), en ordre décroissant.

        Les colonnes créées par l'encodage (ex. « action_tirer », « action_rester »)
        sont regroupées sous leur variable d'origine (« action »).
        """
        self._verifier_entraine()
        noms = self._pipeline.named_steps["pretraitement"].get_feature_names_out()
        valeurs = np.abs(self._importances_brutes())
        origines = [next(c for c in CARACTERISTIQUES if nom == c or nom.startswith(f"{c}_"))
                    for nom in noms]
        par_variable = pd.Series(valeurs, index=origines).groupby(level=0).sum()
        return (par_variable / par_variable.sum()).sort_values(ascending=False)

    def sauvegarder(self, dossier: str) -> None:
        """Écrit le pipeline (joblib) et les métadonnées (JSON) dans le dossier."""
        self._verifier_entraine()
        os.makedirs(dossier, exist_ok=True)
        joblib.dump(self._pipeline, os.path.join(dossier, FICHIER_MODELE))
        with open(os.path.join(dossier, FICHIER_METADONNEES), "w", encoding="utf-8") as fichier:
            json.dump(self._metadonnees, fichier, ensure_ascii=False, indent=2)

    @staticmethod
    def charger(dossier: str) -> "ModeleBase":
        """Recharge un modèle sauvegardé, avec la bonne classe enfant, sans ré-entraîner.

        Échoue : FichierIntrouvableError si un fichier manque ;
                 DonneeInvalideError si le type de modèle est inconnu.
        """
        chemin_modele = os.path.join(dossier, FICHIER_MODELE)
        chemin_meta = os.path.join(dossier, FICHIER_METADONNEES)
        for chemin in (chemin_modele, chemin_meta):
            if not os.path.isfile(chemin):
                raise FichierIntrouvableError(
                    f"Fichier introuvable : {chemin}. Lancez d'abord « python entrainer.py »."
                )
        with open(chemin_meta, encoding="utf-8") as fichier:
            metadonnees = json.load(fichier)
        classe = MODELES.get(metadonnees.get("type"))
        if classe is None:
            raise DonneeInvalideError(f"Type de modèle inconnu : {metadonnees.get('type')!r}.")
        modele = classe()
        modele._pipeline = joblib.load(chemin_modele)
        modele._metadonnees = metadonnees
        modele._est_entraine = True
        return modele

    def __str__(self) -> str:
        return self.nom

    def __repr__(self) -> str:
        etat = "entraîné" if self._est_entraine else "non entraîné"
        return f"{type(self).__name__}({etat})"


class ForetAleatoire(ModeleBase):
    """Forêt aléatoire : capte les interactions (score × carte du croupier × action)."""

    code = "foret"
    nom = "Forêt aléatoire"

    def __init__(self, nb_arbres: int = 200, profondeur_max: int = 8,
                 min_feuille: int = 20) -> None:
        self._nb_arbres = nb_arbres
        self._profondeur_max = profondeur_max
        self._min_feuille = min_feuille
        super().__init__()

    def _creer_estimateur(self) -> RandomForestClassifier:
        # Pas de class_weight="balanced" : il gonflait la probabilité des égalités
        # (classe rare) et rendait les probabilités affichées trompeuses.
        return RandomForestClassifier(
            n_estimators=self._nb_arbres, max_depth=self._profondeur_max,
            min_samples_leaf=self._min_feuille, random_state=GRAINE,
        )

    def _importances_brutes(self) -> np.ndarray:
        return self._pipeline.named_steps["estimateur"].feature_importances_


class RegressionLogistique(ModeleBase):
    """Régression logistique : modèle linéaire, plus simple à interpréter."""

    code = "logistique"
    nom = "Régression logistique"

    def __init__(self, regularisation: float = 1.0) -> None:
        self._regularisation = regularisation
        super().__init__()

    def _creer_pretraitement(self) -> ColumnTransformer:
        """Un modèle linéaire a besoin de variables numériques à la même échelle."""
        return ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), COLONNES_CATEGORIELLES),
             ("num", StandardScaler(), COLONNES_NUMERIQUES)],
            verbose_feature_names_out=False,
        )

    def _creer_estimateur(self) -> LogisticRegression:
        return LogisticRegression(C=self._regularisation, max_iter=1000)

    def _importances_brutes(self) -> np.ndarray:
        # Moyenne, sur les classes, de la valeur absolue des coefficients.
        return np.abs(self._pipeline.named_steps["estimateur"].coef_).mean(axis=0)


# Registre utilisé par ModeleBase.charger pour retrouver la bonne classe enfant.
MODELES: dict[str, type[ModeleBase]] = {
    classe.code: classe for classe in (ForetAleatoire, RegressionLogistique)
}
