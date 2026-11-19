"""Logique de l'application, indépendante de l'interface graphique (testable).

Application est la seule porte d'entrée de l'interface (façade) : elle valide les
saisies, interroge le modèle et tient l'historique.
"""

from dataclasses import dataclass

import pandas as pd

from src.application_v2.historique import Historique
from src.domaine.variables import Variable, variables_depuis_metadonnees
from src.exceptions import DonneeInvalideError
from src.modele_v2.modeles import ModeleBase

TRADUCTION_ISSUE: dict[str, str] = {
    "victoire": "une victoire",
    "defaite": "une défaite",
    "egalite": "une égalité",
}


def esperance(probas: dict[str, float]) -> float:
    """Gain moyen attendu par mise : P(victoire) - P(défaite). Entre -1 et 1."""
    return probas.get("victoire", 0.0) - probas.get("defaite", 0.0)


@dataclass(frozen=True)
class Prediction:
    """Résultat d'une prédiction, prêt à afficher."""

    situation: dict
    probas: dict[str, float]
    comparaison: dict[str, dict[str, float]]

    @property
    def issue_probable(self) -> str:
        return max(self.probas, key=self.probas.get)

    @property
    def conseil(self) -> str:
        """« tirer » ou « rester », selon l'action qui a la meilleure espérance."""
        return max(self.comparaison, key=lambda action: esperance(self.comparaison[action]))

    def __str__(self) -> str:
        action = self.situation["action"]
        issue = TRADUCTION_ISSUE[self.issue_probable]
        confiance = self.probas[self.issue_probable] * 100
        texte = (
            f"Si vous choisissez de {action}, l'issue la plus probable est {issue} "
            f"({confiance:.0f} % de confiance). "
        )
        if self.conseil == action:
            texte += "C'est aussi le meilleur choix d'après le modèle."
        else:
            texte += f"D'après le modèle, il vaudrait mieux {self.conseil}."
        return texte


class Application:
    """Contient le modèle, les variables d'entrée et l'historique (composition)."""

    def __init__(self, modele: ModeleBase, historique: Historique) -> None:
        self._modele = modele
        self._historique = historique
        self._variables: list[Variable] = variables_depuis_metadonnees(modele.metadonnees)

    @classmethod
    def depuis_fichiers(cls, dossier_modele: str, chemin_historique: str) -> "Application":
        """Construit l'application à partir du modèle sauvegardé et du fichier d'historique.

        Échoue : FichierIntrouvableError si le modèle n'a pas encore été entraîné.
        """
        return cls(ModeleBase.charger(dossier_modele), Historique(chemin_historique))

    @property
    def variables(self) -> list[Variable]:
        return list(self._variables)

    @property
    def historique(self) -> Historique:
        return self._historique

    @property
    def nom_modele(self) -> str:
        return self._modele.nom

    @property
    def metriques(self) -> dict:
        return self._modele.metadonnees.get("metriques", {})

    def importances(self) -> pd.Series:
        """Importance relative de chaque variable pour le modèle chargé."""
        return self._modele.importances()

    def valider_saisies(self, saisies: dict) -> dict:
        """Valide chaque saisie brute avec sa variable.

        Renvoie : la situation convertie.
        Échoue : DonneeInvalideError qui regroupe tous les messages d'erreur.
        """
        situation, erreurs = {}, []
        for variable in self._variables:
            try:
                situation[variable.nom] = variable.valider(saisies.get(variable.nom))
            except DonneeInvalideError as erreur:
                erreurs.append(str(erreur))
        if erreurs:
            raise DonneeInvalideError("\n".join(erreurs))
        return situation

    def predire(self, saisies: dict) -> Prediction:
        """Valide les saisies, prédit, enregistre dans l'historique et renvoie la prédiction."""
        situation = self.valider_saisies(saisies)
        prediction = Prediction(
            situation=situation,
            probas=self._modele.probabilites(situation),
            comparaison=self._modele.comparer_actions(situation),
        )
        self._historique.ajouter({
            **situation,
            "issue_probable": prediction.issue_probable,
            "proba_victoire": round(prediction.probas.get("victoire", 0.0), 4),
            "conseil": prediction.conseil,
        })
        return prediction
