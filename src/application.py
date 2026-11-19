# Auteurs : Claude (Claude Code), Wproulx09
"""Couche application (façade) : la seule porte d'entrée de l'interface vers la logique.

L'interface n'a qu'à :
    app = Application.depuis_fichiers()
    for variable in app.variables: ...           # un champ par variable
    prediction = app.predire(saisies)            # DonneeInvalideError si saisie invalide
    prediction.conseil, prediction.comparaison, prediction.serree, str(prediction)
    app.historique.lire()                         # prédictions précédentes

Aucun calcul n'est fait dans l'interface : tout se teste ici avec pytest.
"""

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime

from src.donnees import RACINE, Variable, creer_variables, valider_saisies
from src.modele import ACTIONS, DOSSIER_MODELES, ModeleBase

CHEMIN_HISTORIQUE = os.path.join(RACINE, "historique", "predictions.csv")

# En dessous de cet écart d'espérance entre tirer et rester, le conseil n'est pas fiable :
# les deux actions sont presque équivalentes selon le modèle.
SEUIL_DECISION_SERREE = 0.10

AVERTISSEMENT = ("Exercice pédagogique : ce modèle est entraîné sur les parties d'un agent "
                 "automatique. Ce n'est pas une aide à la décision pour des jeux d'argent.")


def esperance(probas: dict[str, float]) -> float:
    """Gain moyen d'une action pour une mise de 1 : P(victoire) − P(défaite).

    Reçoit : {issue: probabilité}.
    Renvoie : un nombre entre −1 (défaite certaine) et 1 (victoire certaine).
    Échoue : KeyError s'il manque « victoire » ou « defaite ».
    """
    return probas["victoire"] - probas["defaite"]


@dataclass(frozen=True)
class Prediction:
    """Résultat d'une prédiction : les probabilités pour chaque action et le conseil qui en découle.

    Immuable (frozen) : une prédiction affichée ou enregistrée ne peut plus être modifiée.
    """

    situation: dict[str, int | str]
    comparaison: dict[str, dict[str, float]]
    date: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def esperances(self) -> dict[str, float]:
        """Espérance de chaque action, ex. {"tirer": -0.51, "rester": -0.43}."""
        return {action: esperance(probas) for action, probas in self.comparaison.items()}

    @property
    def conseil(self) -> str:
        """L'action qui a la meilleure espérance : « tirer » ou « rester »."""
        return max(self.esperances, key=self.esperances.get)

    @property
    def ecart(self) -> float:
        """Écart d'espérance entre les deux actions (plus il est grand, plus le conseil est net)."""
        valeurs = list(self.esperances.values())
        return abs(valeurs[0] - valeurs[1])

    @property
    def serree(self) -> bool:
        """Vrai si les deux actions sont presque équivalentes : le conseil n'est alors pas fiable."""
        return self.ecart < SEUIL_DECISION_SERREE

    @property
    def probas_conseil(self) -> dict[str, float]:
        """Probabilités des issues si l'on suit le conseil."""
        return self.comparaison[self.conseil]

    def __str__(self) -> str:
        """Phrase prête à afficher, ex. « Conseil : rester. Victoire 32 %, défaite 63 %, égalité 5 %. »."""
        p = {issue: f"{round(proba * 100)} %" for issue, proba in self.probas_conseil.items()}
        texte = (f"Conseil : {self.conseil}. Victoire {p['victoire']}, "
                 f"défaite {p['defaite']}, égalité {p['egalite']}.")
        if self.serree:
            texte += " ⚠️ Décision serrée : les deux actions sont presque équivalentes."
        return texte


class Historique:
    """Enregistre les prédictions dans un fichier CSV et les relit."""

    COLONNES: list[str] = [
        "date", "score_joueur", "score_croupier", "nb_dix", "as_souple", "conseil",
        "victoire_tirer", "victoire_rester", "serree",
    ]

    def __init__(self, chemin: str = CHEMIN_HISTORIQUE) -> None:
        """Prépare l'historique (le fichier n'est créé qu'au premier ajout).

        Reçoit : le chemin du CSV.
        Renvoie : rien.
        Échoue : jamais.
        """
        self._chemin = chemin

    def ajouter(self, prediction: Prediction) -> None:
        """Ajoute une ligne au CSV, en créant le dossier et l'en-tête au besoin.

        Reçoit : la prédiction à enregistrer.
        Renvoie : rien.
        Échoue : OSError si le fichier ne peut pas être écrit.
        """
        os.makedirs(os.path.dirname(self._chemin), exist_ok=True)
        nouveau = not os.path.isfile(self._chemin)
        ligne = {
            "date": prediction.date,
            **{cle: prediction.situation[cle] for cle in self.COLONNES[1:5]},
            "conseil": prediction.conseil,
            "victoire_tirer": round(prediction.comparaison["tirer"]["victoire"], 4),
            "victoire_rester": round(prediction.comparaison["rester"]["victoire"], 4),
            "serree": "oui" if prediction.serree else "non",
        }
        with open(self._chemin, "a", newline="", encoding="utf-8") as fichier:
            ecrivain = csv.DictWriter(fichier, fieldnames=self.COLONNES)
            if nouveau:
                ecrivain.writeheader()
            ecrivain.writerow(ligne)

    def lire(self) -> list[dict[str, str]]:
        """Relit toutes les prédictions, les plus récentes en premier.

        Reçoit : rien.
        Renvoie : une liste de dictionnaires (vide si aucune prédiction n'a été faite).
        Échoue : OSError si le fichier existe mais ne peut pas être lu.
        """
        if not os.path.isfile(self._chemin):
            return []
        with open(self._chemin, encoding="utf-8") as fichier:
            lignes = list(csv.DictReader(fichier))
        return sorted(lignes, key=lambda ligne: ligne["date"], reverse=True)

    def __len__(self) -> int:
        """Nombre de prédictions enregistrées."""
        return len(self.lire())


class Application:
    """Façade : réunit le modèle, les variables et l'historique (composition)."""

    def __init__(self, modele: ModeleBase, historique: Historique) -> None:
        """Assemble l'application.

        Reçoit : un modèle entraîné ou chargé, et l'historique à utiliser.
        Renvoie : rien.
        Échoue : jamais.
        """
        self._modele = modele
        self._historique = historique
        # Variables saisies par l'utilisateur : tout sauf l'action, que l'application
        # fait varier elle-même pour comparer « tirer » et « rester ».
        self._variables = [v for v in creer_variables() if v.nom != "action"]

    @classmethod
    def depuis_fichiers(cls, dossier_modele: str = DOSSIER_MODELES,
                        chemin_historique: str = CHEMIN_HISTORIQUE) -> "Application":
        """Crée l'application à partir du modèle sauvegardé.

        Reçoit : le dossier du modèle (joblib + JSON) et le chemin de l'historique.
        Renvoie : l'application prête à prédire.
        Échoue : FichierIntrouvableError si le modèle n'a pas été entraîné
                 (lancer « python -m src.modele ») ;
                 DonneeInvalideError si le type de modèle est inconnu.
        """
        return cls(ModeleBase.charger(dossier_modele), Historique(chemin_historique))

    @property
    def variables(self) -> list[Variable]:
        """Les variables à demander à l'utilisateur (un champ par variable)."""
        return list(self._variables)

    @property
    def historique(self) -> Historique:
        """L'historique des prédictions."""
        return self._historique

    @property
    def nom_modele(self) -> str:
        """Nom lisible du modèle utilisé, ex. « Forêt aléatoire »."""
        return self._modele.nom

    @property
    def metriques(self) -> dict[str, float]:
        """Métriques du modèle sur l'ensemble de test (vides si inconnues)."""
        return dict(self._modele.metadonnees.get("metriques", {}))

    def predire(self, saisies: dict[str, str]) -> Prediction:
        """Valide les saisies, compare « tirer » et « rester », puis enregistre le résultat.

        Reçoit : {nom de la variable: texte saisi}, ex. {"score_joueur": "16", ...}.
        Renvoie : la Prediction (conseil, probabilités, décision serrée ou non).
        Échoue : DonneeInvalideError qui liste toutes les saisies invalides (une par ligne).
        """
        situation = valider_saisies(saisies, self._variables)
        comparaison = self._modele.comparer_actions(situation)
        prediction = Prediction(situation=situation,
                                comparaison={a: comparaison[a] for a in ACTIONS})
        self._historique.ajouter(prediction)
        return prediction

    def __repr__(self) -> str:
        """Représentation lisible, ex. Application(modele='Forêt aléatoire', predictions=3)."""
        return f"Application(modele={self.nom_modele!r}, predictions={len(self._historique)})"
