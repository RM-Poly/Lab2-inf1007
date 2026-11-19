"""Historique des prédictions, enregistré dans un fichier CSV."""

import csv
import os
from datetime import datetime

COLONNES_HISTORIQUE: list[str] = [
    "date", "score_joueur", "score_croupier", "nb_dix", "nb_cartes",
    "as_souple", "action", "issue_probable", "proba_victoire", "conseil",
]


class Historique:
    """Ajoute et relit les prédictions faites dans l'interface."""

    def __init__(self, chemin: str) -> None:
        self._chemin = chemin

    def ajouter(self, entree: dict) -> None:
        """Ajoute une ligne au CSV (crée le fichier et l'en-tête au besoin).

        Reçoit : un dictionnaire contenant les colonnes de COLONNES_HISTORIQUE sauf « date ».
        """
        dossier = os.path.dirname(self._chemin)
        if dossier:
            os.makedirs(dossier, exist_ok=True)
        nouveau = not os.path.isfile(self._chemin)
        ligne = {"date": datetime.now().isoformat(timespec="seconds"), **entree}
        with open(self._chemin, "a", newline="", encoding="utf-8") as fichier:
            ecrivain = csv.DictWriter(fichier, fieldnames=COLONNES_HISTORIQUE, extrasaction="ignore")
            if nouveau:
                ecrivain.writeheader()
            ecrivain.writerow(ligne)

    def lire(self) -> list[dict]:
        """Relit l'historique, de la plus récente à la plus ancienne.

        Renvoie : une liste de dictionnaires (vide si le fichier n'existe pas encore).
        """
        if not os.path.isfile(self._chemin):
            return []
        with open(self._chemin, newline="", encoding="utf-8") as fichier:
            lignes = list(csv.DictReader(fichier))
        return sorted(lignes, key=lambda ligne: ligne["date"], reverse=True)

    def __len__(self) -> int:
        return len(self.lire())
