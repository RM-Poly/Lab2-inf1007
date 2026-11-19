"""Chemins et constantes du projet, définis à un seul endroit.

Tous les chemins sont construits avec os.path à partir de la racine du dépôt :
le projet fonctionne peu importe le dossier d'où on le lance.
"""

import os

RACINE: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOSSIER_DONNEES: str = os.path.join(RACINE, "donnees")
CHEMIN_DONNEES: str = os.path.join(DOSSIER_DONNEES, "blackjack_decisions.csv")
DOSSIER_MODELES: str = os.path.join(RACINE, "modeles")
DOSSIER_FIGURES: str = os.path.join(RACINE, "figures")
CHEMIN_HISTORIQUE: str = os.path.join(RACINE, "historique", "predictions.csv")

GRAINE: int = 42
TAILLE_TEST: float = 0.2
