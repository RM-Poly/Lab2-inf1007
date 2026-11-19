"""Explore, nettoie, compare deux modèles, évalue et sauvegarde le meilleur.

Usage :  python entrainer.py
"""

import logging
import os

from src.config import CHEMIN_DONNEES, DOSSIER_FIGURES, DOSSIER_MODELES
from src.donnees import CIBLE, JeuDeDonnees
from src.exceptions import DonneeInvalideError, FichierIntrouvableError
from src.graphiques import figure_exploration, figure_matrice_confusion
from src.modele import ForetAleatoire, RegressionLogistique, choisir_meilleur, comparer_modeles

METRIQUE_DE_CHOIX = "log_loss"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(DOSSIER_FIGURES, exist_ok=True)

    # 1. Charger et explorer
    jeu = JeuDeDonnees(CHEMIN_DONNEES)
    try:
        jeu.charger()
    except (FichierIntrouvableError, DonneeInvalideError) as erreur:
        print(f"Erreur : {erreur}")
        return
    print(f"Lignes chargées : {len(jeu)}")
    print(jeu.df.dtypes, "\n")
    print("Valeurs manquantes :\n", jeu.df.isna().sum(), "\n")

    # 2. Nettoyer
    jeu.nettoyer()
    print(f"Lignes retirées au nettoyage : {jeu.nb_lignes_retirees} ; restantes : {len(jeu)}")
    print("Répartition de la cible :\n", jeu.df[CIBLE].value_counts(normalize=True).round(3), "\n")
    figure_exploration(jeu.df).savefig(os.path.join(DOSSIER_FIGURES, "exploration.png"), dpi=120)

    # 3. Séparer, entraîner et comparer
    X_train, X_test, y_train, y_test = jeu.separer()
    candidats = [ForetAleatoire(), RegressionLogistique()]
    tableau = comparer_modeles(candidats, X_train, X_test, y_train, y_test)
    print("=== Comparaison sur l'ensemble de test ===")
    print(tableau.round(3).to_string(), "\n")

    # 4. Garder le meilleur et le sauvegarder
    meilleur = choisir_meilleur(candidats, tableau, METRIQUE_DE_CHOIX)
    print(f"Modèle retenu (plus petite {METRIQUE_DE_CHOIX}) : {meilleur}")
    print("Importance des variables :\n", meilleur.importances().round(3), "\n")
    figure_matrice_confusion(y_test, meilleur.predire(X_test)).savefig(
        os.path.join(DOSSIER_FIGURES, "matrice_confusion.png"), dpi=120)
    meilleur.sauvegarder(DOSSIER_MODELES)
    print(f"Modèle et métadonnées enregistrés dans {DOSSIER_MODELES}")


if __name__ == "__main__":
    main()
