# Auteurs : Claude (Claude Code), Wproulx09
"""Compare les conseils du modèle sauvegardé à la stratégie de base du blackjack.

La stratégie de base est la règle connue qui donne la meilleure décision en moyenne. On s'en sert
ici comme référence plus exigeante que « toujours défaite » : elle montre OÙ le modèle se trompe.

Simplifications (à garder en tête en lisant les résultats) :
  - mains « dures » seulement (sans As compté 11) ;
  - deux actions seulement : tirer ou rester (pas de doubler ni de séparer) ;
  - nb_dix fixé à la valeur la plus fréquente des données pour chaque situation.

Accord pondéré : chaque situation compte selon le nombre de fois où l'agent l'a rencontrée,
pour refléter les mains réellement jouées.

Usage : python outils/comparer_strategie_base.py
"""

import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)  # permet « import src… » en lançant le script directement

import pandas as pd  # noqa: E402

from src.application import Prediction  # noqa: E402
from src.donnees import CHEMIN_DONNEES, JeuDeDonnees  # noqa: E402
from src.modele import DOSSIER_MODELES, ModeleBase  # noqa: E402

SCORES = range(5, 21)          # 5 à 20 : à 21, on reste toujours ; sous 5, impossible en main dure
CARTES_CROUPIER = range(2, 12)  # 2 à 11 (As = 11)


def strategie_base(score: int, croupier: int) -> str:
    """Action de la stratégie de base pour une main dure (tirer ou rester seulement).

    Reçoit : le score de la main (sans As compté 11) et la carte visible du croupier (2 à 11).
    Renvoie : « tirer » ou « rester ».
    Échoue : ValueError si la carte du croupier n'est pas entre 2 et 11.
    """
    if not 2 <= croupier <= 11:
        raise ValueError(f"Carte du croupier invalide : {croupier} (attendu entre 2 et 11).")
    if score >= 17:
        return "rester"
    if score >= 13:
        return "rester" if croupier <= 6 else "tirer"
    if score == 12:
        return "rester" if 4 <= croupier <= 6 else "tirer"
    return "tirer"


def comparer(modele: ModeleBase, df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par situation : conseil du modèle, stratégie de base, accord et poids.

    Reçoit : un modèle entraîné et le jeu de données nettoyé.
    Renvoie : un DataFrame (score_joueur, score_croupier, modele, strategie, accord, serree, poids).
    Échoue : DonneeInvalideError si le modèle n'est pas entraîné.
    """
    dures = df[df["as_souple"] == "non"]
    lignes = []
    for score in SCORES:
        for croupier in CARTES_CROUPIER:
            situation_df = dures[(dures["score_joueur"] == score)
                                 & (dures["score_croupier"] == croupier)]
            nb_dix = int(situation_df["nb_dix"].mode()[0]) if len(situation_df) else 0
            situation = {"score_joueur": score, "score_croupier": croupier,
                         "nb_dix": nb_dix, "as_souple": "non"}
            prediction = Prediction(situation=situation,
                                    comparaison=modele.comparer_actions(situation))
            attendu = strategie_base(score, croupier)
            lignes.append({"score_joueur": score, "score_croupier": croupier,
                           "modele": prediction.conseil, "strategie": attendu,
                           "accord": prediction.conseil == attendu,
                           "serree": prediction.serree, "poids": len(situation_df)})
    return pd.DataFrame(lignes)


def main() -> None:
    """Affiche l'accord global, l'accord pondéré et la liste des désaccords."""
    jeu = JeuDeDonnees(CHEMIN_DONNEES)
    jeu.charger()
    jeu.nettoyer()
    tableau = comparer(ModeleBase.charger(DOSSIER_MODELES), jeu.df)

    accords = int(tableau["accord"].sum())
    pondere = (tableau["accord"] * tableau["poids"]).sum() / tableau["poids"].sum()
    print(f"Situations (mains dures, {len(SCORES)} scores × {len(CARTES_CROUPIER)} cartes) : "
          f"{len(tableau)}")
    print(f"Accord avec la stratégie de base : {accords} / {len(tableau)} "
          f"({accords / len(tableau):.0%})")
    print(f"Accord pondéré par la fréquence des situations : {pondere:.0%}\n")

    desaccords = tableau[~tableau["accord"]]
    print(f"Désaccords ({len(desaccords)}) — « serree » = signalé par l'application :")
    print(desaccords.drop(columns="accord").to_string(index=False))


if __name__ == "__main__":
    main()
