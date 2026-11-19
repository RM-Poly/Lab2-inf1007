"""DONNÉES DE DÉVELOPPEMENT SEULEMENT.

Simule des mains de blackjack et écrit un CSV au même format que celui produit par
l'agent Elixir (agent_elixir/agent.ex). Sert à faire tourner le projet et la CI
avant d'avoir le vrai CSV de l'agent ; à remplacer ensuite par blackjack_decisions_v3.csv.

Usage :  python outils/simuler_parties.py 20000
"""

import csv
import os
import random
import sys

RANGS = [2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A"]
SEUIL_REBRASSAGE = 18
TAUX_EXPLORATION = 0.3
ENTETE = ["score_joueur", "score_croupier", "as_souple", "nb_dix", "cartes_utiles",
          "cartes_bust", "nb_cartes", "action", "resultat"]


def valeur(rang) -> int:
    if rang == "A":
        return 11
    return 10 if rang in ("J", "Q", "K") else rang


def score(main: list) -> tuple[int, bool]:
    """Renvoie (score, as_souple)."""
    total = sum(valeur(r) for r in main)
    as_restants = sum(1 for r in main if r == "A")
    while total > 21 and as_restants:
        total -= 10
        as_restants -= 1
    return total, as_restants > 0


def compter(paquet: list, score_joueur: int) -> tuple[int, int]:
    """Cartes restantes qui amèneraient à 17-21 (utiles) ou feraient sauter (bust)."""
    utiles = sum(1 for r in paquet
                 if (r == "A" and score_joueur + 1 <= 21)
                 or (r != "A" and max(1, 17 - score_joueur) <= valeur(r) <= 21 - score_joueur))
    bust = sum(1 for r in paquet if r != "A" and score_joueur + valeur(r) > 21)
    return utiles, bust


def jouer_main(paquet: list, alea: random.Random) -> list[list]:
    joueur = [paquet.pop(), paquet.pop()]
    croupier = [paquet.pop(), paquet.pop()]
    visible = croupier[0]
    decisions = []
    while True:
        s, souple = score(joueur)
        if s > 21:
            break
        voulu_tirer = s < 17 or (souple and s < 18)
        tirer = (not voulu_tirer) if alea.random() < TAUX_EXPLORATION else voulu_tirer
        if s == 21:
            tirer = False
        utiles, bust = compter(paquet, s)
        nb_dix = sum(1 for r in joueur + [visible] if valeur(r) == 10)
        decisions.append([s, valeur(visible), "oui" if souple else "non", nb_dix,
                          utiles, bust, len(joueur), "tirer" if tirer else "rester"])
        if not tirer:
            break
        joueur.append(paquet.pop())

    s_joueur, _ = score(joueur)
    if s_joueur > 21:
        resultat = "defaite"
    else:
        while score(croupier)[0] < 17:
            croupier.append(paquet.pop())
        s_croupier = score(croupier)[0]
        if s_croupier > 21 or s_joueur > s_croupier:
            resultat = "victoire"
        elif s_joueur < s_croupier:
            resultat = "defaite"
        else:
            resultat = "egalite"
    return [d + [resultat] for d in decisions]


def nouveau_paquet(alea: random.Random) -> list:
    paquet = RANGS * 4
    alea.shuffle(paquet)
    return paquet


def main(nb_lignes: int, chemin: str, graine: int = 42) -> None:
    alea = random.Random(graine)
    paquet = nouveau_paquet(alea)
    lignes: list[list] = []
    while len(lignes) < nb_lignes:
        if len(paquet) < SEUIL_REBRASSAGE:
            paquet = nouveau_paquet(alea)
        lignes.extend(jouer_main(paquet, alea))
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w", newline="", encoding="utf-8") as fichier:
        ecrivain = csv.writer(fichier)
        ecrivain.writerow(ENTETE)
        ecrivain.writerows(lignes[:nb_lignes])
    print(f"{nb_lignes} décisions simulées écrites dans {chemin}")


if __name__ == "__main__":
    racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    main(n, os.path.join(racine, "donnees", "blackjack_decisions.csv"))
