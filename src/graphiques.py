"""Graphiques Matplotlib partagés par le script d'entraînement et l'interface.

Les fonctions renvoient une Figure ; l'affichage ou l'enregistrement est fait ailleurs.
"""

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : la figure est enregistrée ou passée à l'interface

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import ConfusionMatrixDisplay

from src.application_v2.application import esperance

COULEURS_ACTION = {"tirer": "#2a6f97", "rester": "#c4553c"}
# Ordre de lecture : du meilleur au pire résultat.
ISSUES: list[tuple[str, str, str]] = [
    ("victoire", "Victoire", "#2e7d32"),
    ("egalite", "Égalité", "#9e9e9e"),
    ("defaite", "Défaite", "#c62828"),
]
LIBELLES_VARIABLES = {
    "score_joueur": "Score de votre main",
    "score_croupier": "Carte du croupier",
    "nb_dix": "Cartes de valeur 10 visibles",
    "nb_cartes": "Nombre de cartes",
    "action": "Tirer ou rester",
    "as_souple": "As compté 11",
}


def _style_epure(ax) -> None:
    """Retire le cadre superflu pour un graphique plus lisible."""
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)


def figure_comparaison(comparaison: dict[str, dict[str, float]],
                       conseil: str | None = None) -> Figure:
    """Deux barres empilées (tirer, rester) découpées en victoire / égalité / défaite.

    Reçoit : {action: {issue: probabilité}} et l'action conseillée (marquée d'une étoile).
    """
    actions = ["rester", "tirer"]  # « tirer » en haut
    fig, ax = plt.subplots(figsize=(6.2, 2.6), layout="constrained")
    gauche = np.zeros(len(actions))
    for issue, libelle, couleur in ISSUES:
        valeurs = np.array([comparaison[a].get(issue, 0.0) for a in actions])
        barres = ax.barh(actions, valeurs, left=gauche, color=couleur, label=libelle, height=0.6)
        etiquettes = [f"{v:.0%}" if v >= 0.08 else "" for v in valeurs]
        ax.bar_label(barres, labels=etiquettes, label_type="center", color="white",
                     fontsize=10, fontweight="bold")
        gauche += valeurs

    ax.set_yticks(range(len(actions)),
                  [f"{a.capitalize()} {'★' if a == conseil else ''}".strip() for a in actions])
    for etiquette, action in zip(ax.get_yticklabels(), actions):
        etiquette.set_fontweight("bold" if action == conseil else "normal")
        etiquette.set_color(COULEURS_ACTION[action])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Probabilité estimée")
    ax.set_ylabel("Votre choix")
    ax.set_title("Issue de la main selon votre choix", fontsize=12, loc="left")
    fig.legend(ncols=3, loc="outside lower center", frameon=False, fontsize=9)
    _style_epure(ax)
    return fig


def figure_importances(importances: pd.Series, nom_modele: str) -> Figure:
    """Barres horizontales : poids de chaque variable dans les décisions du modèle."""
    serie = importances.sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.barh([LIBELLES_VARIABLES.get(n, n) for n in serie.index], serie.values, color="#2a6f97")
    ax.bar_label(ax.containers[0], labels=[f"{v:.0%}" for v in serie.values], padding=3,
                 fontsize=9)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, max(serie.max() * 1.18, 0.1))
    ax.set_xlabel("Importance relative")
    ax.set_ylabel("Variable")
    ax.set_title(f"Ce qui pèse le plus dans la prédiction\n{nom_modele}", fontsize=12, loc="left")
    _style_epure(ax)
    fig.tight_layout()
    return fig


def figure_exploration(df: pd.DataFrame) -> Figure:
    """Taux de victoire selon le score du joueur, pour chaque action (graphique d'exploration)."""
    taux = (
        df.assign(gagne=(df["resultat"] == "victoire").astype(float))
        .groupby(["score_joueur", "action"])["gagne"].mean()
        .unstack("action")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for action in taux.columns:
        ax.plot(taux.index, taux[action], marker="o", label=action,
                color=COULEURS_ACTION.get(action))
    ax.set_title("Taux de victoire selon le score du joueur et l'action choisie")
    ax.set_xlabel("Score de la main du joueur")
    ax.set_ylabel("Proportion de mains gagnées")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend(title="Action")
    _style_epure(ax)
    fig.tight_layout()
    return fig


def figure_matrice_confusion(y_vrai, y_predit) -> Figure:
    """Matrice de confusion de l'ensemble de test."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_vrai, y_predit, labels=[i for i, _, _ in ISSUES],
                                            ax=ax, colorbar=False)
    ax.set_title("Matrice de confusion (ensemble de test)")
    ax.set_xlabel("Issue prédite")
    ax.set_ylabel("Issue réelle")
    fig.tight_layout()
    return fig
