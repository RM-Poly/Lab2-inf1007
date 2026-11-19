# Auteurs : Claude (Claude Code), Wproulx09
"""Tests des graphiques Matplotlib (ils doivent se construire sans erreur)."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.donnees import CHEMIN_DONNEES
from src.graphiques import figure_comparaison, figure_exploration

COMPARAISON = {
    "tirer": {"victoire": 0.30, "egalite": 0.05, "defaite": 0.65},
    "rester": {"victoire": 0.40, "egalite": 0.05, "defaite": 0.55},
}


def test_figure_comparaison_marque_le_conseil():
    figure = figure_comparaison(COMPARAISON, "rester")
    assert isinstance(figure, Figure)
    etiquettes = [e.get_text() for e in figure.axes[0].get_yticklabels()]
    assert "Rester ★" in etiquettes
    plt.close(figure)


def test_figure_exploration_sur_les_vraies_donnees():
    figure = figure_exploration(pd.read_csv(CHEMIN_DONNEES))
    assert len(figure.axes[0].get_lines()) == 2  # une courbe par action
    plt.close(figure)
