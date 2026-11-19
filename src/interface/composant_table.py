# Auteurs : Claude (Claude Code), Wproulx09
"""Composant Streamlit (st.components.v2) qui dessine la table et renvoie les cartes glissées."""

import os

import streamlit as st

DOSSIER_WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def _lire(nom: str) -> str:
    """Lit un fichier du dossier web/ (JS ou CSS du composant)."""
    chemin = os.path.join(DOSSIER_WEB, nom)
    if not os.path.isfile(chemin):
        raise FileNotFoundError(f"Fichier du composant introuvable : {chemin}")
    with open(chemin, encoding="utf-8") as fichier:
        return fichier.read()


_composant = st.components.v2.component(
    "table_blackjack",
    css=_lire("table.css"),
    js=_lire("table.js"),
)


def table_de_jeu(donnees: dict, key: str = "table_blackjack") -> dict | None:
    """Affiche la table et renvoie le dernier déplacement ({"carte", "zone"}) ou None."""
    resultat = _composant(data=donnees, key=key, on_deplacement_change=lambda: None)
    return getattr(resultat, "deplacement", None)
