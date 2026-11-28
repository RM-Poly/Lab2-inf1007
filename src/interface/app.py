# Auteurs : Claude (Claude Code), Wproulx09
"""Interface graphique Streamlit.

Cette couche ne contient aucune logique de calcul : elle passe par la façade
Application (couche application) et par Table (cartes posées sur la table).
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from src.application import AVERTISSEMENT, CHEMIN_HISTORIQUE, Application, Prediction
from src.cartes import Carte, Table
from src.donnees import Variable, VariableCategorielle, VariableNumerique
from src.exceptions import DonneeInvalideError, FichierIntrouvableError
from src.graphiques import figure_comparaison
from src.interface.composant_table import table_de_jeu
from src.modele import DOSSIER_MODELES

STYLE = """
<style>
.conseil{border-radius:14px;padding:14px 18px;font-size:22px;font-weight:700;color:#fff;
  margin:4px 0 10px}
.conseil.tirer{background:#2a6f97}.conseil.rester{background:#c4553c}
.conseil small{display:block;font-size:13px;font-weight:400;opacity:.9;margin-top:2px}
</style>
"""

NOMS_METRIQUES = {
    "f1_macro": ("F1 macro", "plus haut = mieux"),
    "exactitude": ("Exactitude", "plus haut = mieux"),
    "log_loss": ("Log loss", "plus bas = mieux"),
}


@st.cache_resource
def creer_application() -> Application:
    """Charge le modèle sauvegardé une seule fois par session Streamlit."""
    return Application.depuis_fichiers(DOSSIER_MODELES, CHEMIN_HISTORIQUE)


def afficher_figure(figure: Figure) -> None:
    """Affiche une figure Matplotlib puis libère sa mémoire."""
    st.pyplot(figure, width="content")
    plt.close(figure)


def afficher_resultat(prediction: Prediction) -> None:
    """Conseil et phrase explicative à gauche, graphique tirer/rester à droite."""
    texte, graphique = st.columns([1, 1.25], gap="large")
    with texte:
        conseil = prediction.conseil
        st.markdown(
            f'<div class="conseil {conseil}">Conseil du modèle : {conseil.upper()}'
            f"<small>l'action qui gagne le plus en moyenne</small></div>",
            unsafe_allow_html=True,
        )
        if prediction.serree:
            st.warning(str(prediction))
        else:
            st.success(str(prediction))
    with graphique:
        afficher_figure(figure_comparaison(prediction.comparaison, prediction.conseil))


# ---------- Table de jeu ----------

def _table() -> Table:
    """La table de la session (créée au premier affichage)."""
    if "table" not in st.session_state:
        st.session_state.table = Table()
    return st.session_state.table


def _nouvelle_main() -> None:
    """Remet toutes les cartes au paquet."""
    _table().reinitialiser()
    st.session_state.erreur_table = None
    st.session_state.prediction_table = None


def appliquer_deplacement(table: Table, deplacement: dict) -> None:
    """Applique un déplacement reçu du composant ; mémorise l'erreur éventuelle."""
    st.session_state.erreur_table = None
    st.session_state.prediction_table = None
    try:
        table.placer(Carte.depuis_code(deplacement.get("carte")), deplacement.get("zone"))
    except DonneeInvalideError as erreur:
        st.session_state.erreur_table = str(erreur)


def afficher_onglet_table(app: Application) -> None:
    """Onglet principal : on glisse des cartes, le modèle compare tirer et rester."""
    table = _table()
    deplacement = table_de_jeu(table.vers_donnees_affichage())
    if deplacement:
        appliquer_deplacement(table, deplacement)
        st.rerun()  # redessine la table avec la carte déplacée

    if st.session_state.get("erreur_table"):
        st.error(st.session_state.erreur_table)

    gauche, droite = st.columns([2, 1], vertical_alignment="bottom")
    if gauche.button("Que dit le modèle ?", type="primary", width="stretch"):
        try:
            st.session_state.prediction_table = app.predire(table.vers_saisies())
        except DonneeInvalideError as erreur:
            st.session_state.prediction_table = None
            st.error(str(erreur))
    droite.button("🔄 Nouvelle main", on_click=_nouvelle_main, width="stretch")

    if st.session_state.get("prediction_table"):
        afficher_resultat(st.session_state.prediction_table)


# ---------- Saisie manuelle ----------

def champ_pour(variable: Variable) -> str:
    """Crée le widget adapté au type de la variable et renvoie la saisie brute."""
    if isinstance(variable, VariableCategorielle):
        return st.selectbox(variable.libelle, variable.valeurs, key=variable.nom)
    if isinstance(variable, VariableNumerique):
        return st.text_input(
            variable.libelle, key=variable.nom,
            placeholder=f"entier entre {variable.minimum} et {variable.maximum}",
        )
    return st.text_input(variable.libelle, key=variable.nom)


def afficher_onglet_manuel(app: Application) -> None:
    """Un champ par variable du modèle ; les erreurs de saisie sont toutes listées."""
    st.caption("Mêmes prédictions, en tapant directement les caractéristiques du modèle.")
    colonnes = st.columns(2)
    saisies = {}
    for i, variable in enumerate(app.variables):
        with colonnes[i % 2]:
            saisies[variable.nom] = champ_pour(variable)

    if st.button("Prédire", type="primary"):
        try:
            prediction = app.predire(saisies)
        except DonneeInvalideError as erreur:
            st.error("Vérifiez vos saisies :\n\n" + str(erreur).replace("\n", "\n\n"))
            return
        afficher_resultat(prediction)


# ---------- Le modèle ----------

def tableau_metriques(metriques: dict) -> pd.DataFrame:
    """Tableau « modèle contre référence » lisible par un non-informaticien."""
    lignes = [
        {"Mesure": nom, "Notre modèle": metriques.get(cle),
         "Référence naïve": metriques.get(f"{cle}_reference"), "Lecture": sens}
        for cle, (nom, sens) in NOMS_METRIQUES.items() if cle in metriques
    ]
    return pd.DataFrame(lignes).set_index("Mesure")


def afficher_onglet_modele(app: Application) -> None:
    """Résultats du modèle sur l'ensemble de test, comparés à la référence."""
    st.markdown(
        f"Le modèle est une **{app.nom_modele.lower()}** entraînée sur les décisions d'un agent "
        "de blackjack. La **référence naïve** répond toujours « défaite », l'issue la plus "
        "fréquente : un bon modèle doit faire mieux qu'elle."
    )
    st.dataframe(tableau_metriques(app.metriques).style.format(precision=2), width="stretch")
    st.caption("Le blackjack reste un jeu de hasard : même un modèle parfait se tromperait "
               "souvent sur l'issue d'une main.")


# ---------- Historique ----------

def afficher_onglet_historique(app: Application) -> None:
    """Les prédictions précédentes, les plus récentes en premier."""
    lignes = app.historique.lire()
    if not lignes:
        st.info("Aucune prédiction pour l'instant.")
        return
    st.dataframe(pd.DataFrame(lignes), width="stretch")


def lancer() -> None:
    """Point d'entrée de l'interface."""
    st.set_page_config(page_title="Conseiller blackjack", page_icon="🃏", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("🃏 Conseiller blackjack")
    st.warning(AVERTISSEMENT)
    try:
        app = creer_application()
    except (FichierIntrouvableError, DonneeInvalideError) as erreur:
        st.error(str(erreur))
        return

    onglets = st.tabs(["Table de jeu", "Saisie manuelle", "Le modèle", "Historique"])
    affichages = [afficher_onglet_table, afficher_onglet_manuel, afficher_onglet_modele,
                  afficher_onglet_historique]
    for onglet, afficher in zip(onglets, affichages):
        with onglet:
            afficher(app)
