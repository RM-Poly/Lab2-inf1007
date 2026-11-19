# Auteurs : Claude (Claude Code), Wproulx09
"""Tests de la couche application : prédiction, conseil, décision serrée, historique."""

import os

import pytest

from src.application import (AVERTISSEMENT, SEUIL_DECISION_SERREE, Application, Historique,
                             Prediction, esperance)
from src.donnees import JeuDeDonnees
from src.exceptions import DonneeInvalideError, FichierIntrouvableError
from src.modele import ForetAleatoire

SAISIES = {"score_joueur": "16", "score_croupier": "10", "nb_dix": "2", "as_souple": "non"}


@pytest.fixture(scope="module")
def dossier_modele(tmp_path_factory) -> str:
    """Une petite forêt entraînée sur les vraies données et sauvegardée dans un dossier temporaire."""
    jeu = JeuDeDonnees()
    jeu.charger()
    jeu.nettoyer()
    X_train, X_test, y_train, y_test = jeu.separer()
    modele = ForetAleatoire(nb_arbres=30)
    modele.entrainer(X_train, y_train)
    modele.evaluer(X_test, y_test, X_train, y_train)
    dossier = str(tmp_path_factory.mktemp("modele"))
    modele.sauvegarder(dossier)
    return dossier


@pytest.fixture
def app(dossier_modele, tmp_path) -> Application:
    """Une application neuve, avec un historique vide propre à chaque test."""
    return Application.depuis_fichiers(dossier_modele, os.path.join(tmp_path, "historique.csv"))


def prediction_fixe(victoire_tirer: float, defaite_tirer: float,
                    victoire_rester: float, defaite_rester: float) -> Prediction:
    """Construit une Prediction aux probabilités connues, sans passer par le modèle."""
    return Prediction(
        situation={"score_joueur": 16, "score_croupier": 10, "nb_dix": 2, "as_souple": "non"},
        comparaison={
            "tirer": {"victoire": victoire_tirer, "defaite": defaite_tirer,
                      "egalite": round(1 - victoire_tirer - defaite_tirer, 4)},
            "rester": {"victoire": victoire_rester, "defaite": defaite_rester,
                       "egalite": round(1 - victoire_rester - defaite_rester, 4)},
        },
    )


# ---- cas usuels ----

def test_predire_donne_un_conseil_et_des_probabilites(app):
    prediction = app.predire(SAISIES)
    assert prediction.conseil in ("tirer", "rester")
    for probas in prediction.comparaison.values():
        assert sum(probas.values()) == pytest.approx(1)


def test_le_conseil_est_l_action_de_meilleure_esperance():
    """Rester gagne plus souvent ici : le conseil doit être « rester »."""
    prediction = prediction_fixe(0.30, 0.65, 0.40, 0.55)
    assert prediction.conseil == "rester"
    assert prediction.ecart == pytest.approx(0.20)
    assert not prediction.serree


def test_rester_a_20_est_conseille(app):
    """Vérification de bon sens sur le vrai modèle."""
    assert app.predire({**SAISIES, "score_joueur": "20"}).conseil == "rester"


def test_esperance():
    assert esperance({"victoire": 0.4, "defaite": 0.5, "egalite": 0.1}) == pytest.approx(-0.1)


def test_texte_du_conseil():
    texte = str(prediction_fixe(0.30, 0.65, 0.40, 0.55))
    assert texte == "Conseil : rester. Victoire 40 %, défaite 55 %, égalité 5 %."


def test_les_variables_n_incluent_pas_l_action(app):
    """L'utilisateur ne choisit pas l'action : l'application compare les deux."""
    assert [v.nom for v in app.variables] == ["score_joueur", "score_croupier", "nb_dix",
                                              "as_souple"]


def test_metriques_et_nom_du_modele(app):
    assert app.nom_modele == "Forêt aléatoire"
    assert app.metriques["log_loss"] < app.metriques["log_loss_reference"]


def test_avertissement_present():
    assert "pédagogique" in AVERTISSEMENT


# ---- cas limites ----

def test_decision_serree_juste_sous_le_seuil():
    """Écart d'espérance de 0,04 : trop petit pour un conseil fiable."""
    prediction = prediction_fixe(0.30, 0.62, 0.30, 0.58)
    assert prediction.ecart < SEUIL_DECISION_SERREE
    assert prediction.serree
    assert "Décision serrée" in str(prediction)


def test_actions_parfaitement_egales():
    """Égalité exacte : un conseil est quand même donné, marqué comme serré."""
    prediction = prediction_fixe(0.30, 0.60, 0.30, 0.60)
    assert prediction.conseil in ("tirer", "rester")
    assert prediction.serree


def test_saisies_avec_espaces_et_majuscules(app):
    prediction = app.predire({**SAISIES, "score_joueur": " 16 ", "as_souple": "NON"})
    assert prediction.situation["score_joueur"] == 16
    assert prediction.situation["as_souple"] == "non"


# ---- historique ----

def test_chaque_prediction_est_enregistree(app):
    assert app.historique.lire() == []
    app.predire(SAISIES)
    app.predire({**SAISIES, "score_joueur": "12"})
    assert len(app.historique) == 2
    assert "predictions=2" in repr(app)


def test_historique_plus_recent_en_premier(tmp_path):
    historique = Historique(os.path.join(tmp_path, "h.csv"))
    ancienne = prediction_fixe(0.3, 0.6, 0.4, 0.5)
    recente = Prediction(situation={**ancienne.situation, "score_joueur": 20},
                         comparaison=ancienne.comparaison, date="2099-01-01T00:00:00")
    historique.ajouter(ancienne)
    historique.ajouter(recente)
    assert historique.lire()[0]["score_joueur"] == "20"


def test_une_saisie_invalide_n_est_pas_enregistree(app):
    with pytest.raises(DonneeInvalideError):
        app.predire({**SAISIES, "score_joueur": "abc"})
    assert len(app.historique) == 0


# ---- cas d'erreur ----

def test_toutes_les_erreurs_sont_listees(app):
    with pytest.raises(DonneeInvalideError) as erreur:
        app.predire({"score_joueur": "abc", "score_croupier": "", "nb_dix": "2", "as_souple": "non"})
    assert len(str(erreur.value).splitlines()) == 2


def test_score_hors_bornes_refuse(app):
    with pytest.raises(DonneeInvalideError, match="entre 4 et 21"):
        app.predire({**SAISIES, "score_joueur": "25"})


def test_modele_absent_leve_une_erreur(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        Application.depuis_fichiers(str(tmp_path), os.path.join(tmp_path, "h.csv"))


def test_prediction_est_immuable():
    prediction = prediction_fixe(0.3, 0.6, 0.4, 0.5)
    with pytest.raises(AttributeError):
        prediction.situation = {}
