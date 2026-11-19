# Auteurs : Claude (Claude Code), Wproulx09
"""Tests du modèle : métriques NumPy, entraînement, évaluation, sauvegarde (cas usuels, limites, erreurs)."""

import json
import os

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.donnees import JeuDeDonnees
from src.exceptions import DonneeInvalideError, FichierIntrouvableError
from src.modele import (ACTIONS, MODELES, ForetAleatoire, ModeleBase, RegressionLogistique,
                        choisir_meilleur, comparer_modeles, exactitude, f1_macro)

SITUATION = {"score_joueur": 16, "score_croupier": 10, "nb_dix": 2, "as_souple": "non"}


@pytest.fixture(scope="module")
def separation():
    """Le vrai jeu de données, nettoyé et séparé (graine 42)."""
    jeu = JeuDeDonnees()
    jeu.charger()
    jeu.nettoyer()
    return jeu.separer()


@pytest.fixture(scope="module")
def foret(separation):
    """Une petite forêt (30 arbres, pour des tests rapides), entraînée et évaluée."""
    X_train, X_test, y_train, y_test = separation
    modele = ForetAleatoire(nb_arbres=30)
    modele.entrainer(X_train, y_train)
    modele.evaluer(X_test, y_test, X_train, y_train)
    return modele


# ---- métriques NumPy ----

def test_metriques_identiques_a_scikit_learn():
    """Nos calculs vectorisés doivent donner exactement les valeurs de scikit-learn."""
    rng = np.random.default_rng(0)
    y_vrai = rng.choice(["victoire", "defaite", "egalite"], size=500)
    y_predit = rng.choice(["victoire", "defaite", "egalite"], size=500)
    assert exactitude(y_vrai, y_predit) == pytest.approx(accuracy_score(y_vrai, y_predit))
    assert f1_macro(y_vrai, y_predit) == pytest.approx(f1_score(y_vrai, y_predit, average="macro"))


def test_metriques_predictions_parfaites():
    """Cas limite : tout est juste, donc les deux métriques valent 1."""
    y = ["victoire", "defaite", "egalite", "defaite"]
    assert exactitude(y, y) == 1.0
    assert f1_macro(y, y) == 1.0


def test_f1_macro_d_un_modele_qui_dit_toujours_defaite():
    """Cas limite : une seule classe prédite, les autres ont un F1 de 0."""
    y_vrai = ["defaite", "defaite", "victoire", "egalite"]
    y_predit = ["defaite"] * 4
    # F1 défaite = 2*2 / (2*2 + 2 + 0) = 2/3 ; victoire et égalité = 0.
    assert f1_macro(y_vrai, y_predit) == pytest.approx((2 / 3) / 3)


@pytest.mark.parametrize("y_vrai, y_predit", [([], []), (["victoire"], ["victoire", "defaite"])])
def test_metriques_refusent_des_tableaux_invalides(y_vrai, y_predit):
    """Cas d'erreur : tableaux vides ou de tailles différentes."""
    for metrique in (exactitude, f1_macro):
        with pytest.raises(ValueError):
            metrique(y_vrai, y_predit)


# ---- entraînement et évaluation ----

def test_le_modele_bat_la_reference(foret):
    """Évaluation honnête : sur l'ensemble de test, le modèle doit faire mieux que « toujours défaite »."""
    metriques = foret.metadonnees["metriques"]
    assert metriques["f1_macro"] > metriques["f1_macro_reference"]
    assert metriques["log_loss"] < metriques["log_loss_reference"]


def test_probabilites_valides(foret):
    """Les probabilités couvrent les 3 issues, sont entre 0 et 1 et somment à 1."""
    probas = foret.probabilites({**SITUATION, "action": "rester"})
    assert set(probas) == {"victoire", "defaite", "egalite"}
    assert all(0 <= p <= 1 for p in probas.values())
    assert sum(probas.values()) == pytest.approx(1)


def test_comparer_actions_donne_tirer_et_rester(foret):
    """L'application compare les deux actions dans la même situation."""
    comparaison = foret.comparer_actions(SITUATION)
    assert set(comparaison) == set(ACTIONS)


def test_rester_a_20_est_meilleur_qu_a_12(foret):
    """Vérification de bon sens : rester à 20 gagne plus souvent que rester à 12."""
    a_20 = foret.probabilites({**SITUATION, "score_joueur": 20, "action": "rester"})
    a_12 = foret.probabilites({**SITUATION, "score_joueur": 12, "action": "rester"})
    assert a_20["victoire"] > a_12["victoire"]


def test_predire_renvoie_une_issue_par_ligne(foret, separation):
    """predire() donne une issue connue pour chaque ligne de test."""
    X_test = separation[1]
    predictions = foret.predire(X_test)
    assert len(predictions) == len(X_test)
    assert set(predictions) <= {"victoire", "defaite", "egalite"}


def test_etat_et_repr():
    """est_entraine et __repr__ reflètent l'état du modèle."""
    modele = ForetAleatoire()
    assert not modele.est_entraine
    assert repr(modele) == "ForetAleatoire(non entraîné)"


def test_metadonnees_decrivent_les_variables(foret):
    """Les métadonnées servent à construire l'interface : variables, bornes, classes."""
    meta = foret.metadonnees
    assert meta["type"] == "foret"
    assert meta["numeriques"]["score_joueur"] == {"min": 4, "max": 21}
    assert meta["categorielles"]["action"] == ["rester", "tirer"]


def test_comparer_puis_choisir_le_meilleur(separation):
    """Les deux modèles sont comparés sur la même séparation ; on garde la plus petite log loss."""
    candidats = [ForetAleatoire(nb_arbres=30), RegressionLogistique()]
    tableau = comparer_modeles(candidats, *separation)
    assert list(tableau.index) == ["Forêt aléatoire", "Régression logistique"]
    meilleur = choisir_meilleur(candidats, tableau)
    assert tableau.loc[meilleur.nom, "log_loss"] == tableau["log_loss"].min()


def test_entrainement_reproductible(separation):
    """Même graine, mêmes données : mêmes probabilités."""
    X_train, _, y_train, _ = separation
    premier, second = ForetAleatoire(nb_arbres=10), ForetAleatoire(nb_arbres=10)
    premier.entrainer(X_train, y_train)
    second.entrainer(X_train, y_train)
    situation = {**SITUATION, "action": "tirer"}
    assert premier.probabilites(situation) == second.probabilites(situation)


# ---- sauvegarde et rechargement ----

def test_sauvegarde_puis_chargement_donne_les_memes_probas(foret, tmp_path):
    """Le modèle rechargé (joblib + JSON) prédit exactement comme l'original."""
    foret.sauvegarder(str(tmp_path))
    recharge = ModeleBase.charger(str(tmp_path))
    assert isinstance(recharge, ForetAleatoire)
    situation = {**SITUATION, "action": "rester"}
    assert recharge.probabilites(situation) == pytest.approx(foret.probabilites(situation))


def test_registre_contient_les_deux_modeles():
    """charger() retrouve la bonne classe enfant grâce au registre."""
    assert MODELES == {"foret": ForetAleatoire, "logistique": RegressionLogistique}


# ---- cas d'erreur ----

def test_classe_abstraite_ne_peut_pas_etre_creee():
    with pytest.raises(TypeError):
        ModeleBase()


def test_predire_sans_entrainer_leve_une_erreur():
    with pytest.raises(DonneeInvalideError):
        ForetAleatoire().probabilites({**SITUATION, "action": "tirer"})


def test_charger_un_dossier_vide_leve_une_erreur(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        ModeleBase.charger(str(tmp_path))


def test_charger_un_type_inconnu_leve_une_erreur(foret, tmp_path):
    foret.sauvegarder(str(tmp_path))
    chemin = os.path.join(tmp_path, "metadonnees.json")
    with open(chemin, encoding="utf-8") as fichier:
        meta = json.load(fichier)
    meta["type"] = "reseau_de_neurones"
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(meta, fichier)
    with pytest.raises(DonneeInvalideError, match="inconnu"):
        ModeleBase.charger(str(tmp_path))


def test_comparer_sans_candidat_leve_une_erreur(separation):
    with pytest.raises(ValueError):
        comparer_modeles([], *separation)
