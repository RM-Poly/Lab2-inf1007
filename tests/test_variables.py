# Auteurs : Claude (Claude Code), Wproulx09
"""Tests des classes de variables et de la validation des saisies (cas usuels, limites, erreurs)."""

import pytest

from src.donnees import (CARACTERISTIQUES, Variable, VariableCategorielle, VariableNumerique,
                         creer_variables, valider_saisies)
from src.exceptions import DonneeInvalideError

SAISIES_VALIDES = {"score_joueur": "16", "score_croupier": "10", "nb_dix": "2",
                   "as_souple": "non", "action": "rester"}


@pytest.fixture
def score() -> VariableNumerique:
    return VariableNumerique("score_joueur", "Score de votre main", 4, 21)


@pytest.fixture
def action() -> VariableCategorielle:
    return VariableCategorielle("action", "Action envisagée", {"tirer", "rester"})


# ---- cas usuels ----

def test_numerique_convertit_en_entier(score):
    assert score.valider(" 16 ") == 16


def test_categorielle_normalise_la_casse(action):
    assert action.valider(" Tirer ") == "tirer"


def test_creer_variables_suit_les_caracteristiques():
    variables = creer_variables()
    assert [v.nom for v in variables] == CARACTERISTIQUES
    assert all(isinstance(v, Variable) for v in variables)


def test_creer_variables_depuis_les_metadonnees():
    """Les bornes et les valeurs viennent du JSON du modèle, pas des constantes du code."""
    metadonnees = {"caracteristiques": ["nb_dix", "as_souple"],
                   "numeriques": {"nb_dix": {"min": 0, "max": 3}},
                   "categorielles": {"as_souple": ["non", "oui"]}}
    nb_dix, as_souple = creer_variables(metadonnees)
    assert (nb_dix.minimum, nb_dix.maximum) == (0, 3)
    assert as_souple.valeurs == ["non", "oui"]


def test_metadonnees_incompletes():
    with pytest.raises(DonneeInvalideError):
        creer_variables({"caracteristiques": ["nb_dix"], "numeriques": {}, "categorielles": {}})


def test_polymorphisme_valider_selon_la_classe():
    # Le même appel valider() donne un int ou un str selon la classe de la variable.
    resultats = {v.nom: v.valider(SAISIES_VALIDES[v.nom]) for v in creer_variables()}
    assert resultats == {"score_joueur": 16, "score_croupier": 10, "nb_dix": 2,
                         "as_souple": "non", "action": "rester"}


def test_valider_saisies_renvoie_toutes_les_valeurs():
    assert valider_saisies(SAISIES_VALIDES)["score_joueur"] == 16


def test_repr_et_proprietes(score, action):
    assert repr(score) == "VariableNumerique('score_joueur', 4–21)"
    assert repr(action) == "VariableCategorielle('action', ['rester', 'tirer'])"
    assert (score.minimum, score.maximum) == (4, 21)
    assert action.valeurs == ["rester", "tirer"]


# ---- cas limites ----

@pytest.mark.parametrize("saisie, attendu", [("4", 4), ("21", 21)])
def test_numerique_accepte_les_bornes(score, saisie, attendu):
    assert score.valider(saisie) == attendu


@pytest.mark.parametrize("saisie", ["3", "22"])
def test_numerique_refuse_juste_hors_bornes(score, saisie):
    with pytest.raises(DonneeInvalideError, match="entre 4 et 21"):
        score.valider(saisie)


# ---- cas d'erreur ----

@pytest.mark.parametrize("saisie", ["", "   "])
def test_saisie_vide_est_refusee(score, action, saisie):
    for variable in (score, action):
        with pytest.raises(DonneeInvalideError, match="obligatoire"):
            variable.valider(saisie)


@pytest.mark.parametrize("saisie", ["abc", "15.5", "seize"])
def test_numerique_refuse_ce_qui_n_est_pas_un_entier(score, saisie):
    with pytest.raises(DonneeInvalideError, match="nombre entier"):
        score.valider(saisie)


def test_categorielle_refuse_un_choix_inconnu(action):
    with pytest.raises(DonneeInvalideError, match="rester, tirer"):
        action.valider("doubler")


def test_valider_saisies_liste_toutes_les_erreurs():
    saisies = {**SAISIES_VALIDES, "score_joueur": "abc", "action": ""}
    with pytest.raises(DonneeInvalideError) as erreur:
        valider_saisies(saisies)
    assert len(str(erreur.value).splitlines()) == 2


def test_variable_abstraite_ne_peut_pas_etre_creee():
    with pytest.raises(TypeError):
        Variable("x", "x")


def test_bornes_incoherentes_refusees():
    with pytest.raises(ValueError):
        VariableNumerique("x", "x", 10, 5)


def test_categorielle_sans_valeur_permise_refusee():
    with pytest.raises(ValueError, match="Aucune valeur permise"):
        VariableCategorielle("x", "x", set())
