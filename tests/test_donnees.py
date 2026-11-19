# Auteurs : Claude (Claude Code), Wproulx09
"""Tests du chargement, du nettoyage et de la séparation des données (cas usuels, limites, erreurs)."""

import os

import pytest

from src.donnees import CARACTERISTIQUES, CHEMIN_DONNEES, JeuDeDonnees, lire_par_blocs
from src.exceptions import DonneeInvalideError, FichierIntrouvableError

ENTETE = "score_joueur,score_croupier,as_souple,nb_dix,cartes_utiles,cartes_bust,action,resultat\n"


def ecrire_csv(dossier, lignes: list[str], entete: str = ENTETE) -> str:
    """Écrit un petit CSV de test et renvoie son chemin."""
    chemin = os.path.join(dossier, "test.csv")
    with open(chemin, "w", encoding="utf-8") as fichier:
        fichier.write(entete + "".join(ligne + "\n" for ligne in lignes))
    return chemin


@pytest.fixture(scope="module")
def jeu_reel() -> JeuDeDonnees:
    """Le vrai jeu de données de l'agent, chargé et nettoyé."""
    jeu = JeuDeDonnees(CHEMIN_DONNEES)
    jeu.charger()
    jeu.nettoyer()
    return jeu


# ---- cas usuels ----

def test_charger_le_vrai_jeu_de_donnees():
    jeu = JeuDeDonnees(CHEMIN_DONNEES)
    df = jeu.charger()
    assert len(df) == 4181
    assert set(CARACTERISTIQUES) <= set(df.columns)


def test_nettoyer_le_vrai_jeu_retire_les_mains_depassant_21(jeu_reel):
    assert jeu_reel.nb_lignes_retirees == 3
    assert len(jeu_reel) == 4178
    assert jeu_reel.df["score_joueur"].max() <= 21


def test_separer_respecte_la_proportion_et_la_stratification(jeu_reel):
    X_train, X_test, y_train, y_test = jeu_reel.separer(taille_test=0.2)
    assert len(X_test) == pytest.approx(0.2 * len(jeu_reel), abs=1)
    assert len(X_train) + len(X_test) == len(jeu_reel)
    # Stratification : chaque issue garde la même proportion dans les deux ensembles.
    for issue in ("victoire", "defaite", "egalite"):
        assert (y_test == issue).mean() == pytest.approx((y_train == issue).mean(), abs=0.01)


def test_separer_est_reproductible_avec_la_meme_graine(jeu_reel):
    premier = jeu_reel.separer(graine=42)[1]
    second = jeu_reel.separer(graine=42)[1]
    assert premier.index.equals(second.index)


def test_lire_par_blocs_respecte_la_taille():
    tailles = [len(bloc) for bloc in lire_par_blocs(CHEMIN_DONNEES, taille_bloc=1000)]
    assert sum(tailles) == 4181
    assert max(tailles) == 1000


def test_len_et_repr(tmp_path):
    jeu = JeuDeDonnees(ecrire_csv(tmp_path, ["15,10,non,1,some,low_risk,tirer,victoire"]))
    assert len(jeu) == 0  # rien n'est chargé
    jeu.charger()
    assert len(jeu) == 1
    assert "lignes=1" in repr(jeu)


# ---- cas limites ----

def test_nettoyer_garde_les_valeurs_aux_bornes(tmp_path):
    jeu = JeuDeDonnees(ecrire_csv(tmp_path, [
        "21,11,oui,0,few,high_risk,rester,victoire",   # bornes hautes : 21 et As du croupier
        "4,2,non,0,many,low_risk,tirer,defaite",       # bornes basses
    ]))
    jeu.charger()
    assert len(jeu.nettoyer()) == 2
    assert jeu.nb_lignes_retirees == 0


def test_nettoyer_garde_les_doublons(tmp_path):
    ligne = "16,10,non,2,some,high_risk,rester,defaite"
    jeu = JeuDeDonnees(ecrire_csv(tmp_path, [ligne, ligne, ligne]))
    jeu.charger()
    assert len(jeu.nettoyer()) == 3


@pytest.mark.parametrize("ligne", [
    "22,10,non,1,some,low_risk,tirer,defaite",          # main déjà perdue (> 21)
    "15,1,non,1,some,low_risk,tirer,victoire",          # carte du croupier impossible
    "15,10,peut-etre,1,some,low_risk,tirer,victoire",   # catégorie inconnue
    "15,10,non,1,some,low_risk,doubler,victoire",       # action inconnue
    "15,10,non,1,some,low_risk,tirer,abandon",          # issue inconnue
    "15,,non,1,some,low_risk,rester,defaite",           # valeur manquante
])
def test_nettoyer_retire_les_lignes_aberrantes(tmp_path, ligne):
    jeu = JeuDeDonnees(ecrire_csv(tmp_path, ["15,10,non,1,some,low_risk,tirer,victoire", ligne]))
    jeu.charger()
    assert len(jeu.nettoyer()) == 1
    assert jeu.nb_lignes_retirees == 1


# ---- cas d'erreur ----

def test_fichier_inexistant_leve_une_erreur(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        JeuDeDonnees(os.path.join(tmp_path, "absent.csv")).charger()


def test_colonnes_manquantes_levent_une_erreur(tmp_path):
    chemin = ecrire_csv(tmp_path, ["15,10"], entete="score_joueur,score_croupier\n")
    with pytest.raises(DonneeInvalideError, match="Colonnes manquantes"):
        JeuDeDonnees(chemin).charger()


def test_utiliser_les_donnees_avant_de_charger_leve_une_erreur():
    jeu = JeuDeDonnees(CHEMIN_DONNEES)
    with pytest.raises(DonneeInvalideError):
        jeu.nettoyer()
