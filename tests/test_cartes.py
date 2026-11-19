# Auteurs : Claude (Claude Code), Wproulx09
"""Tests des cartes et de la table de jeu (couche utilisée par l'interface)."""

import pytest

from src.cartes import PAQUET_COMPLET, Carte, Main, Table
from src.exceptions import DonneeInvalideError


def c(code: str) -> Carte:
    return Carte.depuis_code(code)


def main_de(*codes: str) -> Main:
    main = Main()
    for code in codes:
        main.ajouter(c(code))
    return main


@pytest.mark.parametrize("codes, score, souple", [
    (("A♠", "K♥"), 21, True),
    (("A♠", "A♥"), 12, True),
    (("A♠", "A♥", "9♦"), 21, True),
    (("A♠", "9♥", "5♦"), 15, False),
    (("K♠", "Q♥", "5♦"), 25, False),
])
def test_score_et_as_souple(codes, score, souple):
    main = main_de(*codes)
    assert main.score == score
    assert main.as_souple is souple


@pytest.mark.parametrize("code", ["1♠", "A", "", "AX", None])
def test_code_invalide_leve_une_erreur(code):
    with pytest.raises(DonneeInvalideError):
        Carte.depuis_code(code)


def test_code_aller_retour():
    assert Carte.depuis_code("10♦") == Carte("10", "♦")
    assert Carte("10", "♦").code == "10♦"


def test_paquet_complet_et_paquet_restant():
    assert len(set(PAQUET_COMPLET)) == 52
    table = Table()
    table.placer(c("A♠"), "joueur")
    table.placer(c("K♥"), "croupier")
    assert len(table.paquet) == 50
    assert c("A♠") not in table.paquet


def test_deplacer_une_carte_du_joueur_au_croupier():
    table = Table()
    table.placer(c("7♣"), "joueur")
    table.placer(c("7♣"), "croupier")
    assert len(table.main) == 0
    assert table.carte_croupier == c("7♣")


def test_nouvelle_carte_du_croupier_remet_l_ancienne_au_paquet():
    table = Table()
    table.placer(c("9♠"), "croupier")
    table.placer(c("2♥"), "croupier")
    assert c("9♠") in table.paquet
    assert table.carte_croupier == c("2♥")


def test_remettre_au_paquet():
    table = Table()
    table.placer(c("5♦"), "joueur")
    table.placer(c("5♦"), "paquet")
    assert len(table.main) == 0 and len(table.paquet) == 52


def test_zone_inconnue_leve_une_erreur():
    with pytest.raises(DonneeInvalideError, match="Zone"):
        Table().placer(c("5♦"), "poubelle")


def test_vers_saisies_calcule_les_caracteristiques():
    table = Table()
    table.placer(c("Q♠"), "croupier")
    for code in ("A♥", "5♦", "K♣"):
        table.placer(c(code), "joueur")
    assert table.vers_saisies() == {
        "score_joueur": "16", "score_croupier": "10", "nb_dix": "2", "as_souple": "non",
    }


@pytest.mark.parametrize("croupier, joueur, message", [
    (None, ["5♠", "6♠"], "croupier"),
    ("9♠", ["5♠"], "deux cartes"),
])
def test_vers_saisies_situation_incomplete(croupier, joueur, message):
    table = Table()
    if croupier:
        table.placer(c(croupier), "croupier")
    for code in joueur:
        table.placer(c(code), "joueur")
    with pytest.raises(DonneeInvalideError, match=message):
        table.vers_saisies()


def test_main_qui_depasse_21():
    table = Table()
    table.placer(c("9♠"), "croupier")
    for code in ("K♠", "Q♠", "5♠"):
        table.placer(c(code), "joueur")
    with pytest.raises(DonneeInvalideError, match="dépassé 21"):
        table.vers_saisies()
    with pytest.raises(DonneeInvalideError, match="dépasse déjà 21"):
        table.placer(c("2♠"), "joueur")


def test_donnees_affichage():
    table = Table()
    table.placer(c("A♠"), "joueur")
    donnees = table.vers_donnees_affichage()
    assert donnees["paquet"][0] is None  # l'As de pique est sur la table
    assert donnees["joueur"][0]["code"] == "A♠"
    assert donnees["score_joueur"] == 11 and donnees["croupier"] is None


def test_les_saisies_de_la_table_passent_par_l_application(tmp_path):
    """La table produit exactement les saisies attendues par Application.predire."""
    from src.application import Application, Historique
    from src.donnees import JeuDeDonnees
    from src.modele import ForetAleatoire

    jeu = JeuDeDonnees()
    jeu.charger()
    jeu.nettoyer()
    X_train, _, y_train, _ = jeu.separer()
    modele = ForetAleatoire(nb_arbres=10)
    modele.entrainer(X_train, y_train)
    app = Application(modele, Historique(str(tmp_path / "h.csv")))

    table = Table()
    table.placer(c("6♦"), "croupier")
    table.placer(c("10♣"), "joueur")
    table.placer(c("3♥"), "joueur")
    prediction = app.predire(table.vers_saisies())
    assert prediction.conseil in ("tirer", "rester")
    assert sum(prediction.probas_conseil.values()) == pytest.approx(1.0)
