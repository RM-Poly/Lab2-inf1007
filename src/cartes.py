# Auteurs : Claude (Claude Code), Wproulx09
"""Cartes, main et table : transforme les cartes posées sur la table en saisies pour le modèle.

Aucune dépendance à l'interface : cette logique est testée avec pytest.
"""

from dataclasses import dataclass

from src.exceptions import DonneeInvalideError

RANGS: list[str] = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
COULEURS: list[str] = ["♠", "♥", "♦", "♣"]
COULEURS_ROUGES: set[str] = {"♥", "♦"}
RANGS_DIX: set[str] = {"10", "J", "Q", "K"}
ZONES: tuple[str, ...] = ("joueur", "croupier", "paquet")


@dataclass(frozen=True)
class Carte:
    """Une carte d'un paquet de 52."""

    rang: str
    couleur: str

    def __post_init__(self) -> None:
        if self.rang not in RANGS:
            raise DonneeInvalideError(f"Rang de carte inconnu : {self.rang!r}.")
        if self.couleur not in COULEURS:
            raise DonneeInvalideError(f"Couleur de carte inconnue : {self.couleur!r}.")

    @classmethod
    def depuis_code(cls, code: str) -> "Carte":
        """Recrée une carte à partir de son code (ex. « 10♥ »).

        Échoue : DonneeInvalideError si le code ne correspond à aucune carte.
        """
        if not isinstance(code, str) or len(code) < 2:
            raise DonneeInvalideError(f"Code de carte invalide : {code!r}.")
        return cls(code[:-1], code[-1])

    @property
    def code(self) -> str:
        return f"{self.rang}{self.couleur}"

    @property
    def valeur(self) -> int:
        """Valeur au blackjack (As = 11, figures = 10)."""
        if self.rang == "A":
            return 11
        return 10 if self.rang in RANGS_DIX else int(self.rang)

    @property
    def vaut_dix(self) -> bool:
        return self.rang in RANGS_DIX

    @property
    def est_rouge(self) -> bool:
        return self.couleur in COULEURS_ROUGES

    def __str__(self) -> str:
        return self.code


PAQUET_COMPLET: list[Carte] = [Carte(rang, couleur) for couleur in COULEURS for rang in RANGS]


class Main:
    """Les cartes d'un joueur."""

    def __init__(self) -> None:
        self._cartes: list[Carte] = []

    @property
    def cartes(self) -> list[Carte]:
        return list(self._cartes)

    def ajouter(self, carte: Carte) -> None:
        self._cartes.append(carte)

    def retirer(self, carte: Carte) -> bool:
        """Retire la carte si elle est dans la main. Renvoie True si elle y était."""
        if carte in self._cartes:
            self._cartes.remove(carte)
            return True
        return False

    def _score_et_as_souples(self) -> tuple[int, int]:
        total = sum(carte.valeur for carte in self._cartes)
        as_a_11 = sum(1 for carte in self._cartes if carte.rang == "A")
        while total > 21 and as_a_11:
            total -= 10  # un As passe de 11 à 1
            as_a_11 -= 1
        return total, as_a_11

    @property
    def score(self) -> int:
        """Meilleur score possible sans dépasser 21 si on peut."""
        return self._score_et_as_souples()[0]

    @property
    def as_souple(self) -> bool:
        """Vrai si un As compte encore pour 11."""
        return self._score_et_as_souples()[1] > 0

    def __contains__(self, carte: object) -> bool:
        return carte in self._cartes

    def __len__(self) -> int:
        return len(self._cartes)

    def __str__(self) -> str:
        return " ".join(str(carte) for carte in self._cartes) or "(vide)"


class Table:
    """Le paquet, la main du joueur et la carte visible du croupier (composition)."""

    def __init__(self) -> None:
        self._main = Main()
        self._carte_croupier: Carte | None = None

    @property
    def main(self) -> Main:
        return self._main

    @property
    def carte_croupier(self) -> Carte | None:
        return self._carte_croupier

    @property
    def paquet(self) -> list[Carte]:
        """Les cartes qui ne sont pas encore sur la table, dans l'ordre du paquet."""
        return [carte for carte in PAQUET_COMPLET if not self._est_sur_table(carte)]

    def _est_sur_table(self, carte: Carte) -> bool:
        return carte in self._main or carte == self._carte_croupier

    def remettre(self, carte: Carte) -> None:
        """Remet une carte de la table dans le paquet (sans effet si elle y est déjà)."""
        if not self._main.retirer(carte) and carte == self._carte_croupier:
            self._carte_croupier = None

    def placer(self, carte: Carte, zone: str) -> None:
        """Déplace une carte vers « joueur », « croupier » ou « paquet ».

        Une nouvelle carte de croupier renvoie l'ancienne au paquet.
        Échoue : DonneeInvalideError si la zone est inconnue ou si la main dépasse déjà 21.
        """
        if zone not in ZONES:
            raise DonneeInvalideError(f"Zone inconnue : {zone!r}.")
        if zone == "paquet":
            self.remettre(carte)
            return
        if zone == "joueur":
            if carte in self._main:
                return
            if self._main.score > 21:
                raise DonneeInvalideError("Votre main dépasse déjà 21 : retirez une carte d'abord.")
            self.remettre(carte)
            self._main.ajouter(carte)
        else:
            self.remettre(carte)
            self._carte_croupier = carte

    def reinitialiser(self) -> None:
        self._main = Main()
        self._carte_croupier = None

    def vers_donnees_affichage(self) -> dict:
        """État de la table sous forme simple (JSON) pour le composant graphique."""
        def decrire(carte: Carte) -> dict:
            return {"code": carte.code, "rang": carte.rang, "couleur": carte.couleur,
                    "rouge": carte.est_rouge}

        return {
            "paquet": [decrire(c) if not self._est_sur_table(c) else None for c in PAQUET_COMPLET],
            "joueur": [decrire(c) for c in self._main.cartes],
            "croupier": decrire(self._carte_croupier) if self._carte_croupier else None,
            "score_joueur": self._main.score if len(self._main) else None,
            "as_souple": self._main.as_souple,
            "score_croupier": self._carte_croupier.valeur if self._carte_croupier else None,
        }

    def vers_saisies(self) -> dict[str, str]:
        """Traduit la table en saisies brutes pour Application.predire.

        Renvoie : un dictionnaire {caractéristique: texte}, comme si l'utilisateur
        avait rempli les champs à la main (l'application compare elle-même tirer et rester).
        Échoue : DonneeInvalideError si la situation ne permet pas de prédire.
        """
        if self._carte_croupier is None:
            raise DonneeInvalideError("Glissez une carte du paquet vers le croupier.")
        if len(self._main) < 2:
            raise DonneeInvalideError("Glissez au moins deux cartes du paquet vers votre main.")
        if self._main.score > 21:
            raise DonneeInvalideError(
                f"Votre main vaut {self._main.score} : vous avez dépassé 21, la main est perdue."
            )
        visibles = self._main.cartes + [self._carte_croupier]
        return {
            "score_joueur": str(self._main.score),
            "score_croupier": str(self._carte_croupier.valeur),
            "nb_dix": str(sum(1 for c in visibles if c.vaut_dix)),
            "as_souple": "oui" if self._main.as_souple else "non",
        }
