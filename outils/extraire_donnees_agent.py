# Auteurs : Claude (Claude Code), Wproulx09
"""Extrait le jeu de données du fichier de modèle de notre agent blackjack (Elixir, v3).

Source : agent_v3_model.bin, sauvegardé par l'agent du dépôt blackjack-agent
(lib/agent/serializer.ex : :erlang.term_to_binary({table, metadonnees}, [:compressed])).
La table contient, pour chaque (situation, action), le nombre de victoires, de défaites
et d'égalités obtenues contre le serveur de blackjack.

Étapes :
  1. décompresser le fichier (zlib / deflate) ;
  2. décoder le format binaire d'Erlang (External Term Format) ;
  3. « déplier » les compteurs : une ligne par décision réellement prise par l'agent,
     étiquetée avec le résultat final de la main.

Usage : python outils/extraire_donnees_agent.py [modele.bin] [sortie.csv]
"""

import csv
import os
import struct
import sys
import zlib
from typing import Any

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTREE_DEFAUT = os.path.join(RACINE, "donnees", "agent_v3_model.bin")
SORTIE_DEFAUT = os.path.join(RACINE, "donnees", "blackjack_decisions.csv")

ENTETE = ["score_joueur", "score_croupier", "as_souple", "nb_dix",
          "cartes_utiles", "cartes_bust", "action", "resultat"]
ACTIONS = {"hit": "tirer", "hold": "rester"}
ISSUES = ("victoire", "defaite", "egalite")  # même ordre que le triplet de l'agent


class LecteurErlang:
    """Décode le sous-ensemble du format binaire d'Erlang utilisé par l'agent."""

    def __init__(self, octets: bytes) -> None:
        """Prépare la lecture.

        Reçoit : les octets d'un terme Erlang, sans l'octet de version (131).
        Renvoie : rien.
        Échoue : jamais.
        """
        self._octets = octets
        self._pos = 0

    def _lire(self, n: int) -> bytes:
        """Lit les n octets suivants et avance la position de lecture.

        Reçoit : le nombre d'octets à lire.
        Renvoie : les octets lus (moins de n si la fin des données est atteinte).
        Échoue : jamais.
        """
        morceau = self._octets[self._pos:self._pos + n]
        self._pos += n
        return morceau

    def _entier(self, format_struct: str) -> int:
        """Lit un entier non signé ou signé au format big-endian demandé.

        Reçoit : un format du module struct (ex. ">B" pour 1 octet, ">I" pour 4 octets).
        Renvoie : l'entier lu.
        Échoue : struct.error s'il ne reste pas assez d'octets.
        """
        taille = struct.calcsize(format_struct)
        return struct.unpack(format_struct, self._lire(taille))[0]

    def terme(self) -> Any:
        """Lit le terme suivant et le convertit en objet Python.

        Reçoit : rien (lit à la position courante).
        Renvoie : int, float, str, bool, None, tuple, list ou dict selon le type Erlang.
        Échoue : ValueError si le type Erlang n'est pas pris en charge ;
                 struct.error ou IndexError si les données sont tronquées.
        """
        etiquette = self._lire(1)[0]
        if etiquette == 97:                       # SMALL_INTEGER_EXT
            return self._entier(">B")
        if etiquette == 98:                       # INTEGER_EXT
            return self._entier(">i")
        if etiquette == 70:                       # NEW_FLOAT_EXT
            return struct.unpack(">d", self._lire(8))[0]
        if etiquette in (100, 118):               # ATOM_EXT, ATOM_UTF8_EXT
            return self._atome(self._entier(">H"))
        if etiquette in (115, 119):               # SMALL_ATOM_EXT, SMALL_ATOM_UTF8_EXT
            return self._atome(self._entier(">B"))
        if etiquette == 104:                      # SMALL_TUPLE_EXT
            return tuple(self.terme() for _ in range(self._entier(">B")))
        if etiquette == 105:                      # LARGE_TUPLE_EXT
            return tuple(self.terme() for _ in range(self._entier(">I")))
        if etiquette == 106:                      # NIL_EXT (liste vide)
            return []
        if etiquette == 108:                      # LIST_EXT
            elements = [self.terme() for _ in range(self._entier(">I"))]
            self.terme()                          # queue de liste (toujours [])
            return elements
        if etiquette == 107:                      # STRING_EXT (liste d'octets)
            return list(self._lire(self._entier(">H")))
        if etiquette == 109:                      # BINARY_EXT (chaîne Elixir)
            return self._lire(self._entier(">I")).decode("utf-8")
        if etiquette == 116:                      # MAP_EXT
            paires = [(self.terme(), self.terme()) for _ in range(self._entier(">I"))]
            return dict(paires)
        if etiquette == 110:                      # SMALL_BIG_EXT
            n, signe = self._entier(">B"), self._entier(">B")
            valeur = int.from_bytes(self._lire(n), "little")
            return -valeur if signe else valeur
        raise ValueError(f"Type Erlang non pris en charge : {etiquette}")

    def _atome(self, longueur: int) -> str | bool | None:
        """Lit un atome Erlang et convertit true / false / nil en équivalent Python.

        Reçoit : la longueur du nom de l'atome, en octets.
        Renvoie : True, False, None, ou le nom de l'atome (ex. "hit").
        Échoue : UnicodeDecodeError si le nom n'est pas en UTF-8.
        """
        texte = self._lire(longueur).decode("utf-8")
        return {"true": True, "false": False, "nil": None}.get(texte, texte)


def lire_modele(chemin: str) -> tuple[list, dict]:
    """Lit le fichier sauvegardé par l'agent.

    Reçoit : le chemin du fichier .bin.
    Renvoie : (table, métadonnées) ; la table est une liste de
              ((situation..., action), (victoires, défaites, égalités)).
    Échoue : FileNotFoundError si le fichier n'existe pas ;
             ValueError si ce n'est pas un terme Erlang, si un type est inconnu
             ou si la taille décompressée ne correspond pas à celle annoncée ;
             zlib.error si la partie compressée est corrompue.
    """
    if not os.path.isfile(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    with open(chemin, "rb") as fichier:
        octets = fichier.read()
    if octets[0] != 131:
        raise ValueError("Ce fichier n'est pas au format binaire d'Erlang.")
    if octets[1] == 80:  # 'P' : terme compressé avec zlib
        taille = struct.unpack(">I", octets[2:6])[0]
        octets = bytes([131]) + zlib.decompress(octets[6:])
        if len(octets) != taille + 1:
            raise ValueError(
                f"Taille décompressée inattendue ({len(octets) - 1} octets au lieu de "
                f"{taille}) : le fichier est peut-être corrompu."
            )
    table, metadonnees = LecteurErlang(octets[1:]).terme()
    return table, metadonnees


def deplier(table: list) -> list[list]:
    """Transforme les compteurs en une ligne par décision.

    Reçoit : la table de l'agent (voir lire_modele).
    Renvoie : une liste de lignes au format de ENTETE ; chaque situation est répétée
              autant de fois que chaque issue a été observée.
    Échoue : KeyError si une action n'est ni « hit » ni « hold » ;
             ValueError si une entrée de la table n'a pas la forme attendue.
    """
    lignes = []
    for (joueur, croupier, souple, dix, utiles, bust, action), compteurs in table:
        situation = [joueur, croupier, "oui" if souple else "non", dix, utiles, bust,
                     ACTIONS[action]]
        for issue, nombre in zip(ISSUES, compteurs):
            lignes.extend([situation + [issue]] * nombre)
    return lignes


def main(entree: str = ENTREE_DEFAUT, sortie: str = SORTIE_DEFAUT) -> int:
    """Lit le fichier de l'agent et écrit le CSV des décisions.

    Reçoit : le chemin du .bin et celui du CSV à écrire (valeurs par défaut dans donnees/).
    Renvoie : le nombre de lignes écrites (sans l'en-tête).
    Échoue : les mêmes erreurs que lire_modele et deplier ;
             OSError si le CSV ne peut pas être écrit.
    """
    table, metadonnees = lire_modele(entree)
    lignes = deplier(table)
    with open(sortie, "w", newline="", encoding="utf-8") as fichier:
        ecrivain = csv.writer(fichier)
        ecrivain.writerow(ENTETE)
        ecrivain.writerows(lignes)
    print(f"Modèle sauvegardé le {metadonnees.get('timestamp')} "
          f"après {metadonnees.get('session_count')} sessions.")
    print(f"{len(table)} situations -> {len(lignes)} décisions écrites dans {sortie}")
    return len(lignes)


if __name__ == "__main__":
    main(*sys.argv[1:3])
