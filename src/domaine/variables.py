"""Variables d'entrée du modèle : chacune valide la saisie à sa façon (polymorphisme)."""

from abc import ABC, abstractmethod
from typing import Any

from src.exceptions import DonneeInvalideError


class Variable(ABC):
    """Classe parente abstraite d'une variable d'entrée."""

    def __init__(self, nom: str, libelle: str) -> None:
        self._nom = nom
        self._libelle = libelle

    @property
    def nom(self) -> str:
        """Nom de la colonne dans le jeu de données."""
        return self._nom

    @property
    def libelle(self) -> str:
        """Texte affiché à l'utilisateur."""
        return self._libelle

    @abstractmethod
    def valider(self, saisie: Any) -> Any:
        """Convertit et valide une saisie brute.

        Renvoie : la valeur convertie, prête pour le modèle.
        Échoue : DonneeInvalideError avec un message clair pour l'utilisateur.
        """

    def __eq__(self, autre: object) -> bool:
        return isinstance(autre, Variable) and type(self) is type(autre) and self._nom == autre._nom

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._nom))


class VariableNumerique(Variable):
    """Variable entière bornée (ex. score du joueur entre 4 et 21)."""

    def __init__(self, nom: str, libelle: str, minimum: int, maximum: int) -> None:
        super().__init__(nom, libelle)
        if minimum > maximum:
            raise ValueError("minimum doit être <= maximum")
        self._minimum = minimum
        self._maximum = maximum

    @property
    def minimum(self) -> int:
        return self._minimum

    @property
    def maximum(self) -> int:
        return self._maximum

    def valider(self, saisie: Any) -> int:
        """Accepte un entier (ou un texte représentant un entier) dans [minimum, maximum]."""
        texte = str(saisie).strip() if saisie is not None else ""
        if texte == "":
            raise DonneeInvalideError(f"« {self._libelle} » est vide.")
        try:
            valeur = int(texte)
        except ValueError:
            raise DonneeInvalideError(
                f"« {self._libelle} » doit être un nombre entier (reçu : « {texte} »)."
            ) from None
        if not self._minimum <= valeur <= self._maximum:
            raise DonneeInvalideError(
                f"« {self._libelle} » doit être entre {self._minimum} et {self._maximum} "
                f"(reçu : {valeur})."
            )
        return valeur

    def __repr__(self) -> str:
        return f"VariableNumerique({self._nom!r}, [{self._minimum}, {self._maximum}])"


class VariableCategorielle(Variable):
    """Variable dont la valeur doit appartenir à une liste fixe."""

    def __init__(self, nom: str, libelle: str, valeurs: list[str]) -> None:
        super().__init__(nom, libelle)
        if not valeurs:
            raise ValueError("Une variable catégorielle a besoin d'au moins une valeur.")
        self._valeurs = list(valeurs)

    @property
    def valeurs(self) -> list[str]:
        return list(self._valeurs)

    def valider(self, saisie: Any) -> str:
        """Accepte uniquement une des valeurs connues."""
        texte = str(saisie).strip() if saisie is not None else ""
        if texte == "":
            raise DonneeInvalideError(f"« {self._libelle} » est vide.")
        if texte not in self._valeurs:
            raise DonneeInvalideError(
                f"« {self._libelle} » doit être l'une de : {', '.join(self._valeurs)}."
            )
        return texte

    def __repr__(self) -> str:
        return f"VariableCategorielle({self._nom!r}, {self._valeurs})"


LIBELLES: dict[str, str] = {
    "score_joueur": "Score de votre main",
    "score_croupier": "Carte visible du croupier (As = 11)",
    "nb_dix": "Nombre de cartes de valeur 10 visibles",
    "nb_cartes": "Nombre de cartes dans votre main",
    "as_souple": "Avez-vous un As compté comme 11 ?",
    "action": "Action envisagée",
}


def variables_depuis_metadonnees(metadonnees: dict) -> list[Variable]:
    """Construit la liste des variables à partir du JSON de métadonnées du modèle.

    Reçoit : le dictionnaire de métadonnées (clés « numeriques » et « categorielles »).
    Renvoie : une liste de Variable, dans l'ordre des caractéristiques du modèle.
    Échoue : DonneeInvalideError si une clé attendue manque.
    """
    try:
        numeriques = metadonnees["numeriques"]
        categorielles = metadonnees["categorielles"]
        ordre = metadonnees["caracteristiques"]
    except KeyError as erreur:
        raise DonneeInvalideError(f"Métadonnées incomplètes : clé {erreur} absente.") from None

    variables: dict[str, Variable] = {}
    for nom, bornes in numeriques.items():
        variables[nom] = VariableNumerique(nom, LIBELLES.get(nom, nom), bornes["min"], bornes["max"])
    for nom, valeurs in categorielles.items():
        variables[nom] = VariableCategorielle(nom, LIBELLES.get(nom, nom), valeurs)
    return [variables[nom] for nom in ordre]
