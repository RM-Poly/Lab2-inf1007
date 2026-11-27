# Auteurs : Claude (Claude Code), Wproulx09
"""Exceptions propres au projet : des erreurs claires plutôt qu'un plantage."""


class DonneeInvalideError(ValueError):
    """Une donnée (fichier, colonne ou saisie) ne respecte pas le format attendu."""


class FichierIntrouvableError(FileNotFoundError):
    """Un fichier nécessaire (données, modèle) n'existe pas."""
