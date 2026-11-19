"""Métriques calculées « à la main » avec NumPy, sans boucle Python (calcul vectorisé).

Les tests vérifient qu'elles coïncident avec celles de scikit-learn.
"""

import time
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np

journal = logging.getLogger(__name__)


def exactitude(y_vrai: np.ndarray, y_predit: np.ndarray) -> float:
    """Proportion de prédictions correctes.

    Reçoit : deux tableaux de même longueur.
    Renvoie : un flottant entre 0 et 1.
    Échoue : ValueError si les longueurs diffèrent ou si les tableaux sont vides.
    """
    y_vrai, y_predit = np.asarray(y_vrai), np.asarray(y_predit)
    if y_vrai.shape != y_predit.shape or y_vrai.size == 0:
        raise ValueError("Les tableaux doivent être non vides et de même taille.")
    return float(np.mean(y_vrai == y_predit))


def f1_macro(y_vrai: np.ndarray, y_predit: np.ndarray) -> float:
    """F1 moyen sur toutes les classes (chaque classe pèse autant).

    Le calcul est vectorisé : une matrice booléenne (classes x observations)
    remplace la boucle sur les classes.
    Renvoie : un flottant entre 0 et 1.
    Échoue : ValueError si les longueurs diffèrent ou si les tableaux sont vides.
    """
    y_vrai, y_predit = np.asarray(y_vrai), np.asarray(y_predit)
    if y_vrai.shape != y_predit.shape or y_vrai.size == 0:
        raise ValueError("Les tableaux doivent être non vides et de même taille.")
    classes = np.union1d(y_vrai, y_predit)[:, None]  # forme (k, 1)
    est_vrai = y_vrai[None, :] == classes  # forme (k, n)
    est_predit = y_predit[None, :] == classes
    vp = np.sum(est_vrai & est_predit, axis=1)
    fp = np.sum(~est_vrai & est_predit, axis=1)
    fn = np.sum(est_vrai & ~est_predit, axis=1)
    denominateur = 2 * vp + fp + fn
    f1 = np.divide(2 * vp, denominateur, out=np.zeros(len(vp)), where=denominateur > 0)
    return float(np.mean(f1))


def chronometrer(fonction: Callable) -> Callable:
    """Décorateur qui mesure la durée d'exécution et la consigne dans le journal."""

    @wraps(fonction)
    def enveloppe(*args: Any, **kwargs: Any) -> Any:
        debut = time.perf_counter()
        resultat = fonction(*args, **kwargs)
        duree = time.perf_counter() - debut
        journal.info("%s exécuté en %.2f s", fonction.__name__, duree)
        return resultat

    return enveloppe
