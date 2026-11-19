# Lab2-inf1007
données (CSV) → modèle entraîné → application avec interface graphique

## Données

**Provenance** : la sauvegarde (`donnees/agent_v3_model.bin`) de mon agent blackjack par
apprentissage par renforcement, fait lors d'un cours collégial :
https://github.com/Wproulx09/blackjack-agent. Elle contient, pour chaque situation de jeu et
chaque action, le nombre de victoires, défaites et égalités obtenues par l'agent.

Licence : données produites par nous, publiées sous licence
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (domaine public).


**Conversion** : le script décompresse le fichier et écrit une ligne par décision dans
`donnees/blackjack_decisions.csv` (4 181 lignes).

```bash
python outils/extraire_donnees_agent.py
```

**Colonnes**

| Colonne | Description |
|---|---|
| `score_joueur` | Valeur de la main du joueur |
| `score_croupier` | Carte visible du croupier (As = 11) |
| `as_souple` | oui / non : la main a un As compté 11 |
| `nb_dix` | Cartes de valeur 10 visibles |
| `cartes_utiles` | Cartes restantes qui aideraient la main (`very_few` à `many`) |
| `cartes_bust` | Cartes restantes qui feraient dépasser 21 (`low_risk` à `high_risk`) |
| `action` | tirer / rester |
| **`resultat`** | **Cible** : victoire / defaite / egalite |
