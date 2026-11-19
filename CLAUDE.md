# Règles du projet

## Journal IA (OBLIGATOIRE!)

Après chaque tâche qui modifie le repo, ajoute une ligne au tableau de `Journal_IA.md`, avec les colonnes suivantes :

Outil | Utilisateur | Ce que nous avons demandé | Réponse reçue | Vérification faite | Décision.

- Outil : " Claude (Claude Code) "
- Utilisateur : prends le résultat de `git config user.name`.
- Ne remplis JAMAIS `Vérification faite` ni `Décision` à la place de l'équipe. Tu dois écrire `#TODO` à la place.

## En-tête des fichiers de code (OBLIGATOIRE!)

Ajoute toujours, au tout début de chaque fichier de code que tu crées, l'en-tête suivant :

```python
# Auteurs : Claude (Claude Code), <nom>
```

- `<nom>` est la personne qui t'a demandé le fichier : prends le résultat de `git config user.name`.
- Si tu modifies un fichier qui a déjà cet en-tête et que la personne n'y est pas, ajoute son nom à la liste.
