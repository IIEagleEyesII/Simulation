# Projet : décimales de pi et tests pseudo-aléatoires

Ce projet étudie expérimentalement les décimales de pi avec des tests statistiques simples. Il construit aussi un générateur uniforme sur `[0,1[` à partir de ces décimales, puis le compare à `random.random()` de Python.

## Fichiers principaux

- `main.py` : script principal. Il lit les données, applique les tests, affiche les résultats et génère les figures.
- `pi.txt` : fichier contenant les décimales de pi.
- `report.tex` : rapport LaTeX final.
- `figures/` : dossier des figures générées automatiquement.
- `out/report.pdf` : PDF compilé du rapport, si LaTeX a été lancé localement.

## Méthode

Le script commence par étudier directement les chiffres des décimales de pi :

- test du chi-deux sur les fréquences des chiffres `0,...,9` ;
- gap test sur les distances entre apparitions successives d'un chiffre cible.

Ensuite, il construit un générateur uniforme basé sur pi. Les décimales sont regroupées en blocs non chevauchants de taille `k`. Chaque bloc est transformé en entier, puis divisé par `10^k`.

Exemple avec `k = 6` :

```text
141592 -> 141592 / 10^6 = 0.141592
```

Le générateur ainsi obtenu est comparé à `random.random()` avec trois tests :

- test du chi-deux sur dix classes de `[0,1[` ;
- test de Kolmogorov-Smirnov contre la loi uniforme ;
- gap test sur l'événement `0.2 <= U < 0.3`.

## Exécution

Depuis le dossier du projet :

```sh
python3 main.py
```

Il est aussi possible de fournir un autre fichier de décimales :

```sh
python3 main.py autre_fichier.txt
```

## Dépendances

Le script utilise `scipy` pour calculer les statistiques et les p-values des tests. Il utilise `matplotlib` pour générer les figures.

```sh
python3 -m pip install scipy matplotlib
```

ou directement :

```sh
python3 -m pip install -r requirements.txt
```

Si `matplotlib` n'est pas installé, le script peut quand même afficher les résultats numériques, mais les figures ne sont pas régénérées. En revanche, `scipy` est nécessaire pour exécuter les tests statistiques.

## Interprétation statistique

Les tests sont interprétés au seuil de 5 %. Une p-value inférieure à 5 % conduit à rejeter l'hypothèse nulle pour le test considéré. Une p-value supérieure à 5 % ne prouve pas l'hypothèse nulle ; elle signifie seulement que les résultats observés sont compatibles avec elle.

Le projet ne cherche donc pas à prouver que les décimales de pi sont réellement aléatoires. Il vérifie seulement si elles présentent, sur l'échantillon étudié, des comportements compatibles avec ceux attendus d'une suite pseudo-aléatoire.
