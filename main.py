"""Analyse statistique des decimales de pi.

Ce script realise tout le traitement utilise dans le rapport :

1. lecture des decimales de pi depuis ``pi.txt`` ;
2. test du chi-deux sur les frequences des chiffres 0,...,9 ;
3. gap test sur les chiffres de pi ;
4. construction d'un generateur uniforme sur [0,1[ a partir de blocs
   non chevauchants de k chiffres de pi ;
5. comparaison de ce generateur avec ``random.random()`` de Python.

Le generateur base sur pi est volontairement simple :
un bloc de k chiffres est transforme en entier, puis divise par 10^k.
Par exemple, avec k=6, le bloc 141592 devient 0.141592.

Les conclusions statistiques sont formulees au seuil de 5 %. Une p-value
superieure au seuil ne valide pas H0 ; elle indique seulement que les
observations sont compatibles avec H0 pour le test utilise.
"""

import os
import sys
import random

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from scipy.stats import chisquare, kstest
except ModuleNotFoundError as error:
    raise SystemExit(
        "scipy is required for the statistical p-values.\n"
        "Install it with: python3 -m pip install scipy"
    ) from error


ALPHA = 0.05
FIGURES_DIR = "figures"


# ---------------------------------------------------------------------------
# Lecture des donnees et aide a l'interpretation statistique
# ---------------------------------------------------------------------------

def load_pi_digits(filepath):
    """Charge les decimales de pi sous forme de liste d'entiers.

    Le fichier peut contenir des espaces, des retours a la ligne ou le
    separateur decimal. On conserve uniquement les chiffres. Si le fichier
    commence par ``3.`` ou ``3,``, le chiffre 3 est retire pour que l'etude
    porte uniquement sur les decimales.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    digits = [int(char) for char in text if char.isdigit()]

    # Le fichier commence par 3. On ne garde que les decimales.
    if digits and digits[0] == 3:
        digits = digits[1:]

    return digits


def conclusion(p_value, alpha=ALPHA):
    """Retourne la conclusion correcte d'un test d'hypothese.

    On ne dit jamais que H0 est vraie. On dit seulement que l'on rejette ou
    que l'on ne rejette pas H0 au seuil choisi.
    """
    if p_value < alpha:
        return f"On rejette H0 au seuil de {alpha * 100:.0f}%."
    return f"On ne rejette pas H0 au seuil de {alpha * 100:.0f}%."


# ---------------------------------------------------------------------------
# Fonctions statistiques de base
# ---------------------------------------------------------------------------
#Used the library because don't want to bother implementing a p-value calculator

def chi_square_test(observed, expected):
    """Calcule la statistique du chi-deux et sa p-value.

    La p-value est la probabilite P(X >= statistique), ou X suit une loi
    du chi-deux a r-1 degres de liberte. Le calcul est delegue a scipy pour
    garder le script simple et lisible.
    """
    statistic, p_value = chisquare(f_obs=observed, f_exp=expected)
    return statistic, p_value


def chi_squared_digits(digits):
    """Test du chi-deux pour l'uniformite des chiffres 0,...,9."""
    observed = [0] * 10
    for digit in digits:
        observed[digit] += 1

    expected = [len(digits) / 10] * 10
    statistic, p_value = chi_square_test(observed, expected)
    return observed, expected, statistic, p_value


def pi_uniform_generator(digits, block_size):
    """Construit des valeurs de [0,1[ a partir de blocs de decimales de pi.

    Les blocs sont non chevauchants. Si ``block_size`` vaut 6, les chiffres
    sont lus par paquets de 6 : le premier paquet donne la premiere valeur,
    le deuxieme paquet donne la deuxieme valeur, etc. Les chiffres restants
    en fin de fichier sont ignores s'ils ne forment pas un bloc complet.
    """
    values = []
    number_of_blocks = len(digits) // block_size

    for i in range(number_of_blocks):
        block = digits[i * block_size : (i + 1) * block_size]
        integer_value = 0

        for digit in block:
            integer_value = 10 * integer_value + digit

        values.append(integer_value / (10**block_size))

    return values


def chi_squared_uniform(values, number_of_classes=10):
    """Test du chi-deux sur des classes de meme largeur de [0,1[."""
    observed = [0] * number_of_classes

    for value in values:
        # La classe i correspond a [i/m, (i+1)/m[, avec m classes.
        index = int(value * number_of_classes)
        if index == number_of_classes:
            index = number_of_classes - 1
        observed[index] += 1

    expected = [len(values) / number_of_classes] * number_of_classes
    statistic, p_value = chi_square_test(observed, expected)
    return observed, expected, statistic, p_value


def ks_uniform(values):
    """Test de Kolmogorov-Smirnov contre la loi uniforme sur [0,1[.

    La statistique est la distance maximale entre la fonction de repartition
    empirique et F(x)=x. scipy calcule la statistique et la p-value.
    """
    statistic, p_value = kstest(values, "uniform")
    return statistic, p_value


def gap_test_events(events, event_probability, max_gap):
    """Gap test generique pour une suite d'evenements booleens.

    ``events`` contient True lorsque l'evenement etudie se produit. Le gap
    est le nombre d'observations sans evenement entre deux occurrences
    successives. Sous independance, il suit une loi geometrique :
    P(G=m)=p(1-p)^m.
    """
    gaps = []
    current_gap = 0
    found_first_event = False

    for event in events:
        if event:
            if found_first_event:
                gaps.append(current_gap)
            found_first_event = True
            current_gap = 0
        elif found_first_event:
            current_gap += 1

    observed = [0] * (max_gap + 2)
    for gap in gaps:
        if gap <= max_gap:
            observed[gap] += 1
        else:
            observed[-1] += 1

    number_of_gaps = len(gaps)
    expected = []
    for gap in range(max_gap + 1):
        probability = event_probability * ((1 - event_probability) ** gap)
        expected.append(number_of_gaps * probability)
    # Derniere classe : tous les ecarts strictement superieurs a max_gap.
    expected.append(number_of_gaps * ((1 - event_probability) ** (max_gap + 1)))

    statistic, p_value = chi_square_test(observed, expected)
    return observed, expected, statistic, p_value


def gap_test_digits(digits, target_digit, max_gap=17):
    """Gap test pour l'apparition d'un chiffre cible dans les decimales."""
    events = [digit == target_digit for digit in digits]
    return gap_test_events(events, event_probability=0.1, max_gap=max_gap)


def gap_test_uniform(values, interval_start=0.2, interval_end=0.3, max_gap=17):
    """Gap test pour l'evenement interval_start <= U < interval_end."""
    events = [interval_start <= value < interval_end for value in values]
    event_probability = interval_end - interval_start
    return gap_test_events(events, event_probability, max_gap)


def ks_by_block_size(digits, block_sizes):
    """Applique le test KS au generateur pi pour plusieurs tailles de blocs."""
    results = []

    for block_size in block_sizes:
        values = pi_uniform_generator(digits, block_size)
        statistic, p_value = ks_uniform(values)
        results.append((block_size, len(values), statistic, p_value))

    return results


def print_chi_squared_digits(digits):
    """Affiche et trace le test du chi-deux sur les decimales de pi."""
    observed, expected, statistic, p_value = chi_squared_digits(digits)
    expected_each = expected[0]

    print("1) Test du chi-deux sur les decimales de pi")
    print("H0: les chiffres 0,...,9 sont uniformement repartis.")
    for digit, count in enumerate(observed):
        print(f"{digit}: {count} ({count - expected_each:+.2f})")
    print(f"Statistique = {statistic:.6f}")
    print(f"p-value = {p_value:.6f}")
    print("Conclusion:", conclusion(p_value))
    print()

    plot_digit_frequencies(observed, expected_each, statistic, p_value)


def print_gap_digits(digits):
    """Affiche le gap test sur les chiffres et genere les figures associees."""
    max_gap = 17
    target_digit = 7
    observed, expected, statistic, p_value = gap_test_digits(digits, target_digit, max_gap)

    print("2) Gap test sur les decimales de pi")
    print(f"H0: les apparitions du chiffre {target_digit} suivent le modele geometrique attendu.")
    print_gap_counts(observed, max_gap)
    print(f"Statistique = {statistic:.6f}")
    print(f"p-value = {p_value:.6f}")
    print("Conclusion:", conclusion(p_value))
    print()

    plot_gap_test(
        observed,
        expected,
        max_gap,
        title=f"Gap test pour le chiffre cible {target_digit}",
        filepath=f"{FIGURES_DIR}/gap_target_{target_digit}.png",
    )

    results = []
    for target_digit in range(10):
        observed, expected, statistic, p_value = gap_test_digits(digits, target_digit, max_gap)
        results.append((target_digit, observed, expected, statistic, p_value))

    print("Gap test pour chaque chiffre cible")
    print("chiffre | statistique | p-value | conclusion")
    for target_digit, observed, expected, statistic, p_value in results:
        print(f"{target_digit} | {statistic:.6f} | {p_value:.6f} | {conclusion(p_value)}")
    print()

    plot_all_digit_gap_tests(results, max_gap)


def print_pi_generator_study(digits, block_size):
    """Etudie le generateur base sur pi pour une taille de bloc donnee."""
    values = pi_uniform_generator(digits, block_size)

    print("3) Generateur uniforme construit avec les decimales de pi")
    print("Construction: blocs non chevauchants de k chiffres, entier du bloc, division par 10^k.")
    print(f"k = {block_size}")
    print(f"Nombre de valeurs generees = {len(values)}")
    print()

    print_uniform_tests("Generateur pi", values)

    results = ks_by_block_size(digits, range(1, 10))
    print("Etude KS du generateur pi en fonction de k")
    print("k | valeurs | statistique | p-value | conclusion")
    for k, number_of_values, statistic, p_value in results:
        print(f"{k} | {number_of_values} | {statistic:.6f} | {p_value:.6f} | {conclusion(p_value)}")
    print()

    plot_ks_by_block_size(results)
    return values


def print_python_generator_study(number_of_values, seed=12345):
    """Genere et etudie les valeurs de random.random().

    La graine est fixee pour que les resultats soient reproductibles :
    deux executions du script donnent les memes valeurs Python.
    """
    random.seed(seed)
    values = [random.random() for _ in range(number_of_values)]

    print("4) Generateur random.random() de Python")
    print(f"Graine utilisee pour rendre les resultats reproductibles: {seed}")
    print(f"Nombre de valeurs generees = {len(values)}")
    print()

    print_uniform_tests("Generateur Python", values)
    return values


def print_uniform_tests(name, values):
    """Applique les trois tests utilises pour une suite de valeurs dans [0,1[."""
    observed, expected, statistic, p_value = chi_squared_uniform(values, number_of_classes=10)

    print(f"Test du chi-deux sur classes de [0,1[ - {name}")
    print("H0: les valeurs sont uniformement reparties dans les classes.")
    for index, count in enumerate(observed):
        start = index / 10
        end = (index + 1) / 10
        print(f"[{start:.1f},{end:.1f}[: {count}")
    print(f"Statistique = {statistic:.6f}")
    print(f"p-value = {p_value:.6f}")
    print("Conclusion:", conclusion(p_value))
    print()

    statistic, p_value = ks_uniform(values)
    print(f"Test de Kolmogorov-Smirnov - {name}")
    print("H0: les valeurs suivent une loi uniforme sur [0,1[.")
    print(f"Statistique = {statistic:.6f}")
    print(f"p-value = {p_value:.6f}")
    print("Conclusion:", conclusion(p_value))
    print()

    max_gap = 17
    interval_start = 0.2
    interval_end = 0.3
    observed, expected, statistic, p_value = gap_test_uniform(
        values,
        interval_start=interval_start,
        interval_end=interval_end,
        max_gap=max_gap,
    )

    print(f"Gap test sur l'evenement {interval_start:.1f} <= U < {interval_end:.1f} - {name}")
    print("H0: les ecarts suivent le modele geometrique attendu.")
    print_gap_counts(observed, max_gap)
    print(f"Statistique = {statistic:.6f}")
    print(f"p-value = {p_value:.6f}")
    print("Conclusion:", conclusion(p_value))
    print()

def print_gap_counts(observed, max_gap):
    """Affiche les effectifs observes du gap test."""
    for index, count in enumerate(observed):
        label = str(index) if index <= max_gap else f"{max_gap + 1}+"
        print(f"ecart {label}: {count}")


def plot_digit_frequencies(observed, expected_each, statistic, p_value):
    """Trace les frequences des chiffres de pi avec ecarts et p-value."""
    if plt is None:
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    digits = list(range(10))
    n = sum(observed)

    plt.figure(figsize=(8, 5))
    plt.bar(digits, observed, label="Frequences observees")
    plt.axhline(expected_each, color="red", linestyle="--", linewidth=0.8, label="Frequence attendue")

    offset = max(1, 0.0015 * n)
    for digit, count in enumerate(observed):
        difference = count - expected_each
        if difference >= 0:
            y_position = count + offset
            vertical_alignment = "bottom"
        else:
            y_position = count - offset
            vertical_alignment = "top"
        plt.text(
            digit,
            y_position,
            f"ecart={difference:+.0f}",
            ha="center",
            va=vertical_alignment,
            fontsize=8,
        )

    plt.xticks(digits)
    plt.xlabel("Chiffre")
    plt.ylabel("Nombre d'apparitions")
    plt.title(
        "Test du chi-deux sur les decimales de pi\n"
        f"statistique={statistic:.3f}, p-value={p_value:.3f}"
    )
    plt.ylim(min(observed) - 3 * offset, max(observed) + 4 * offset)
    plt.legend()
    plt.savefig(f"{FIGURES_DIR}/chi_squared.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_uniform_frequencies(observed, expected, title, filepath):
    """Trace un histogramme de classes uniformes de [0,1[.

    Cette fonction est conservee comme utilitaire general ; les figures de
    comparaison finales utilisent plutot les fonctions groupees par test.
    """
    if plt is None:
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    classes = list(range(len(observed)))
    labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in classes]

    plt.figure(figsize=(8, 5))
    plt.bar(classes, observed, label="Frequences observees")
    plt.axhline(expected, color="red", linestyle="--", linewidth=0.8, label="Frequence attendue")
    plt.xticks(classes, labels, rotation=45)
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'observations")
    plt.title(title)
    plt.legend()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gap_test(observed, expected, max_gap, title, filepath):
    """Trace les effectifs observes et attendus pour un gap test."""
    if plt is None:
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    labels = [str(i) for i in range(max_gap + 1)]
    labels.append(f"{max_gap + 1}+")

    x_values = list(range(len(labels)))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([x - width / 2 for x in x_values], observed, width=width, label="Observes")
    plt.bar([x + width / 2 for x in x_values], expected, width=width, label="Attendus")
    plt.xticks(x_values, labels)
    plt.xlabel("Longueur de l'ecart")
    plt.ylabel("Nombre d'observations")
    plt.title(title)
    plt.legend()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_digit_gap_tests(results, max_gap):
    """Genere une figure de gap test par chiffre et une synthese des p-values."""
    if plt is None:
        return

    folder = f"{FIGURES_DIR}/gap_tests"
    os.makedirs(folder, exist_ok=True)

    targets = []
    p_values = []

    for target_digit, observed, expected, statistic, p_value in results:
        plot_gap_test(
            observed,
            expected,
            max_gap,
            title=f"Gap test pour le chiffre cible {target_digit}",
            filepath=f"{folder}/gap_target_{target_digit}.png",
        )
        targets.append(target_digit)
        p_values.append(p_value)

    plt.figure(figsize=(8, 5))
    plt.bar(targets, p_values)
    plt.axhline(ALPHA, color="red", linestyle="--", linewidth=0.8, label="Seuil 5%")
    plt.xticks(targets)
    plt.xlabel("Chiffre cible")
    plt.ylabel("p-value")
    plt.title("p-values du gap test par chiffre cible")
    plt.legend()
    plt.savefig(f"{folder}/gap_pvalues_by_target.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ks_by_block_size(results):
    """Trace l'effet de la taille des blocs sur le test KS du generateur pi."""
    if plt is None:
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    block_sizes = [row[0] for row in results]
    statistics = [row[2] for row in results]
    p_values = [row[3] for row in results]

    plt.figure(figsize=(8, 5))
    plt.plot(block_sizes, statistics, marker="o")
    plt.xticks(block_sizes)
    plt.xlabel("Nombre de chiffres par bloc (k)")
    plt.ylabel("Statistique KS")
    plt.title("Statistique KS en fonction de k")
    plt.savefig(f"{FIGURES_DIR}/ks_statistic_by_k.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(block_sizes, p_values, marker="o")
    plt.axhline(ALPHA, color="red", linestyle="--", linewidth=0.8, label="Seuil 5%")
    plt.xticks(block_sizes)
    plt.xlabel("Nombre de chiffres par bloc (k)")
    plt.ylabel("p-value")
    plt.title("p-value du test KS en fonction de k")
    plt.legend()
    plt.savefig(f"{FIGURES_DIR}/ks_pvalue_by_k.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_generator_comparison(pi_values, python_values):
    """Genere les figures de comparaison entre le generateur pi et Python.

    Les figures sont groupees par test : une figure pour le chi-deux, une
    pour Kolmogorov-Smirnov et une pour le gap test.
    """
    if plt is None:
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_generator_chi_squared_comparison(pi_values, python_values)
    plot_generator_ks_comparison(pi_values, python_values)
    plot_generator_gap_comparison(pi_values, python_values)


def plot_generator_chi_squared_comparison(pi_values, python_values):
    """Compare les deux generateurs avec le test du chi-deux par classes."""
    pi_observed, pi_expected, pi_statistic, pi_p_value = chi_squared_uniform(pi_values)
    python_observed, python_expected, python_statistic, python_p_value = chi_squared_uniform(python_values)

    classes = list(range(10))
    labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in classes]
    width = 0.35

    plt.figure(figsize=(11, 5.5))
    pi_positions = [x - width / 2 for x in classes]
    python_positions = [x + width / 2 for x in classes]
    plt.bar(pi_positions, pi_observed, width=width, label="Generateur pi")
    plt.bar(python_positions, python_observed, width=width, label="Python")
    plt.axhline(pi_expected[0], color="red", linestyle="--", linewidth=0.8, label="Frequence attendue")

    offset = max(1, 0.002 * len(pi_values))
    for position, count in zip(pi_positions, pi_observed):
        difference = count - pi_expected[0]
        plt.text(
            position,
            count + offset,
            f"{difference:+.0f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
        )
    for position, count in zip(python_positions, python_observed):
        difference = count - python_expected[0]
        plt.text(
            position,
            count + offset,
            f"{difference:+.0f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
        )

    plt.xticks(classes, labels, rotation=45)
    plt.xlabel("Classe de [0,1[")
    plt.ylabel("Nombre d'observations")
    plt.title(
        "Chi-deux sur classes de [0,1[\n"
        f"pi: stat={pi_statistic:.3f}, p={pi_p_value:.3f} | "
        f"Python: stat={python_statistic:.3f}, p={python_p_value:.3f}"
    )
    plt.ylim(0, max(max(pi_observed), max(python_observed)) + 5 * offset)
    plt.legend()
    plt.savefig(f"{FIGURES_DIR}/comparison_chi_squared_uniform.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_generator_ks_comparison(pi_values, python_values):
    """Compare les statistiques et p-values KS des deux generateurs."""
    labels = ["pi", "Python"]
    pi_statistic, pi_p_value = ks_uniform(pi_values)
    python_statistic, python_p_value = ks_uniform(python_values)
    statistics = [pi_statistic, python_statistic]
    p_values = [pi_p_value, python_p_value]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, statistics)
    axes[0].set_ylabel("Statistique KS")
    axes[0].set_title("Statistique")

    axes[1].bar(labels, p_values)
    axes[1].axhline(ALPHA, color="red", linestyle="--", linewidth=0.8, label="Seuil 5%")
    axes[1].set_ylabel("p-value")
    axes[1].set_title("p-value")
    axes[1].legend()

    fig.suptitle("Test de Kolmogorov-Smirnov: comparaison des generateurs")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/comparison_ks_uniform.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_generator_gap_comparison(pi_values, python_values):
    """Compare les deux generateurs avec le gap test sur [0.2, 0.3[."""
    max_gap = 17
    interval_start = 0.2
    interval_end = 0.3

    pi_observed, pi_expected, pi_statistic, pi_p_value = gap_test_uniform(
        pi_values,
        interval_start=interval_start,
        interval_end=interval_end,
        max_gap=max_gap,
    )
    python_observed, python_expected, python_statistic, python_p_value = gap_test_uniform(
        python_values,
        interval_start=interval_start,
        interval_end=interval_end,
        max_gap=max_gap,
    )

    labels = [str(i) for i in range(max_gap + 1)]
    labels.append(f"{max_gap + 1}+")
    x_values = list(range(len(labels)))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar([x - width for x in x_values], pi_observed, width=width, label="Generateur pi observe")
    plt.bar(x_values, python_observed, width=width, label="Python observe")
    plt.bar([x + width for x in x_values], pi_expected, width=width, label="Attendu", alpha=0.75)
    plt.xticks(x_values, labels)
    plt.xlabel("Longueur de l'ecart")
    plt.ylabel("Nombre d'observations")
    plt.title(
        f"Gap test sur {interval_start:.1f} <= U < {interval_end:.1f}\n"
        f"pi: stat={pi_statistic:.3f}, p={pi_p_value:.3f} | "
        f"Python: stat={python_statistic:.3f}, p={python_p_value:.3f}"
    )
    plt.legend()
    plt.savefig(f"{FIGURES_DIR}/comparison_gap_uniform.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Point d'entree du projet.

    Par defaut, le script lit ``pi.txt`` dans le dossier courant. Il est
    aussi possible de donner un autre fichier en argument :

        python3 main.py autre_fichier.txt

    Le script affiche les resultats dans le terminal et regenere les figures
    dans le dossier ``figures/`` lorsque matplotlib est installe.
    """
    filepath = "pi.txt"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]

    block_size = 6
    digits = load_pi_digits(filepath)

    print(f"Nombre de decimales chargees: {len(digits)}")
    print()

    print_chi_squared_digits(digits)
    print_gap_digits(digits)

    pi_values = print_pi_generator_study(digits, block_size)
    python_values = print_python_generator_study(len(pi_values))
    plot_generator_comparison(pi_values, python_values)

    if plt is None:
        print("matplotlib n'est pas installe: les calculs sont affiches, mais les figures ne sont pas regenerees.")
    else:
        print("Figures generees dans le dossier figures/.")


if __name__ == "__main__":
    main()
