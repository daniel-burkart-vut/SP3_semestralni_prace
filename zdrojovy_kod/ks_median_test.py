import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt   # <--- přidat


def ks_median_test_children_vs_adults(model, column="poměr", alpha=0.05):
    """
    Kolmogorov-Smirnovův test shodnosti rozdělení (a nepřímo medianu)
    pro děti (0–20) a dospělé (20+).

    Používá se dvouvýběrový KS test na vybraném sloupci (např. 'poměr').
    """

    # ------------------------------------------------------------
    # Kontrola, že data jsou načtena
    # ------------------------------------------------------------
    df = model.data
    if df is None:
        print("Chyba: model.data je None – nejprve načti data.")
        return

    if column not in df.columns:
        print(f"Chyba: ve DataFrame chybí sloupec '{column}'.")
        return

    if "věk" not in df.columns:
        print("Chyba: ve DataFrame chybí sloupec 'věk'.")
        return

    # ------------------------------------------------------------
    # Rozdělení na děti (0–20) a dospělé (20+)
    # Předpokládáme věkové kategorie ve formátu '0-10', '10-20', ...
    # ------------------------------------------------------------
    children_categories = ["0-10", "10-20"]
    adults_categories = ["20-40", "40-60", "60-80"]

    children_mask = df["věk"].isin(children_categories)
    adults_mask = df["věk"].isin(adults_categories)

    children_values = df.loc[children_mask, column].dropna().astype(float).values
    adults_values = df.loc[adults_mask, column].dropna().astype(float).values

    # Kontrola, že máme dostatek dat v obou skupinách
    if len(children_values) < 5 or len(adults_values) < 5:
        print("Nedostatek dat pro KS test (méně než 5 pozorování v jedné ze skupin).")
        return

    # ------------------------------------------------------------
    # Výpočet medianů obou skupin
    # ------------------------------------------------------------
    children_median = np.median(children_values)
    adults_median = np.median(adults_values)

    # ------------------------------------------------------------
    # Kolmogorov-Smirnovův dvouvýběrový test
    # H0: rozdělení dětí a dospělých je stejné
    # H1: rozdělení se liší (two-sided)
    # ------------------------------------------------------------
    ks_stat, p_value = stats.ks_2samp(children_values, adults_values, alternative="two-sided")

    # ------------------------------------------------------------
    # Interpretace výsledku
    # ------------------------------------------------------------
    reject_H0 = p_value < alpha

    # ------------------------------------------------------------
    # Příprava textového výstupu
    # ------------------------------------------------------------
    lines = []
    lines.append("Kolmogorov-Smirnovův test shodnosti rozdělení (děti vs. dospělí)")
    lines.append("=" * 80)
    lines.append(f"Analyzovaný sloupec: {column}")
    lines.append("")
    lines.append("Definice skupin:")
    lines.append(f"  Děti: věk v kategoriích {children_categories}")
    lines.append(f"  Dospělí: věk v kategoriích {adults_categories}")
    lines.append("")
    lines.append(f"Počet pozorování (děti):    n_children = {len(children_values)}")
    lines.append(f"Počet pozorování (dospělí): n_adults   = {len(adults_values)}")
    lines.append("")
    lines.append(f"Median (děti):    {children_median:.4f}")
    lines.append(f"Median (dospělí): {adults_median:.4f}")
    lines.append("")
    lines.append("KS test:")
    lines.append(f"  Statistika D = {ks_stat:.4f}")
    lines.append(f"  p-hodnota    = {p_value:.6f}")
    lines.append(f"  Hladina významnosti alpha = {alpha}")
    lines.append("")

    if reject_H0:
        lines.append("Závěr: Zamítáme H0 o shodnosti rozdělení (a tedy i mediánů) mezi dětmi a dospělými.")
    else:
        lines.append("Závěr: Nezamítáme H0 – data neukazují na rozdíl v rozdělení (a tedy ani v mediánech) mezi dětmi a dospělými.")

    text = "\n".join(lines)

    # ------------------------------------------------------------
    # ECDF + vyznačení KS statistiky
    # ------------------------------------------------------------
    # seřazené hodnoty a ECDF (kroková funkce)
    children_sorted = np.sort(children_values)
    adults_sorted = np.sort(adults_values)
    ecdf_children_y = np.arange(1, len(children_sorted) + 1) / len(children_sorted)
    ecdf_adults_y = np.arange(1, len(adults_sorted) + 1) / len(adults_sorted)

    # mřížka všech hodnot pro přesné určení místa max. rozdílu
    all_values = np.sort(np.concatenate([children_sorted, adults_sorted]))
    F_children = np.searchsorted(children_sorted, all_values, side="right") / len(children_sorted)
    F_adults = np.searchsorted(adults_sorted, all_values, side="right") / len(adults_sorted)
    diff = np.abs(F_children - F_adults)
    max_idx = np.argmax(diff)
    x_star = all_values[max_idx]          # x, kde je |F1-F2| maximální
    D_at_x = diff[max_idx]                # mělo by být ≈ ks_stat

    # hodnoty ECDF v bodě x_star
    F_c_star = np.searchsorted(children_sorted, x_star, side="right") / len(children_sorted)
    F_a_star = np.searchsorted(adults_sorted, x_star, side="right") / len(adults_sorted)

    plt.figure(figsize=(8, 6))

    # ECDF jako krokové funkce
    plt.step(children_sorted, ecdf_children_y, where="post", label="Děti (ECDF)")
    plt.step(adults_sorted, ecdf_adults_y, where="post", label="Dospělí (ECDF)")

    # vertikální úsečka vyznačující KS rozdíl
    ks_color = "green"  

    plt.vlines(
        x_star,
        ymin=min(F_c_star, F_a_star),
        ymax=max(F_c_star, F_a_star),
        color=ks_color,
        linewidth=2,
        label=f"D = Max |F1 - F2| = {D_at_x:.4f}"
    )

    
    plt.title("Kolmogorov–Smirnov: ECDF dětí vs. dospělých")
    plt.xlabel(column)
    plt.ylabel("ECDF")
    plt.grid(alpha=0.3)
    plt.legend()

    # ------------------------------------------------------------
    # Výpis do konzole
    # ------------------------------------------------------------
    print("\n" + text + "\n")

    # ------------------------------------------------------------
    # Složka pro výstupy
    # ------------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(__file__))  # projektový root
    out_dir = os.path.join(base_dir, "vystupy", "statistika")
    os.makedirs(out_dir, exist_ok=True)

    plot_path = os.path.join(out_dir, "ks_ecdf_children_vs_adults.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Graf ECDF s vyznačenou KS statistikou uložen do: {plot_path}")

    # ------------------------------------------------------------
    # Uložení textového výstupu
    # ------------------------------------------------------------
    out_path = os.path.join(out_dir, "ks_median_children_vs_adults.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Textový výstup KS testu uložen do: {out_path}")
