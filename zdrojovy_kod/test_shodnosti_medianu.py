import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def Wilcoxon(model, column="poměr", alpha=0.05):

    df = model.data
    
    if df is None:
        print(" Chyba: model.data je None – nejprve načti data.")
        return None

    if column not in df.columns:
        print(f"Chyba: ve DataFrame chybí sloupec '{column}'.")
        return None

    if "věk" not in df.columns:
        print("Chyba: ve DataFrame chybí sloupec 'věk'.")
        return None

    # Definice věkových kategorií
    # Děti (0-20)
    children_categories = ["0-10", "10-20"]
    # Dospělí (20+)
    adults_categories = ["20-40", "40-60", "60-80"]

    # Vytvoření masky pro výběr dat
    children_mask = df["věk"].isin(children_categories)
    adults_mask = df["věk"].isin(adults_categories)

    # Vybrané hodnoty jako float pole, bez chybějících hodnot
    children_values = df.loc[children_mask, column].dropna().astype(float).values
    adults_values = df.loc[adults_mask, column].dropna().astype(float).values
    
    n_children = len(children_values)
    n_adults = len(adults_values)
    
    
    print("\n===== 📊 MANNŮV–WHITNEYŮV U-TEST (DVOUVÝBĚROVÝ WILCOXON) =====")
    print(f"Skupina Děti (0-20): N={n_children}, Medián={np.median(children_values):.4f}")
    print(f"Skupina Dospělí (20+): N={n_adults}, Medián={np.median(adults_values):.4f}")
    
    if n_children < 5 or n_adults < 5:
        print("!!! Недостаточно данных (N < 5) в одной из групп для надежного теста. !!!")
        print(f"Nedostatek Dvouvýběrový Wilcoxon test (Děti: {n_children}, Dospělí: {n_adults}).")
        return None

    
    # ------------------------------------------------------------
    # Dvouvýběrový Wilcoxon
    # ------------------------------------------------------------
    # H₀: Rozdělení (poloha) obou skupin je shodná.
    
    u_statistic, p_mw = mannwhitneyu(children_values, adults_values, alternative='two-sided')
    
    print(f"U-statistika: {u_statistic:.4f}, P-hodnota: {p_mw:.4f}")
    
    if p_mw < alpha:
        mw_conclusion = f"Zamítáme H₀ (p < {alpha}). Poloha rozdělení poměru X/Y se statisticky významně liší."
    else:
        mw_conclusion = f"Nezamítáme H₀ (p ≥ {alpha}). Poloha rozdělení poměru X/Y je shodná."
    print(f"Závěr: {mw_conclusion}")
    print("========================================================\n")
    
    return {
        "n_children": n_children,
        "n_adults": n_adults,
        "u_statistic": u_statistic,
        "p_value": p_mw,
        "conclusion": mw_conclusion
    }