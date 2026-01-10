import numpy as np
from scipy.stats import gaussian_kde, norm


def asymptotic_median_ratio(model, column_x="X", column_y="Y"):
    """
    Vypočítá asymptotické rozdělení mediánu poměru X/Y.
    Vrací medián, směrodatnou odchylku a 95% interval spolehlivosti.
    """

    df = model.data

    # Zkontrolujeme, zda sloupce existují
    if column_x not in df.columns or column_y not in df.columns:
        raise ValueError(f"Sloupce '{column_x}' nebo '{column_y}' nebyly nalezeny v datech.")

    # Poměr
    ratio = df[column_x] / df[column_y]
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

    n = len(ratio)
    if n < 10:
        raise ValueError("Nedostatek dat pro spolehlivý asymptotický odhad mediánu.")

    # MEDIÁN
    med = np.median(ratio)

    # KDE – odhad hustoty f(m)
    kde = gaussian_kde(ratio)
    f_m = kde.evaluate(med)[0]

    # Asymptotická varianace mediánu
    # Var = 1 / (4 n f(m)^2)
    var = 1 / (4 * n * f_m ** 2)
    sd = np.sqrt(var)

    # 95% interval spolehlivosti
    z = norm.ppf(0.975)
    ci_low = med - z * sd
    ci_high = med + z * sd

    # Výpis výsledků
    print("\n===== ASYMPTOTICKÉ ROZDĚLENÍ MEDIÁNU X/Y =====")
    print(f"Počet pozorování: {n}")
    print(f"Medián poměru X/Y: {med:.5f}")
    print(f"Odhad hustoty f(m): {f_m:.5f}")
    print(f"Asymptotická směrodatná odchylka: {sd:.5f}")
    print(f"95% interval spolehlivosti: ({ci_low:.5f}, {ci_high:.5f})")
    print("==============================================\n")

    return {
        "median": med,
        "sd": sd,
        "ci_low": ci_low,
        "ci_high": ci_high
    }
