import os
import numpy as np
from scipy import stats


def normal_parameter_estimates(data, column, confidence=0.95):
    """
    Funkce provede:
    - odhad střední hodnoty, rozptylu a směrodatné odchylky
    - výpočet intervalu spolehlivosti pro:
        * střední hodnotu (t-rozdělení)
        * rozptyl       (chi-kvadrát)
        * směrodatnou odchylku (ze CI pro rozptyl)

    data:  pandas DataFrame
    column: název numerického sloupce
    confidence: hladina spolehlivosti (např. 0.95 pro 95 %)
    """

    # vytažení hodnot sloupce a převod na numpy pole
    x = data[column].dropna().astype(float).values
    n = len(x)

    if n < 2:
        raise ValueError(f"Nedostatek dat pro odhad: {column}")

    # ----------------------------------------------------
    # ODHADY PARAMETRŮ
    # ----------------------------------------------------
    mu_hat = np.mean(x)              # odhad střední hodnoty
    sigma2_hat = np.var(x, ddof=1)   # odhad rozptylu (výběrový)
    sigma_hat = np.sqrt(sigma2_hat)  # odhad směrodatné odchylky

    # ----------------------------------------------------
    # INTERVAL SPOLEHLIVOSTI PRO STŘEDNÍ HODNOTU
    # ----------------------------------------------------
    alpha = 1 - confidence

    # kritická hodnota t-rozdělení
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    # směrodatná chyba odhadu střední hodnoty
    se = sigma_hat / np.sqrt(n)

    # CI pro střední hodnotu:  m ± t * SE
    mean_CI = (mu_hat - t_crit * se,
               mu_hat + t_crit * se)

    # ----------------------------------------------------
    # INTERVAL SPOLEHLIVOSTI PRO ROZPTYL
    # (n-1)*S^2 / sigma^2 ~ chi-kvadrát(df = n-1)
    # ----------------------------------------------------
    chi2_lower = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n - 1)

    var_CI = ((n - 1) * sigma2_hat / chi2_upper,
              (n - 1) * sigma2_hat / chi2_lower)

    # ----------------------------------------------------
    # INTERVAL SPOLEHLIVOSTI PRO SMĚRODATNOU ODCHYLKU
    # stačí odmocnit CI pro rozptyl
    # ----------------------------------------------------
    std_CI = (np.sqrt(var_CI[0]), np.sqrt(var_CI[1]))

    return {
        "n": n,
        "mu_hat": mu_hat,
        "sigma2_hat": sigma2_hat,
        "sigma_hat": sigma_hat,
        "CI_mean": mean_CI,
        "CI_variance": var_CI,
        "CI_std": std_CI
    }


def vypocet_odhadu_normalnich_parametru(model, confidence=0.95):
    """
    Hlavní funkce kompatibilní s main.py:
    - očekává, že model.data je pandas DataFrame
    - spočítá odhady parametrů pro vybrané sloupce
    - vytiskne výsledky do konzole
    - uloží výsledky do textového souboru ve složce 'vystupy/statistika'
    """

    df = model.data
    if df is None:
        print("Chyba: model.data je None – nejprve načti data.")
        return

    # vybereme sloupce, které v datech existují
    analyzovane_sloupce = [c for c in ["A", "B", "poměr", "odchylka"] if c in df.columns]
    if not analyzovane_sloupce:
        print("V datech nejsou žádné očekávané numerické sloupce (A, B, poměr, odchylka).")
        return

    # připravíme výstupní složku a soubor
    base_dir = os.path.dirname(os.path.dirname(__file__))  # projektový root
    vystup_dir = os.path.join(base_dir, "vystupy", "statistika")
    os.makedirs(vystup_dir, exist_ok=True)

    out_path = os.path.join(vystup_dir, "odhady_normalni_rozdelni.txt")

    lines = []
    lines.append(f"Odhady parametrů normálního rozdělení (hladina spolehlivosti {int(confidence*100)} %)")
    lines.append("=" * 80)

    for col in analyzovane_sloupce:
        try:
            res = normal_parameter_estimates(df, col, confidence=confidence)
        except ValueError as e:
            lines.append(f"\nSloupec: {col}")
            lines.append(f"  {e}")
            continue

        lines.append(f"\nSloupec: {col}")
        lines.append(f"  Počet pozorování n = {res['n']}")
        lines.append(f"  Odhad střední hodnoty (průměr): {res['mu_hat']:.4f}")
        lines.append(f"  Odhad rozptylu: {res['sigma2_hat']:.4f}")
        lines.append(f"  Odhad směrodatné odchylky: {res['sigma_hat']:.4f}")

        m_low, m_high = res["CI_mean"]
        v_low, v_high = res["CI_variance"]
        s_low, s_high = res["CI_std"]

        lines.append(f"  {int(confidence*100)}% CI pro střední hodnotu: "
                     f"({m_low:.4f}, {m_high:.4f})")
        lines.append(f"  {int(confidence*100)}% CI pro rozptyl: "
                     f"({v_low:.4f}, {v_high:.4f})")
        lines.append(f"  {int(confidence*100)}% CI pro směrodatnou odchylku: "
                     f"({s_low:.4f}, {s_high:.4f})")

    # vytisknout do konzole
    text = "\n".join(lines)
    print("\n" + text + "\n")

    # zapsat do souboru
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Textový výstup uložen do: {out_path}")
