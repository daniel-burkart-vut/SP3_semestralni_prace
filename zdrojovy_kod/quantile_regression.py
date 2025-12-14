import os
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def quantile_analysis(model, dependent="poměr", covariates = None, quantiles=None, bootstrap=1000):
    """
    Kvantilová regrese pro vybranou závislou proměnnou.
    Používá pouze prediktory 'věk' a 'pohlaví'.
    Výsledky ukládá do .txt a grafy do .png ve složce 'vystupy/QR'.
    """

    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    if covariates is None:
        covariates = ["věk", "pohlaví"]

    # Převod kategoriálních proměnných na typ 'category'
    for cat_var in covariates:
        if cat_var in model.data.columns:
            model.data[cat_var] = model.data[cat_var].astype("category")

    formula = f"{dependent} ~ " + " + ".join(covariates)
    print(f"Formule kvantilové regrese: {formula}\n")

    # ------------------------------------------------------------
    # Připravení složky pro výstupy: vystupy/QR
    # ------------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(__file__))  # kořen projektu
    out_dir = os.path.join(base_dir, "vystupy", "QR")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Uložení textového výstupu
    # ------------------------------------------------------------
    txt_path = os.path.join(out_dir, f"kvantilova_regrese_{dependent}.txt")
    with open(txt_path, "w", encoding="utf-8") as f_out:
        for q in quantiles:
            f_out.write(f"--- Kvantil {q} ---\n")
            mod = smf.quantreg(formula, model.data)
            res = mod.fit(q=q, cov_type='boot', bootstrap=bootstrap)
            f_out.write(res.summary().as_text() + "\n\n")

    print(f"Textový výstup kvantilové regrese uložen do: {txt_path}")

    # ------------------------------------------------------------
    # Vykreslení grafů
    # ------------------------------------------------------------
    plot_quantiles_for_covariates(model, covariates=covariates, dependent=dependent, quantiles=quantiles, out_dir=out_dir)


def plot_quantiles_for_covariates(model, covariates, dependent="poměr", quantiles=None, out_dir=None):
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]

    data = model.data
    colors = ['red', 'green', 'blue', 'purple', 'orange']

    for cov in covariates:
        plt.figure(figsize=(8, 6))
        phi = 1.618
        plt.axhline(y=phi, color='orange', linestyle='--', label='Golden ratio φ')

        # Kategoriální proměnná
        if pd.api.types.is_categorical_dtype(data[cov]):
            categories = data[cov].cat.categories
            for i, q in enumerate(quantiles):
                formula = f"{dependent} ~ {cov}"
                mod = smf.quantreg(formula, data)
                res = mod.fit(q=q)
                preds = [res.predict(pd.DataFrame({cov: [cat]}))[0] for cat in categories]
                plt.plot(categories, preds, marker='o', color=colors[i % len(colors)], label=f'Kvantil {q}')
            plt.xlabel(cov)
            plt.ylabel(dependent)
            plt.title(f'Predikce kvantilové regrese pro {cov}')
        # Numerická proměnná
        else:
            x = np.linspace(data[cov].min(), data[cov].max(), 100)
            plt.scatter(data[cov], data[dependent], alpha=0.5, label="Data")
            for i, q in enumerate(quantiles):
                formula = f"{dependent} ~ {cov}"
                mod = smf.quantreg(formula, data)
                res = mod.fit(q=q)
                y = res.params['Intercept'] + res.params[cov] * x
                plt.plot(x, y, color=colors[i % len(colors)], label=f'Kvantil {q}')
            plt.xlabel(cov)
            plt.ylabel(dependent)
            plt.title(f'Kvantilová regrese: {dependent} vs {cov}')

        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if out_dir:
            plot_path = os.path.join(out_dir, f"kvantilova_regrese_{dependent}_vs_{cov}.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"Graf uložen: {plot_path}")
        else:
            plt.show()
