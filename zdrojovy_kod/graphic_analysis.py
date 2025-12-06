import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats  # for QQ-plot


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_scatter(df, title, output_path):
    if df.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(df["A"], df["B"], alpha=0.6)
    plt.xlabel("A")
    plt.ylabel("B")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_time_series(df, title, output_path):
    if df.empty:
        return
    df_sorted = df.sort_values("datetime")
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted["datetime"], df_sorted["A"], label="A")
    plt.plot(df_sorted["datetime"], df_sorted["B"], label="B")
    plt.xlabel("Čas")
    plt.ylabel("Hodnota")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_boxplot(data, labels, title, ylabel, output_path):
    if len(data) == 0:
        return
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_hist_kde(series, title, output_path):
    s = series.dropna().astype(float)
    if s.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.hist(s, bins=30, density=True, alpha=0.6)
    try:
        s.plot(kind="kde")
    except Exception:
        pass
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Hustota")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_qq(series, title, output_path):
    s = series.dropna().astype(float)
    if s.empty:
        return
    plt.figure(figsize=(6, 6))
    stats.probplot(s, dist="norm", plot=plt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_corr_heatmap(df, title, output_path):
    if df.empty:
        return
    corr = df.corr()
    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_pairplot_grid(df, cols, title, output_path):
    df = df[cols].dropna()
    if df.empty:
        return
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # diagonal: histogram
                ax.hist(df[cols[i]], bins=30, alpha=0.7)
            else:
                ax.scatter(df[cols[j]], df[cols[i]], alpha=0.5, s=10)
            if i == n - 1:
                ax.set_xlabel(cols[j])
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(cols[i])
            else:
                ax.set_yticklabels([])
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_violin_by_group(df, value_col, group_col, title, output_path):
    df = df[[value_col, group_col]].dropna()
    if df.empty:
        return
    groups = sorted(df[group_col].unique())
    data = [df[df[group_col] == g][value_col].values for g in groups]
    plt.figure(figsize=(10, 6))
    parts = plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks(range(1, len(groups) + 1), groups)
    plt.title(title)
    plt.ylabel(value_col)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def detect_outliers_iqr(series):
    s = series.dropna().astype(float)
    if s.empty:
        return pd.Series([], dtype=bool)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def graphic_analysis(model):
    print("Provádí se grafické zpracování dat...")

    df = model.data
    if df is None:
        print("Chyba: model.data je None – nejprve načti data.")
        return

    # typy
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    for col in ["A", "B", "poměr", "odchylka", "pohlaví"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime", "A", "B"])

    # základní cesty
    base_dir = os.path.dirname(os.path.dirname(__file__))
    vystupy = ensure_dir(os.path.join(base_dir, "vystupy"))

    # ---------------------------------------------------------
    # 1) SCATTER + TIME: ALL / POHLAVI / VEK
    # ---------------------------------------------------------
    all_dir = ensure_dir(os.path.join(vystupy, "scatter_time", "ALL"))
    save_scatter(df, "Závislost A a B — Všechna data",
                 os.path.join(all_dir, "scatter_A_B.png"))
    save_time_series(df, "Časová osa A a B — Všechna data",
                     os.path.join(all_dir, "time_A_B.png"))

    sex_root = ensure_dir(os.path.join(vystupy, "scatter_time", "pohlavi"))
    for sex in sorted(df["pohlaví"].dropna().unique()):
        sub = df[df["pohlaví"] == sex]
        group_dir = ensure_dir(os.path.join(sex_root, str(int(sex))))
        save_scatter(sub, f"A vs B — Pohlaví {int(sex)}",
                     os.path.join(group_dir, "scatter_A_B.png"))
        save_time_series(sub, f"Časová osa — Pohlaví {int(sex)}",
                         os.path.join(group_dir, "time_A_B.png"))

    vek_root = ensure_dir(os.path.join(vystupy, "scatter_time", "vek"))
    for age_group in sorted(df["věk"].dropna().unique()):
        sub = df[df["věk"] == age_group]
        safe_name = age_group.replace("–", "-").replace(" ", "")
        group_dir = ensure_dir(os.path.join(vek_root, safe_name))
        save_scatter(sub, f"A vs B — Věk {age_group}",
                     os.path.join(group_dir, "scatter_A_B.png"))
        save_time_series(sub, f"Časová osa — Věk {age_group}",
                         os.path.join(group_dir, "time_A_B.png"))

    # ---------------------------------------------------------
    # 2) BOXPLOTS: poměr podle věku a podle pohlaví
    # ---------------------------------------------------------
    box_root = ensure_dir(os.path.join(vystupy, "boxplots"))

    # věk
    age_groups = sorted(df["věk"].dropna().unique())
    age_data = [df[df["věk"] == g]["poměr"].dropna().values for g in age_groups]
    save_boxplot(
        age_data,
        age_groups,
        "Boxplot poměru podle věkových kategorií",
        "poměr",
        os.path.join(box_root, "box_pomer_by_vek.png"),
    )

    # pohlaví
    sex_groups = sorted(df["pohlaví"].dropna().unique())
    sex_labels = [str(int(s)) for s in sex_groups]
    sex_data = [df[df["pohlaví"] == s]["poměr"].dropna().values for s in sex_groups]
    save_boxplot(
        sex_data,
        sex_labels,
        "Boxplot poměru podle pohlaví",
        "poměr",
        os.path.join(box_root, "box_pomer_by_pohlavi.png"),
    )

    # ---------------------------------------------------------
    # 3) HISTOGRAMY + KDE pro A, B, poměr
    # ---------------------------------------------------------
    hist_root = ensure_dir(os.path.join(vystupy, "histograms"))
    for col in ["A", "B", "poměr"]:
        if col in df.columns:
            save_hist_kde(
                df[col],
                f"Histogram + KDE pro {col}",
                os.path.join(hist_root, f"hist_{col}.png"),
            )

    # ---------------------------------------------------------
    # 4) QQ-PLOTS pro A, B, poměr
    # ---------------------------------------------------------
    qq_root = ensure_dir(os.path.join(vystupy, "qqplots"))
    for col in ["A", "B", "poměr"]:
        if col in df.columns:
            save_qq(
                df[col],
                f"QQ-plot pro {col}",
                os.path.join(qq_root, f"qq_{col}.png"),
            )

    # ---------------------------------------------------------
    # 5) KORELAČNÍ MATICE — děti vs dospělí
    # ---------------------------------------------------------
    corr_root = ensure_dir(os.path.join(vystupy, "correlation"))
    # děti: věk 0–20, dospělí: 20+
    # podle tvých kategorií: 0-10, 10-20 -> děti; 20-40, 40-60, 60-80 -> dospělí
    children_mask = df["věk"].isin(["0-10", "10-20"])
    adults_mask = df["věk"].isin(["20-40", "40-60", "60-80"])

    cols_corr = [c for c in ["A", "B", "poměr", "odchylka"] if c in df.columns]

    children = df.loc[children_mask, cols_corr]
    adults = df.loc[adults_mask, cols_corr]

    save_corr_heatmap(
        children,
        "Korelační matice — Děti (0–20)",
        os.path.join(corr_root, "corr_children.png"),
    )
    save_corr_heatmap(
        adults,
        "Korelační matice — Dospělí (20+)",
        os.path.join(corr_root, "corr_adults.png"),
    )

    # ---------------------------------------------------------
    # 6) PAIRPLOT-STYLE GRID (A, B, poměr, odchylka)
    # ---------------------------------------------------------
    pair_root = ensure_dir(os.path.join(vystupy, "pairplot"))
    cols_pair = [c for c in ["A", "B", "poměr", "odchylka"] if c in df.columns]
    if len(cols_pair) >= 2:
        save_pairplot_grid(
            df,
            cols_pair,
            "Párové grafy (A, B, poměr, odchylka)",
            os.path.join(pair_root, "pairplot_AB_pomer_odchylka.png"),
        )

    # ---------------------------------------------------------
    # 7) VIOLIN PLOTS — poměr by věk & by pohlaví
    # ---------------------------------------------------------
    violin_root = ensure_dir(os.path.join(vystupy, "violin"))
    if "poměr" in df.columns:
        save_violin_by_group(
            df,
            "poměr",
            "věk",
            "Violin plot poměru podle věku",
            os.path.join(violin_root, "violin_pomer_by_vek.png"),
        )
        save_violin_by_group(
            df,
            "poměr",
            "pohlaví",
            "Violin plot poměru podle pohlaví",
            os.path.join(violin_root, "violin_pomer_by_pohlavi.png"),
        )

    # ---------------------------------------------------------
    # 8) OUTLIERS — poměr by věk (textový výpis)
    # ---------------------------------------------------------
    out_root = ensure_dir(os.path.join(vystupy, "outliers"))
    if "poměr" in df.columns:
        lines = []
        for age_group in age_groups:
            sub = df[df["věk"] == age_group].copy()
            if sub.empty:
                continue
            mask_out = detect_outliers_iqr(sub["poměr"])
            outs = sub[mask_out]
            if not outs.empty:
                lines.append(f"Věk {age_group}: {len(outs)} odlehlých pozorování")
                for _, row in outs.iterrows():
                    lines.append(
                        f"  datetime={row['datetime']}, A={row['A']}, "
                        f"B={row['B']}, poměr={row['poměr']}, pohlaví={row['pohlaví']}"
                    )
            else:
                lines.append(f"Věk {age_group}: žádná odlehlá pozorování")

        out_path = os.path.join(out_root, "outliers_pomer_by_vek.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print("Grafická analýza dokončena. Výsledky jsou ve složce 'vystupy'.")
