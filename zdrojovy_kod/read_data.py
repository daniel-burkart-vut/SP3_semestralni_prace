import pandas as pd
import numpy as np


def read_data(model):
    print(f"Načítají se data ze souboru: {model.input_file} ve formátu: {model.input_format}")

    df = None

    if model.input_format == "structured_csv":
        # Načtení CSV
        df = pd.read_csv(model.input_file)

        # Pokus o převod sloupce datetime na skutečný datetime
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        # Číselné sloupce – zkus převést na čísla
        numeric_cols = ["A", "B", "poměr", "odchylka", "pohlaví"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Pokud by náhodou v CSV bylo "4.53%" místo "4.53"
        if "odchylka" in df.columns and df["odchylka"].dtype == object:
            if df["odchylka"].astype(str).str.endswith("%").any():
                df["odchylka"] = (
                    df["odchylka"]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                )
                df["odchylka"] = pd.to_numeric(df["odchylka"], errors="coerce")

        return df

    elif model.input_format == "txt":
        # Nechceme TXT vůbec používat – vstup musí být CSV
        raise NotImplementedError("TXT formát už nepoužívej, nejdřív si ho převeď na CSV.")

    return df
