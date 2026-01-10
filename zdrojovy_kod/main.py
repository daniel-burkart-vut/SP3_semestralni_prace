import os
from read_data import read_data
from graphic_analysis import graphic_analysis
from parametric_analysis import compute_normal_parameter_estimates
from asymptotic_median_ratio import asymptotic_median_ratio
from ks_median_test import ks_median_test_children_vs_adults
from quantile_regression import quantile_analysis
from test_shodnosti_medianu import Wilcoxon



class Model:
    def __init__(self) -> None:
        # Projektový root = o jeden adresář výše než tento soubor
        base_dir = os.path.dirname(os.path.dirname(__file__))

        # Tady natvrdo řekneme, že vstup je CSV v rootu projektu
        self.input_format = "structured_csv"
        self.input_file = os.path.join(base_dir, "vstupni_data.csv")

        # Případně sem můžeš přidat další „úkoly“
        self.tasks = ["none"]

        # Sem si uložíme načtená data (DataFrame)
        self.data = None


def main() -> None:
    model = Model()

    # Načti data z CSV
    model.data = read_data(model)

    # Rychlá kontrola – vypíše prvních pár řádků
    print(model.data.head())

    # Ulohy 2 .a 2b.: graficka analiza 
    graphic_analysis(model)

    # Uloha 4.: Odhady parametrů normálního rozdělení + intervaly spolehlivosti
    compute_normal_parameter_estimates(model)

    # Uloha 7.: test shodnosti medianu pro deti a dospele - uzitim Kolmogorov-Smirnovova testu				
    ks_median_test_children_vs_adults(model, column="poměr")

    # Uloha 5: Asymptotické rozdělení mediánu poměru X/Y
    asymptotic_median_ratio(model, column_x="A", column_y="B")

    # Uloha 6: test shodnosti medianu X/Y pro deti a dospele 
    Wilcoxon(model, column="poměr", alpha=0.05)

    # Uloha 10: kvantilova regrese
    quantile_analysis(model, dependent="poměr", covariates=["pohlaví", "věk"])
    
    # Tady pak můžeš volat další analýzy, např. graphic_analysis.run(model)


if __name__ == "__main__":
    main()
