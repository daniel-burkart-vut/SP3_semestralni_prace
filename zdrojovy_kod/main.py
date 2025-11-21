import os
from read_data import read_data
from graphic_analysis import graphic_analysis
from parametric_analysis import vypocet_odhadu_normalnich_parametru



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

    # 2 a 2b. uloha
    graphic_analysis(model)
    
    # Tady pak můžeš volat další analýzy, např. graphic_analysis.run(model)


if __name__ == "__main__":
    main()
