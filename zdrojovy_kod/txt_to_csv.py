import os
import csv

# Base dir = one level above this script (project root)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

input_path = os.path.join(BASE_DIR, "vstupni_data.txt")
output_path = os.path.join(BASE_DIR, "vstupni_data.csv")

# Column order for the CSV
fieldnames = ["datetime", "A", "B", "poměr", "odchylka", "pohlaví", "věk"]

def parse_line(line: str) -> dict:
    line = line.strip()
    if not line:
        return None

    parts = line.split(" | ")
    if len(parts) < 7:
        # malformed line – skip or handle differently if you want
        return None

    row = {}
    # first part is date + time together
    row["datetime"] = parts[0]

    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()

        # remove % sign from odchylka
        if key == "odchylka" and value.endswith("%"):
            value = value[:-1]

        row[key] = value

    return row

def main():
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8", newline="") as fout:

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for line in fin:
            row = parse_line(line)
            if row is not None:
                writer.writerow(row)

    print(f"Done. Wrote CSV to: {output_path}")

if __name__ == "__main__":
    main()
