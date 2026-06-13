"""
Parsing utility for the AME2020 atomic mass table (mass_1.mas20.txt).
"""
import pandas as pd


def parse_ame2020(filepath: str) -> pd.DataFrame:
    """Parse the AME2020 mass table from a fixed-width text file.

    Extrapolated values (marked with '#') and uncalculable entries ('*')
    are silently skipped.

    Args:
        filepath (str): Path to the AME2020 table file (mass_1.mas20.txt).

    Returns:
        pd.DataFrame with columns:
            A              — mass number
            Z              — atomic number
            BE_per_A       — binding energy per nucleon (keV)
            Error_per_A    — experimental uncertainty on BE/A (keV)
            Mass_Excess    — atomic mass excess (keV)
            ME_Error       — experimental uncertainty on mass excess (keV)
            Element        — chemical symbol
    """
    data = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    start_parsing = False
    for line in lines:
        if "0  1    1    0    1" in line:
            start_parsing = True
        if not start_parsing or len(line) < 80:
            continue

        try:
            A_str            = line[14:19]
            Z_str            = line[9:14]
            BE_per_A_str     = line[54:67]
            BE_per_A_err_str = line[68:78]
            ME_str           = line[29:43]
            ME_err_str       = line[43:55]
            element_symbol   = line[20:22].strip()

            # Skip extrapolated or missing entries
            if "#" in BE_per_A_str or "#" in BE_per_A_err_str or "*" in BE_per_A_str:
                continue
            if "#" in ME_str or "*" in ME_str:
                continue

            A            = int(A_str)
            Z            = int(Z_str)
            BE_per_A     = float(BE_per_A_str)
            BE_per_A_err = float(BE_per_A_err_str)
            ME           = float(ME_str)
            ME_err       = float(ME_err_str)

            data.append([A, Z, BE_per_A, BE_per_A_err, ME, ME_err, element_symbol])
        except ValueError:
            continue

    return pd.DataFrame(data, columns=[
        "A", "Z", "BE_per_A", "Error_per_A", "Mass_Excess", "ME_Error", "Element",
    ])
