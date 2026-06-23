import pandas as pd
import os

def load_and_clean_data(filepath="data/raw/exoplanetdata.csv"):
    """
    Loads and cleans exoplanet data from NASA Exoplanet Archive.
    Generates deterministic habitability labels based on astrophysical criteria.
    """
    if not os.path.exists(filepath):
        # Handle execution from inside directories like src/
        alternative_path = os.path.join("..", filepath)
        if os.path.exists(alternative_path):
            filepath = alternative_path
        else:
            raise FileNotFoundError(f"Exoplanet dataset not found at {filepath} or {alternative_path}")

    # Header starts from row 89 (index 88)
    df = pd.read_csv(filepath, header=88)

    required_columns = [
        "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt", "pl_insol",
        "pl_orbeccen", "st_teff", "st_rad", "st_mass", "st_met", "sy_dist"
    ]
    df = df.dropna(subset=required_columns)

    # Classify habitability based on planetary radius, equilibrium temp, and insolation flux
    df['is_habitable'] = (
        (df['pl_rade'] >= 0.5) & (df['pl_rade'] <= 2.5) &
        (df['pl_eqt'] >= 160) & (df['pl_eqt'] <= 330) &
        (df['pl_insol'] >= 0.3) & (df['pl_insol'] <= 1.8)
    ).astype(int)

    return df
