"""
PHASE 4 - Préparation des données
Nettoyage et transformation du dataset des permis miniers de Madagascar.
"""
import pandas as pd
import numpy as np
import re
import sqlite3

RAW_PATH = "data/permis_miniers_brut.csv"
CLEAN_CSV = "data/permis_miniers_propre.csv"
DB_PATH = "data/mining_data.db"

def load_raw(path=RAW_PATH):
    df = pd.read_csv(path, parse_dates=["Date d'octroi initial", "Date fin de validité initiale"])
    return df

def clean(df):
    df = df.copy()

    # 1. Suppression des doublons
    n0 = len(df)
    df = df.drop_duplicates(subset=["N°permis"])
    n1 = len(df)

    # 2. Normalisation des noms de colonnes
    df.columns = [
        "num_permis", "type", "titulaire", "classification", "nombre_carres",
        "substances", "date_octroi", "date_fin_validite", "dernier_fa_paye",
        "en_cours", "localisation"
    ]

    # 3. Nettoyage des textes
    df["titulaire"] = df["titulaire"].str.strip()
    df["substances"] = df["substances"].fillna("Inconnu").str.strip()
    df["substances"] = df["substances"].str.lstrip("-")
    df["classification"] = df["classification"].str.strip()

    # 4. Gestion des valeurs manquantes
    df["localisation"] = df["localisation"].fillna("Non renseigné")
    df["en_cours"] = df["en_cours"].fillna("Aucune procédure en cours")

    # dates manquantes -> on garde NaT mais on le signale avec un flag
    df["date_octroi_manquante"] = df["date_octroi"].isna()
    df["date_fin_manquante"] = df["date_fin_validite"].isna()

    # 5. Variables dérivées (feature engineering)
    df["nb_substances"] = df["substances"].apply(
        lambda s: len([x for x in re.split(r"[-,]", s) if x.strip()]) if s != "Inconnu" else 0
    )
    df["substance_principale"] = df["substances"].apply(lambda s: s.split("-")[0].strip() if s != "Inconnu" else "Inconnu")

    df["duree_validite_annees"] = (df["date_fin_validite"] - df["date_octroi"]).dt.days / 365.25
    df["duree_validite_annees"] = df["duree_validite_annees"].round(1)

    df["annee_octroi"] = df["date_octroi"].dt.year

    # Région principale = premier toponyme cité dans la localisation
    def premiere_region(loc):
        if not isinstance(loc, str) or loc == "Non renseigné":
            return "Non renseigné"
        premier = loc.split("/")[0]
        premier = re.sub(r"\s*\(\d+\)", "", premier).strip()
        return premier if premier else "Non renseigné"
    df["region_principale"] = df["localisation"].apply(premiere_region)

    # Cible IA : permis à risque (annulation en cours ou en attente de décision d'annulation)
    df["a_risque_annulation"] = df["classification"].str.contains("ANNULATION", na=False).astype(int)

    # 6. Encodage simple du statut "en règle" (paiement récent, ex: FA payé >= 2015)
    df["fa_recent"] = (df["dernier_fa_paye"] >= 2015).astype(int)

    # 7. Suppression des lignes totalement inexploitables (num_permis nul)
    df = df.dropna(subset=["num_permis"])

    rapport = {
        "lignes_initiales": n0,
        "doublons_supprimes": n0 - n1,
        "lignes_finales": len(df),
        "valeurs_manquantes_substances": int(df["substances"].eq("Inconnu").sum()),
        "valeurs_manquantes_dates_octroi": int(df["date_octroi_manquante"].sum()),
        "valeurs_manquantes_dates_fin": int(df["date_fin_manquante"].sum()),
    }
    return df, rapport

def save(df, csv_path=CLEAN_CSV, db_path=DB_PATH):
    df.to_csv(csv_path, index=False)
    conn = sqlite3.connect(db_path)
    df.to_sql("permis_miniers", conn, if_exists="replace", index=False)
    conn.close()

if __name__ == "__main__":
    df_raw = load_raw()
    df_clean, rapport = clean(df_raw)
    save(df_clean)
    print("=== Rapport de nettoyage ===")
    for k, v in rapport.items():
        print(f"{k}: {v}")
    print(f"\nDataset propre : {df_clean.shape[0]} lignes, {df_clean.shape[1]} colonnes")
    print(f"Sauvegardé dans {CLEAN_CSV} et {DB_PATH}")
