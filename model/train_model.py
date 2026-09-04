"""
PHASE 7 - Partie Intelligence Artificielle
Option 2 : Classification des permis miniers à risque d'annulation.

Cible : a_risque_annulation (1 = permis en cours/attente d'annulation, 0 = sinon)
Algorithmes comparés : Regression logistique (baseline), Decision Tree, Random Forest
"""
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

df = pd.read_csv("data/permis_miniers_propre.csv")

# --- Sélection des features ---
features = ["type", "nombre_carres", "nb_substances", "duree_validite_annees",
            "annee_octroi", "fa_recent", "region_principale", "substance_principale"]
target = "a_risque_annulation"

data = df[features + [target]].copy()
data["duree_validite_annees"] = data["duree_validite_annees"].fillna(data["duree_validite_annees"].median())
data["annee_octroi"] = data["annee_octroi"].fillna(data["annee_octroi"].median())

# Réduire cardinalité des régions rares -> "Autre"
top_regions = data["region_principale"].value_counts().nlargest(25).index
data["region_principale"] = data["region_principale"].where(data["region_principale"].isin(top_regions), "Autre")
top_subst = data["substance_principale"].value_counts().nlargest(20).index
data["substance_principale"] = data["substance_principale"].where(data["substance_principale"].isin(top_subst), "Autre")

encoders = {}
for col in ["type", "region_principale", "substance_principale"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Regression Logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42),
}

results = {}
best_model, best_name, best_f1 = None, None, -1

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    results[name] = metrics
    if metrics["f1_score"] > best_f1:
        best_f1 = metrics["f1_score"]
        best_model = model
        best_name = name

print("=== Comparaison des modèles ===")
for name, m in results.items():
    print(f"\n{name}:")
    for k, v in m.items():
        print(f"  {k}: {v}")

print(f"\n>>> Meilleur modèle retenu : {best_name} (F1 = {best_f1})")

# Feature importance si dispo
if hasattr(best_model, "feature_importances_"):
    importances = sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1])
    print("\nImportance des variables :")
    for f, imp in importances:
        print(f"  {f}: {round(imp, 4)}")

# Sauvegarde
with open("model/model.pkl", "wb") as f:
    pickle.dump({"model": best_model, "encoders": encoders, "features": features, "name": best_name}, f)

with open("model/evaluation_report.json", "w", encoding="utf-8") as f:
    json.dump({"results": results, "best_model": best_name, "best_f1": best_f1}, f, ensure_ascii=False, indent=2)

print("\nModèle sauvegardé dans model/model.pkl")
print("Rapport sauvegardé dans model/evaluation_report.json")
