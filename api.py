"""
PHASE 6 - Backend (API Flask)
==============================
API REST complémentaire au dashboard Streamlit, exposant les données et le
modèle IA pour une éventuelle intégration à d'autres clients (frontend web,
application mobile, etc.).

Lancement :
    pip install -r requirements.txt
    python api.py
    -> http://127.0.0.1:5000

Endpoints :
    GET  /api/permis                 liste paginée des permis (filtres query params)
    GET  /api/permis/<num_permis>    détail d'un permis
    GET  /api/stats                  statistiques agrégées (KPI)
    POST /api/predict                prédiction IA (risque d'annulation)
"""
import pickle
import sqlite3

import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

DB_PATH = "data/mining_data.db"
with open("model/model.pkl", "rb") as f:
    MODEL_BUNDLE = pickle.load(f)


def get_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM permis_miniers", conn)
    conn.close()
    return df


@app.route("/api/permis", methods=["GET"])
def list_permis():
    df = get_df()

    type_ = request.args.get("type")
    classification = request.args.get("classification")
    region = request.args.get("region")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))

    if type_:
        df = df[df["type"] == type_]
    if classification:
        df = df[df["classification"] == classification]
    if region:
        df = df[df["region_principale"].str.contains(region, case=False, na=False)]

    total = len(df)
    start = (page - 1) * per_page
    df_page = df.iloc[start:start + per_page]

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "results": df_page.to_dict(orient="records"),
    })


@app.route("/api/permis/<int:num_permis>", methods=["GET"])
def get_permis(num_permis):
    df = get_df()
    row = df[df["num_permis"] == num_permis]
    if row.empty:
        return jsonify({"error": "Permis introuvable"}), 404
    return jsonify(row.iloc[0].to_dict())


@app.route("/api/stats", methods=["GET"])
def stats():
    df = get_df()
    return jsonify({
        "total_permis": len(df),
        "titulaires_distincts": int(df["titulaire"].nunique()),
        "superficie_totale_carres": int(df["nombre_carres"].sum()),
        "taux_risque_annulation": round(float(df["a_risque_annulation"].mean()) * 100, 2),
        "repartition_type": df["type"].value_counts().to_dict(),
        "repartition_classification": df["classification"].value_counts().to_dict(),
        "top_substances": df["substance_principale"].value_counts().nlargest(10).to_dict(),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    model = MODEL_BUNDLE["model"]
    encoders = MODEL_BUNDLE["encoders"]
    features = MODEL_BUNDLE["features"]

    def encode_safe(col, value):
        enc = encoders[col]
        return enc.transform([value])[0] if value in enc.classes_ else 0

    try:
        row = pd.DataFrame([{
            "type": encode_safe("type", payload["type"]),
            "nombre_carres": payload["nombre_carres"],
            "nb_substances": payload["nb_substances"],
            "duree_validite_annees": payload["duree_validite_annees"],
            "annee_octroi": payload["annee_octroi"],
            "fa_recent": payload["fa_recent"],
            "region_principale": encode_safe("region_principale", payload["region_principale"]),
            "substance_principale": encode_safe("substance_principale", payload["substance_principale"]),
        }])[features]
    except KeyError as e:
        return jsonify({"error": f"Champ manquant: {e}"}), 400

    proba = model.predict_proba(row)[0][1]
    pred = int(model.predict(row)[0])

    return jsonify({
        "prediction": "a_risque" if pred == 1 else "regulier",
        "probabilite_risque": round(float(proba), 4),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
