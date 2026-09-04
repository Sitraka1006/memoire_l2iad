# SITRAKA — Plateforme Analytique des Données Minières à Madagascar

Projet complet : nettoyage, EDA, dashboard Streamlit professionnel, API Flask,
modèle IA (prédiction du risque d'annulation des permis).

## Nouveautés (version complétée)

- **Design professionnel** : en-tête dégradé, typographie Inter, onglets soignés
- **Toutes les analyses EDA** intégrées au dashboard (substances, titulaires, temporalité, relations, synthèse 4 graphiques)
- **Carte interactive** avec coordonnées approximatives des localités majeures (Plotly Mapbox + Folium optionnel) + liens OpenStreetMap / Google Maps
- **Import multi-formats** : CSV, TSV, TXT, XLSX, XLS, XLSM, XLSB, ODS, JSON
- Export CSV et Excel des données filtrées

## Structure

```
project/
├── app.py                 # Dashboard Streamlit (complet)
├── api.py                 # API Flask
├── requirements.txt
├── README.md
├── data/
│   ├── mining_data.db
│   ├── permis_miniers_propre.csv
│   ├── permis_miniers_brut.csv
│   └── clean_data.py
├── model/
│   ├── train_model.py
│   ├── model.pkl
│   └── evaluation_report.json
├── notebooks/
│   └── Nettoyage et Analyse exploratoire.ipynb
├── figures/
├── uml/
└── memoire/
```

## Installation et lancement

```bash
cd project
pip install -r requirements.txt

# (optionnel) régénérer données / modèle
python data/clean_data.py
python model/train_model.py

# Dashboard
streamlit run app.py
# → http://localhost:8501

# API (optionnel)
python api.py
# → http://localhost:5000
```

## Fonctionnalités du dashboard

| Onglet | Contenu |
|--------|---------|
| Tableau de bord | KPI, classification, types, treemap localités, synthèse 4 graphiques |
| EDA générale | Stats descriptives, histogrammes, boxplots, scatter, corrélations |
| Substances | Top N, évolution temporelle des 5 principales, distribution |
| Titulaires | Top 20, superficie cumulée |
| Carte | Scatter mapbox + Folium, liens OSM/Google Maps |
| Temporalité | Courbe des octrois, curseur année, FA payé |
| Module IA | Métriques Random Forest, simulation de prédiction + jauge |
| Données | Table filtrable, export CSV/XLSX, dictionnaire |

## Résultats IA

- **Random Forest** : Accuracy 84.7 %, F1-score 0.729
- Variables clés : durée de validité, année d'octroi, régularité FA

## Limites

- Pas de GPS natifs → coordonnées approximatives pour les localités majeures
- ~38 % de dates d'octroi manquantes dans la source
- Pas de tonnage/production → module « rendement » non implémenté (remplacé par classification du risque)

## Stack

Pandas · SQLite · Streamlit · Plotly · Folium · Scikit-learn · Flask
