"""
OreMetrics - Plateforme Analytique des Données Minières à Madagascar
====================================================================
Dashboard Streamlit professionnel : KPI, EDA complète, carte interactive,
module IA (prédiction du risque d'annulation), import multi-formats.

Lancement :
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import pickle
import re
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Folium pour la carte (optionnel)
try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

st.set_page_config(
    page_title="OreMetrics | Données Minières Madagascar",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DB = Path("data/mining_data.db")
MODEL_PATH = Path("model/model.pkl")
EVAL_PATH = Path("model/evaluation_report.json")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f4c5c 0%, #1a7a6d 50%, #2d9f8f 100%);
        padding: 1.4rem 1.8rem; border-radius: 12px; margin-bottom: 1.2rem;
        color: white; box-shadow: 0 4px 14px rgba(15, 76, 92, 0.25);
    }
    .main-header h1 { margin: 0; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; }
    .main-header p { margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 0.95rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 10px 18px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #0f4c5c; color: white; }
    div[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; }
    .info-box {
        background: #e8f5f3; border-left: 4px solid #1a7a6d;
        padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; margin: 0.8rem 0; font-size: 0.92rem;
    }
    footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# Coordonnées approximatives (OpenStreetMap / géographie publique)
LOCALITY_COORDS = {
    "Anjoma Ramartina": (-19.65, 45.55), "Antsiafabositra": (-18.15, 47.05),
    "Benenitra": (-23.45, 45.15), "Ankilimary": (-22.35, 44.85),
    "Ambatofinandrahana": (-20.55, 46.82), "Antanimbary": (-18.92, 46.45),
    "Andohan' Ilakaka": (-22.68, 45.22), "Ilakaka": (-22.68, 45.22),
    "Ambatolahy": (-20.35, 45.55), "Fenoarivo": (-17.38, 49.40),
    "Ampondra": (-13.45, 49.95), "Ambolotarakely": (-18.55, 47.35),
    "Bemahatazana": (-19.85, 46.15), "Andriamena": (-17.58, 47.18),
    "Ambohinihaonana": (-21.45, 47.55), "Ianapera": (-23.55, 44.95),
    "Belitsaka": (-17.95, 48.15), "Ambalarondra": (-18.95, 48.75),
    "Beraketa": (-24.15, 45.55), "Fitampito": (-21.15, 46.35),
    "Ambohipaky": (-19.25, 46.85), "Morafeno": (-18.85, 48.25),
    "Fotadrevo": (-24.55, 44.85), "Andranomiely": (-18.65, 47.25),
    "Mahazoma": (-17.85, 46.95), "Ambolobozo": (-14.55, 47.95),
    "Maromby": (-24.35, 46.55), "Manantenina": (-24.25, 47.35),
    "Ambodilazana": (-17.75, 49.25), "Antsalova": (-18.65, 44.65),
    "Dabolava": (-19.65, 45.45), "Mandrosonoro": (-20.55, 46.25),
    "Ambodinonoka": (-18.45, 48.85), "Iakora": (-23.15, 46.15),
    "Ampandroantraka": (-18.25, 48.55), "Ambahita": (-24.35, 44.65),
    "Androrangavola": (-21.35, 47.85), "Marohazo": (-19.15, 46.55),
    "Andilamena": (-17.05, 48.55), "Andranobolaha": (-18.95, 48.85),
    "Manakana": (-17.55, 47.25), "Amboditavolo": (-18.85, 48.95),
    "Sahamatevina": (-18.75, 48.65), "Anjahamana": (-18.95, 48.75),
    "Kianjavato": (-21.35, 47.85), "Ibity": (-20.05, 47.05),
    "Sakaraha": (-22.95, 44.55), "Miary Taheza": (-23.35, 44.35),
    "Tranomaro": (-24.55, 46.65), "Ampasimpotsy Gara": (-18.95, 48.35),
    "Andasibe": (-18.95, 48.45), "Morarano Gara": (-18.85, 48.25),
    "Betsiaka": (-13.15, 49.55), "Maevatanana": (-16.95, 46.85),
    "Tolanaro": (-25.03, 46.98), "Farafangana": (-22.82, 47.83),
    "Ambovombe": (-25.18, 46.08), "Vatomandry": (-19.35, 48.95),
    "Mahanoro": (-19.90, 48.80), "Nosy Varika": (-20.58, 48.53),
    "Soanierana": (-16.85, 49.55), "Mahatalaky": (-24.85, 47.05),
    "Benonoky": (-24.55, 44.75), "Brieville": (-17.65, 47.25),
    "Ranopiso": (-25.05, 46.65), "Fetraomby": (-18.95, 48.55),
    "Antsampanana": (-18.85, 49.05), "Andevoranto": (-18.95, 49.05),
    "Ampasimadinika Manambolo": (-18.95, 48.85), "Ambinaninony": (-18.95, 48.75),
    "Sahanivotry": (-19.85, 46.55), "Tsivory": (-24.05, 46.35),
    "Zazafotsy": (-22.15, 46.35), "Esira": (-24.25, 46.75),
    "Fenoarivobe": (-18.45, 46.55), "Tsiroanomandidy": (-18.75, 46.05),
    "Antananarivo": (-18.88, 47.51), "Toamasina": (-18.15, 49.40),
    "Mahajanga": (-15.72, 46.32), "Toliara": (-23.35, 43.67),
    "Antsiranana": (-12.28, 49.29), "Fianarantsoa": (-21.45, 47.09),
}

@st.cache_data
def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DATA_DB)
    df = pd.read_sql("SELECT * FROM permis_miniers", conn)
    conn.close()
    df["date_octroi"] = pd.to_datetime(df["date_octroi"], errors="coerce")
    df["date_fin_validite"] = pd.to_datetime(df["date_fin_validite"], errors="coerce")
    return df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_eval_report():
    if EVAL_PATH.exists():
        with open(EVAL_PATH) as f:
            return json.load(f)
    return None

def parse_uploaded_file(uploaded) -> pd.DataFrame | None:
    """Support CSV, TSV, XLSX, XLS, ODS, JSON."""
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        if name.endswith(".tsv") or name.endswith(".txt"):
            return pd.read_csv(uploaded, sep="\t")
        if name.endswith((".xlsx", ".xls", ".xlsm", ".xlsb")):
            return pd.read_excel(uploaded)
        if name.endswith(".ods"):
            return pd.read_excel(uploaded, engine="odf")
        if name.endswith(".json"):
            return pd.read_json(uploaded)
        try:
            return pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            return pd.read_excel(uploaded)
    except Exception as e:
        st.sidebar.error(f"Erreur d'import : {e}")
        return None

def extract_substances_list(series: pd.Series) -> list:
    all_subs = []
    for s in series.dropna():
        parts = re.split(r"[-,;/|]", str(s))
        for p in parts:
            p = p.strip()
            if p and p.lower() not in ("inconnu", "nan", ""):
                all_subs.append(p)
    return all_subs

df = load_data()
model_bundle = load_model()
eval_report = load_eval_report()

# Sidebar
st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size:2rem;">⛏️</div>
        <div style="font-size:1.35rem; font-weight:700; color:#0f4c5c;">OreMetrics</div>
        <div style="font-size:0.8rem; color:#64748b;">Données minières · Madagascar</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("📁 Importer des données")
uploaded = st.sidebar.file_uploader(
    "Formats : CSV, TSV, XLSX, XLS, XLSM, ODS, JSON",
    type=["csv", "tsv", "txt", "xlsx", "xls", "xlsm", "xlsb", "ods", "json"],
    help="Aperçu uniquement — non fusionné automatiquement à la base.",
)
if uploaded is not None:
    new_df = parse_uploaded_file(uploaded)
    if new_df is not None:
        st.sidebar.success(f"✅ {len(new_df):,} lignes · {len(new_df.columns)} colonnes")
        with st.sidebar.expander("Aperçu des données importées"):
            st.dataframe(new_df.head(8), use_container_width=True)

st.sidebar.header("🔎 Filtres")
types_sel = st.sidebar.multiselect("Type de permis", sorted(df["type"].dropna().unique()), default=None)
classif_sel = st.sidebar.multiselect("Classification", sorted(df["classification"].dropna().unique()), default=None)
annees = df["annee_octroi"].dropna()
if len(annees):
    an_min, an_max = int(annees.min()), int(annees.max())
    annee_range = st.sidebar.slider("Année d'octroi", an_min, an_max, (an_min, an_max))
else:
    annee_range = None
subst_options = sorted(df["substance_principale"].dropna().unique())
subst_sel = st.sidebar.multiselect("Substance principale", subst_options[:80], default=None)

fdf = df.copy()
if types_sel:
    fdf = fdf[fdf["type"].isin(types_sel)]
if classif_sel:
    fdf = fdf[fdf["classification"].isin(classif_sel)]
if annee_range:
    fdf = fdf[((fdf["annee_octroi"] >= annee_range[0]) & (fdf["annee_octroi"] <= annee_range[1])) | fdf["annee_octroi"].isna()]
if subst_sel:
    fdf = fdf[fdf["substance_principale"].isin(subst_sel)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Permis affichés : **{len(fdf):,}** / {len(df):,}")

st.markdown(
    """
    <div class="main-header">
        <h1>OreMetrics — Plateforme Analytique Minière</h1>
        <p>Analyse exploratoire, cartographie et intelligence artificielle sur les permis miniers de Madagascar</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_kpi, tab_eda, tab_sub, tab_tit, tab_map, tab_temp, tab_ia, tab_data = st.tabs([
    "📊 Tableau de bord", "📈 EDA générale", "💎 Substances", "🏢 Titulaires",
    "🗺️ Carte", "📅 Temporalité", "🤖 Module IA", "🗂️ Données",
])

# TAB KPI
with tab_kpi:
    st.subheader("Indicateurs clés")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Permis (filtrés)", f"{len(fdf):,}".replace(",", " "))
    c2.metric("Titulaires", f"{fdf['titulaire'].nunique():,}".replace(",", " "))
    c3.metric("Carrés (superficie)", f"{int(fdf['nombre_carres'].sum()):,}".replace(",", " "))
    taux_risque = fdf["a_risque_annulation"].mean() * 100 if len(fdf) else 0
    c4.metric("À risque d'annulation", f"{taux_risque:.1f} %")
    c5.metric("Substances uniques", f"{fdf['substance_principale'].nunique():,}".replace(",", " "))

    col_a, col_b = st.columns(2)
    with col_a:
        classif_counts = fdf["classification"].value_counts().reset_index()
        classif_counts.columns = ["classification", "nombre"]
        fig = px.bar(classif_counts, x="nombre", y="classification", orientation="h",
                     title="Répartition par classification", color="nombre",
                     color_continuous_scale="Tealgrn", text="nombre")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        type_counts = fdf["type"].value_counts().reset_index()
        type_counts.columns = ["type", "nombre"]
        fig2 = px.pie(type_counts, names="type", values="nombre",
                      title="Répartition par type (R / PRE / E)", hole=0.45,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 15 localités minières")
    top_loc = fdf["region_principale"].value_counts().nlargest(15).reset_index()
    top_loc.columns = ["localite", "nombre_permis"]
    fig_tm = px.treemap(top_loc, path=["localite"], values="nombre_permis",
                        title="Concentration géographique des permis",
                        color="nombre_permis", color_continuous_scale="YlOrRd")
    fig_tm.update_layout(height=420)
    st.plotly_chart(fig_tm, use_container_width=True)

    st.subheader("Synthèse interactive")
    fig19 = make_subplots(rows=2, cols=2,
        subplot_titles=("Type de permis", "Classification (top 5)", "Top 10 substances", "Évolution des octrois"),
        specs=[[{"type": "pie"}, {"type": "bar"}], [{"type": "bar"}, {"type": "scatter"}]])
    type_pie = fdf["type"].value_counts()
    fig19.add_trace(go.Pie(labels=type_pie.index, values=type_pie.values, hole=0.35, showlegend=False), row=1, col=1)
    class_bar = fdf["classification"].value_counts().head(5)
    fig19.add_trace(go.Bar(x=class_bar.index, y=class_bar.values, marker_color="#2d9f8f", showlegend=False), row=1, col=2)
    sub_counter = Counter(extract_substances_list(fdf["substances"]))
    top10 = pd.DataFrame(sub_counter.most_common(10), columns=["Substance", "Fréquence"])
    if len(top10):
        fig19.add_trace(go.Bar(x=top10["Fréquence"], y=top10["Substance"], orientation="h",
                               marker_color="#0f4c5c", showlegend=False), row=2, col=1)
    year_counts = fdf.dropna(subset=["annee_octroi"]).groupby("annee_octroi").size().reset_index(name="Nombre")
    if len(year_counts):
        fig19.add_trace(go.Scatter(x=year_counts["annee_octroi"], y=year_counts["Nombre"],
                                   mode="lines+markers", marker_color="#1a7a6d", showlegend=False), row=2, col=2)
    fig19.update_layout(height=720, title_text="Tableau de bord synthétique")
    st.plotly_chart(fig19, use_container_width=True)

# TAB EDA
with tab_eda:
    st.subheader("Statistiques descriptives")
    num_cols = ["nombre_carres", "nb_substances", "duree_validite_annees", "annee_octroi"]
    st.dataframe(fdf[num_cols].describe().round(2), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution du nombre de carrés")
        fig5 = px.histogram(fdf, x="nombre_carres", nbins=50, title="Superficie des permis (carrés)",
                            color_discrete_sequence=["#1a7a6d"])
        fig5.update_layout(height=360)
        st.plotly_chart(fig5, use_container_width=True)
    with col2:
        st.subheader("Durée de validité (années)")
        fig_d = px.box(fdf.dropna(subset=["duree_validite_annees"]), y="duree_validite_annees", color="type",
                       title="Durée de validité par type de permis",
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig_d.update_layout(height=360)
        st.plotly_chart(fig_d, use_container_width=True)

    st.subheader("Relations entre variables")
    col3, col4 = st.columns(2)
    with col3:
        fig_sc = px.scatter(fdf.dropna(subset=["nombre_carres", "nb_substances"]),
                            x="nombre_carres", y="nb_substances", color="type", size="nombre_carres",
                            hover_data=["num_permis", "titulaire"], title="Carrés vs nombre de substances", opacity=0.6)
        fig_sc.update_layout(height=400)
        st.plotly_chart(fig_sc, use_container_width=True)
    with col4:
        risk_by_type = fdf.groupby("type")["a_risque_annulation"].mean().reset_index().rename(
            columns={"a_risque_annulation": "taux_risque"})
        risk_by_type["taux_risque"] *= 100
        fig_r = px.bar(risk_by_type, x="type", y="taux_risque", title="Taux de risque d'annulation par type (%)",
                       color="taux_risque", color_continuous_scale="Reds", text_auto=".1f")
        fig_r.update_layout(height=400)
        st.plotly_chart(fig_r, use_container_width=True)

    st.subheader("Matrice de corrélation (variables numériques)")
    corr_cols = [c for c in num_cols + ["a_risque_annulation", "fa_recent"] if c in fdf.columns]
    corr = fdf[corr_cols].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Corrélations", aspect="auto")
    fig_corr.update_layout(height=420)
    st.plotly_chart(fig_corr, use_container_width=True)

# TAB Substances
with tab_sub:
    st.subheader("Analyse des substances minières")
    all_subs = extract_substances_list(fdf["substances"])
    sub_counter = Counter(all_subs)
    top_n = st.slider("Nombre de substances à afficher", 10, 40, 20)
    top_df = pd.DataFrame(sub_counter.most_common(top_n), columns=["Substance", "Fréquence"])
    fig4 = px.bar(top_df, x="Fréquence", y="Substance", orientation="h",
                  title=f"Top {top_n} substances mentionnées", color="Fréquence",
                  color_continuous_scale="Viridis", text="Fréquence")
    fig4.update_layout(yaxis={"categoryorder": "total ascending"}, height=520)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Évolution des 5 principales substances par année")
    top5 = [s for s, _ in sub_counter.most_common(5)]
    if top5:
        df_tmp = fdf.copy()
        for sub in top5:
            df_tmp[sub] = df_tmp["substances"].apply(lambda x, s=sub: 1 if isinstance(x, str) and s in x else 0)
        pivot = df_tmp.groupby("annee_octroi")[top5].sum().reset_index()
        melt = pivot.melt(id_vars="annee_octroi", var_name="Substance", value_name="Nombre")
        melt = melt.dropna(subset=["annee_octroi"])
        fig20 = px.line(melt, x="annee_octroi", y="Nombre", color="Substance", markers=True,
                        title="Évolution des 5 substances principales",
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig20.update_layout(height=450)
        st.plotly_chart(fig20, use_container_width=True)

    st.subheader("Nombre de substances par permis")
    fig_nb = px.histogram(fdf, x="nb_substances", title="Distribution du nombre de substances déclarées",
                          color_discrete_sequence=["#0f4c5c"], nbins=15)
    st.plotly_chart(fig_nb, use_container_width=True)
    st.markdown(
        '<div class="info-box"><b>Interprétation</b> : L\'Or, le Béryl, le Cristal, la Tourmaline et le Graphite '
        "figurent parmi les substances les plus fréquentes. De nombreux permis combinent plusieurs substances.</div>",
        unsafe_allow_html=True,
    )

# TAB Titulaires
with tab_tit:
    st.subheader("Analyse des titulaires")
    top_tit = fdf["titulaire"].value_counts().nlargest(20).reset_index()
    top_tit.columns = ["Titulaire", "Nombre de permis"]
    fig_t = px.bar(top_tit, x="Nombre de permis", y="Titulaire", orientation="h",
                   title="Top 20 titulaires par nombre de permis", color="Nombre de permis",
                   color_continuous_scale="Blues", text="Nombre de permis")
    fig_t.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
    st.plotly_chart(fig_t, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.metric("Titulaires distincts (filtrés)", fdf["titulaire"].nunique())
    with col_t2:
        if len(top_tit):
            top1 = top_tit.iloc[0]
            st.metric("Titulaire le plus actif", f"{str(top1['Titulaire'])[:40]}… ({int(top1['Nombre de permis'])})")

    st.subheader("Superficie totale par titulaire (top 15)")
    area = fdf.groupby("titulaire")["nombre_carres"].sum().nlargest(15).reset_index().rename(
        columns={"nombre_carres": "Total carrés"})
    fig_a = px.bar(area, x="Total carrés", y="titulaire", orientation="h", title="Superficie cumulée (carrés)",
                   color="Total carrés", color_continuous_scale="Teal")
    fig_a.update_layout(yaxis={"categoryorder": "total ascending"}, height=480)
    st.plotly_chart(fig_a, use_container_width=True)

# TAB Carte
with tab_map:
    st.subheader("Cartographie des permis miniers")
    st.markdown(
        '<div class="info-box">Les données sources ne contiennent pas de GPS. Des <b>coordonnées approximatives</b> '
        "ont été associées aux principales localités (OpenStreetMap / géographie publique). "
        "Les localités non référencées n'apparaissent pas sur la carte. "
        "Vous pouvez aussi ouvrir une localité sur OpenStreetMap ou Google Maps.</div>",
        unsafe_allow_html=True,
    )
    loc_counts = fdf["region_principale"].value_counts().reset_index()
    loc_counts.columns = ["localite", "nombre"]
    loc_counts["lat"] = loc_counts["localite"].map(lambda x: LOCALITY_COORDS.get(x, (None, None))[0])
    loc_counts["lon"] = loc_counts["localite"].map(lambda x: LOCALITY_COORDS.get(x, (None, None))[1])
    geo_df = loc_counts.dropna(subset=["lat", "lon"])
    st.write(f"**{len(geo_df)}** localités géoréférencées sur **{len(loc_counts)}** localités présentes.")

    if len(geo_df) > 0:
        fig_map = px.scatter_mapbox(
            geo_df, lat="lat", lon="lon", size="nombre", color="nombre",
            hover_name="localite", hover_data={"nombre": True, "lat": False, "lon": False},
            color_continuous_scale="YlOrRd", size_max=35, zoom=5.2,
            center={"lat": -19.5, "lon": 46.5}, mapbox_style="open-street-map",
            title="Carte interactive — concentration des permis par localité", height=560,
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

        if HAS_FOLIUM:
            st.subheader("Carte Folium (alternative)")
            m = folium.Map(location=[-19.5, 46.5], zoom_start=6, tiles="OpenStreetMap")
            for _, row in geo_df.iterrows():
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=min(4 + row["nombre"] * 0.35, 25),
                    popup=f"<b>{row['localite']}</b><br>{int(row['nombre'])} permis",
                    tooltip=row["localite"], color="#0f4c5c", fill=True,
                    fill_color="#2d9f8f", fill_opacity=0.65,
                ).add_to(m)
            st_folium(m, width=None, height=500)

    st.subheader("Ouvrir une localité sur un site tiers")
    choice = st.selectbox("Choisir une localité", options=sorted(loc_counts["localite"].unique()),
                          index=0 if len(loc_counts) else None)
    if choice:
        coords = LOCALITY_COORDS.get(choice)
        q = choice.replace(" ", "+") + ",+Madagascar"
        osm_url = f"https://www.openstreetmap.org/search?query={q}"
        gmaps_url = f"https://www.google.com/maps/search/?api=1&query={q}"
        if coords:
            osm_url = f"https://www.openstreetmap.org/?mlat={coords[0]}&mlon={coords[1]}#map=10/{coords[0]}/{coords[1]}"
            gmaps_url = f"https://www.google.com/maps?q={coords[0]},{coords[1]}"
        c1, c2 = st.columns(2)
        c1.markdown(f"[🗺️ Ouvrir dans OpenStreetMap]({osm_url})")
        c2.markdown(f"[📍 Ouvrir dans Google Maps]({gmaps_url})")

    st.subheader("Classement des localités")
    st.dataframe(
        loc_counts[["localite", "nombre"]].head(30).rename(columns={"localite": "Localité", "nombre": "Nb permis"}),
        use_container_width=True, hide_index=True,
    )

# TAB Temporalité
with tab_temp:
    st.subheader("Évolution temporelle des octrois")
    par_annee = fdf.dropna(subset=["annee_octroi"]).groupby("annee_octroi").size().reset_index(name="nombre")
    fig3 = px.line(par_annee, x="annee_octroi", y="nombre", markers=True,
                   title="Nombre de permis octroyés par année", color_discrete_sequence=["#0f4c5c"])
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, use_container_width=True)

    if len(par_annee):
        year_sel = st.slider("Filtrer l'affichage autour d'une année",
                             int(par_annee["annee_octroi"].min()), int(par_annee["annee_octroi"].max()),
                             int(par_annee["annee_octroi"].median()))
        window = fdf[(fdf["annee_octroi"] >= year_sel - 2) & (fdf["annee_octroi"] <= year_sel + 2)]
        st.caption(f"Permis entre {year_sel - 2} et {year_sel + 2} : **{len(window)}**")
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            t_counts = window["type"].value_counts().reset_index()
            t_counts.columns = ["type", "n"]
            st.plotly_chart(px.pie(t_counts, names="type", values="n", title=f"Types autour de {year_sel}", hole=0.4),
                            use_container_width=True)
        with col_y2:
            s_counts = window["substance_principale"].value_counts().nlargest(8).reset_index()
            s_counts.columns = ["substance", "n"]
            st.plotly_chart(px.bar(s_counts, x="substance", y="n", title=f"Substances autour de {year_sel}"),
                            use_container_width=True)

    st.subheader("Dernier FA payé")
    fa = fdf["dernier_fa_paye"].value_counts().sort_index().reset_index()
    fa.columns = ["Année FA", "Nombre"]
    fa = fa[fa["Année FA"] > 0]
    fig_fa = px.bar(fa, x="Année FA", y="Nombre", title="Répartition du dernier paiement FA",
                    color="Nombre", color_continuous_scale="Teal")
    st.plotly_chart(fig_fa, use_container_width=True)
    st.markdown(
        '<div class="info-box"><b>Pics d\'activité</b> observés notamment en 1999, 2001, 2015 et 2016. '
        "Forte activité de renouvellement et de cession dans les procédures en cours.</div>",
        unsafe_allow_html=True,
    )

# TAB IA
with tab_ia:
    st.subheader("Prédiction du risque d'annulation d'un permis")
    st.markdown(
        "Modèle **Random Forest** entraîné sur les caractéristiques administratives des permis "
        "(type, superficie, substances, durée, année d'octroi, régularité FA, localité, substance)."
    )
    if eval_report:
        best = eval_report.get("best_model", "Random Forest")
        st.success(f"Modèle retenu : **{best}** (F1 = {eval_report.get('best_f1', 0):.3f})")
        results = eval_report.get("results", {})
        if results:
            rows = [{"Modèle": name, "Accuracy": round(m["accuracy"], 3), "Precision": round(m["precision"], 3),
                     "Recall": round(m["recall"], 3), "F1-score": round(m["f1_score"], 3)} for name, m in results.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if "Random Forest" in results:
                m = results["Random Forest"]
                cols = st.columns(4)
                cols[0].metric("Accuracy", f"{m['accuracy']:.1%}")
                cols[1].metric("Precision", f"{m['precision']:.1%}")
                cols[2].metric("Recall", f"{m['recall']:.1%}")
                cols[3].metric("F1-score", f"{m['f1_score']:.3f}")

    st.subheader("🔮 Simuler une prédiction")
    col1, col2, col3 = st.columns(3)
    with col1:
        in_type = st.selectbox("Type de permis", sorted(df["type"].unique()))
        in_carres = st.number_input("Nombre de carrés", min_value=1, value=32)
        in_nb_subst = st.slider("Nombre de substances", 1, 10, 1)
    with col2:
        in_duree = st.number_input("Durée de validité (années)", min_value=0.0, value=20.0)
        in_annee = st.number_input("Année d'octroi", min_value=1990, max_value=2026, value=2015)
        in_fa_recent = st.selectbox("FA payé récemment (≥ 2015) ?", ["Oui", "Non"])
    with col3:
        regions = sorted(df["region_principale"].dropna().unique())
        in_region = st.selectbox("Région / localité principale", regions[:250] if len(regions) > 250 else regions)
        substances = sorted(df["substance_principale"].dropna().unique())
        in_substance = st.selectbox("Substance principale", substances[:250] if len(substances) > 250 else substances)

    if st.button("Prédire le risque d'annulation", type="primary"):
        model = model_bundle["model"]
        encoders = model_bundle["encoders"]
        features = model_bundle["features"]

        def encode_safe(encoder, value):
            return encoder.transform([value])[0] if value in encoder.classes_ else 0

        row = pd.DataFrame([{
            "type": encode_safe(encoders["type"], in_type),
            "nombre_carres": in_carres,
            "nb_substances": in_nb_subst,
            "duree_validite_annees": in_duree,
            "annee_octroi": in_annee,
            "fa_recent": 1 if in_fa_recent == "Oui" else 0,
            "region_principale": encode_safe(encoders["region_principale"], in_region),
            "substance_principale": encode_safe(encoders["substance_principale"], in_substance),
        }])[features]
        proba = model.predict_proba(row)[0][1]
        pred = model.predict(row)[0]
        if pred == 1:
            st.error(f"⚠️ Permis classé **à risque d'annulation** (probabilité : {proba:.1%})")
        else:
            st.success(f"✅ Permis classé **régulier** (probabilité de risque : {proba:.1%})")
        fig_p = go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100, title={"text": "Probabilité de risque (%)"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#c0392b" if pred == 1 else "#1a7a6d"},
                   "steps": [{"range": [0, 30], "color": "#d5f5e3"},
                             {"range": [30, 60], "color": "#fdebd0"},
                             {"range": [60, 100], "color": "#f5b7b1"}],
                   "threshold": {"line": {"color": "black", "width": 2}, "value": 50}}))
        fig_p.update_layout(height=280)
        st.plotly_chart(fig_p, use_container_width=True)

# TAB Données
with tab_data:
    st.subheader("Données filtrées")
    st.caption(f"{len(fdf):,} lignes · {len(fdf.columns)} colonnes")
    st.dataframe(fdf, use_container_width=True, height=480)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Exporter CSV", fdf.to_csv(index=False).encode("utf-8"),
                           file_name="permis_miniers_filtres.csv", mime="text/csv")
    with c2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            fdf.to_excel(writer, index=False, sheet_name="permis")
        st.download_button("⬇️ Exporter Excel (XLSX)", buffer.getvalue(),
                           file_name="permis_miniers_filtres.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with st.expander("Dictionnaire des colonnes"):
        st.markdown("""
| Colonne | Description |
|---|---|
| `num_permis` | Identifiant du permis |
| `type` | R (recherche), PRE (préliminaire), E (exploitation) |
| `titulaire` | Détenteur du permis |
| `classification` | Statut administratif |
| `nombre_carres` | Superficie en carrés miniers |
| `substances` | Substances déclarées |
| `date_octroi` / `date_fin_validite` | Dates de validité |
| `dernier_fa_paye` | Dernière année de paiement FA |
| `region_principale` | Localité principale extraite |
| `substance_principale` | Première substance listée |
| `a_risque_annulation` | Cible IA (1 = risque) |
| `fa_recent` | FA payé ≥ 2015 |
""")
