# ============================================================================
# PEC FRAMEWORK — Généralisation Multi-Domaines
# Plateforme Streamlit (port 8504)
# Auteur : Rosa Elysabeth Ralinirina | ORCID : 0009-0003-3048-1765
# ============================================================================
# PREDICT — EXPLAIN — CERTIFY : 3 domaines, 3 continents, 9 cibles
# Climat (Asie du Sud-Est) | Fintech (Afrique de l'Ouest) | Santé (Afrique de l'Est)
# ============================================================================

import streamlit as st

def pec_url(app):
    """URL dynamique: localhost en local, secrets.toml en cloud."""
    local = {'mada': 'http://localhost:8501', 'aide': 'http://localhost:8502',
             'whatif': 'http://localhost:8503', 'gen': 'http://localhost:8504'}
    try:
        cloud = {'mada': st.secrets.urls.mada, 'aide': st.secrets.urls.aide,
                 'whatif': st.secrets.urls.whatif, 'gen': st.secrets.urls.gen}
        return cloud.get(app, local.get(app, 'http://localhost:8501'))
    except Exception:
        return local.get(app, 'http://localhost:8501')

import pandas as pd
import numpy as np
import pickle, joblib, json, os, warnings
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PEC Généralisation | Multi-Domaines",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CHEMINS
# ============================================================================
BASE = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.join(BASE, "generalisation")
DATA_DIR = os.path.join(GEN_DIR, "data")
MODELS_DIR = os.path.join(GEN_DIR, "models")
RESULTS_DIR = os.path.join(GEN_DIR, "results")

# ============================================================================
# CSS PARTAGÉ
# ============================================================================
st.markdown("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">""", unsafe_allow_html=True)
st.markdown("""<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">""", unsafe_allow_html=True)

_css_path = os.path.join(BASE, 'shared_pec.css')
if os.path.exists(_css_path):
    with open(_css_path, 'r', encoding='utf-8') as _f:
        st.markdown(f'<style>{_f.read()}</style>', unsafe_allow_html=True)
else:
    st.markdown("""<style>
:root {
  --primary: #1a5276; --primary-light: #2e86c1; --secondary: #27ae60;
  --purple: #8e44ad; --red: #c0392b; --orange: #f39c12;
  --bg: #f4f6f9; --card: #fff; --text: #1a1a2e; --muted: #6b7280;
  --border: #e5e7eb; --radius: 12px; --shadow: 0 4px 24px rgba(0,0,0,.06);
}
.interp-box { background:#eef2ff; border-left:4px solid #2e86c1; border-radius:8px; padding:1rem 1.2rem; margin:1rem 0; font-size:0.92rem; line-height:1.55; color:#1a1a2e; }
.interp-box b { color:#1a5276; }
</style>""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION DES 3 DOMAINES
# ============================================================================
DOMAINES = {
    'climat': {
        'nom_fr': 'Climat — Asie du Sud-Est',
        'nom_en': 'Climate — Southeast Asia',
        'icone': 'fa-cloud-bolt',
        'couleur': '#2e86c1',
        'emoji': '🌏',
        'fichier': 'dataset_climat.csv',
        'zone_col': 'Zone',
        'annee_col': 'Annee',
        'cibles': {
            'Indice_Risque_Inondation': {'fr': 'Indice de Risque d\'Inondation', 'en': 'Flood Risk Index', 'unit': ''},
            'Degats_millions_USD': {'fr': 'Dégâts (millions USD)', 'en': 'Damages (M USD)', 'unit': 'M USD'},
            'Population_a_risque_M': {'fr': 'Population à Risque (M)', 'en': 'Population at Risk (M)', 'unit': 'M'},
        },
        'description_fr': 'Données climatiques couvrant 12 zones d\'Asie du Sud-Est (2014-2024) : précipitations, température, humidité, élévation, couverture forestière, cyclones, infrastructures de protection.',
        'description_en': 'Climate data covering 12 Southeast Asian zones (2014-2024): precipitation, temperature, humidity, elevation, forest cover, cyclones, protection infrastructure.',
        'adaptation_certification_fr': 'Seuil CV&lt;15% (Coefficient de Variation) au lieu de τR≥0,80 — les données climatiques sont plus volatiles, donc on mesure la stabilité relative.',
        'adaptation_certification_en': 'CV&lt;15% threshold (Coefficient of Variation) instead of τR≥0.80 — climate data is more volatile, so we measure relative stability.',
        'interp_predict_fr': 'Le domaine climatique montre que <b>Random Forest</b> surpasse Ridge sur l\'indice de risque d\'inondation (R²=0,89) grâce à sa capacité à capturer les interactions non-linéaires entre précipitations, élévation et infrastructures. Cependant, <b>Ridge domine</b> sur les dégâts (R²=0,96) et la population à risque (R²=0,99) grâce à la forte autocorrélation temporelle de ces variables.',
        'interp_predict_en': 'The climate domain shows that <b>Random Forest</b> outperforms Ridge on the flood risk index (R²=0.89) thanks to its ability to capture non-linear interactions between precipitation, elevation and infrastructure. However, <b>Ridge dominates</b> on damages (R²=0.96) and population at risk (R²=0.99) due to strong temporal autocorrelation.',
        'interp_explain_fr': 'Les variables les plus influentes en climat sont les <b>moyennes mobiles 3 ans (ma3)</b> et les <b>valeurs retardées (lag)</b>, confirmant la persistance temporelle des risques. Les précipitations maximales et la couverture forestière jouent aussi un rôle clé. Les zonesà forte densité de population et faible infrastructure sont les plus vulnérables.',
        'interp_explain_en': 'The most influential variables in climate are <b>3-year moving averages (ma3)</b> and <b>lagged values</b>, confirming temporal persistence of risks. Maximum precipitation and forest cover also play a key role. Areas with high population density and weak infrastructure are the most vulnerable.',
        'interp_certify_fr': 'Le domaine climat obtient <b>2 Grades A et 1 Grade B</b>. Le Grade B sur l\'indice de risque s\'explique par la plus grande volatilité des événements extrêmes (cyclones, inondations soudaines). L\'adaptation du seuil (CV&lt;15% au lieu de τR≥0,80) prend en compte cette volatilité naturelle.',
        'interp_certify_en': 'The climate domain achieves <b>2 Grades A and 1 Grade B</b>. The Grade B on the risk index is explained by greater volatility of extreme events (cyclones, sudden floods). The adapted threshold (CV&lt;15% instead of τR≥0.80) accounts for this natural volatility.',
    },
    'fintech': {
        'nom_fr': 'Fintech — Afrique de l\'Ouest',
        'nom_en': 'Fintech — West Africa',
        'icone': 'fa-coins',
        'couleur': '#f39c12',
        'emoji': '🏦',
        'fichier': 'dataset_fintech.csv',
        'zone_col': 'Pays',
        'annee_col': 'Annee',
        'cibles': {
            'Taux_Defaut_pct': {'fr': 'Taux de Défaut (%)', 'en': 'Default Rate (%)', 'unit': '%'},
            'Ratio_NPL_pct': {'fr': 'Ratio de Prêts Improductifs (%)', 'en': 'NPL Ratio (%)', 'unit': '%'},
            'Penetration_Fintech_pct': {'fr': 'Pénétration Fintech (%)', 'en': 'Fintech Penetration (%)', 'unit': '%'},
        },
        'description_fr': 'Données bancaires et fintech couvrant 15 pays d\'Afrique de l\'Ouest (2016-2024) : PIB, adoption fintech, taux d\'intérêt, littératie, mobile money, réglementation.',
        'description_en': 'Banking and fintech data covering 15 West African countries (2016-2024): GDP, fintech adoption, interest rates, literacy, mobile money, regulation.',
        'adaptation_certification_fr': 'Poids <b>wT=0,40</b> (au lieu de 0,30) — l\'équité territoriale est critique en fintech pour éviter qu\'un pays soit systématiquement désavantagé par les modèles de scoring de crédit.',
        'adaptation_certification_en': 'Weight <b>wT=0.40</b> (instead of 0.30) — territorial equity is critical in fintech to prevent any country from being systematically disadvantaged by credit scoring models.',
        'interp_predict_fr': 'Le domaine fintech montre que <b>CatBoost</b> surpasse Ridge sur le taux de défaut (R²=0,85) et le ratio NPL (R²=0,87), grâce à son traitement natif des variables catégorielles (pays, réglementation). En revanche, <b>Ridge domine</b> largement sur la pénétration fintech (R²=0,99), une variable à forte tendance temporelle.',
        'interp_predict_en': 'The fintech domain shows that <b>CatBoost</b> outperforms Ridge on the default rate (R²=0.85) and NPL ratio (R²=0.87), thanks to its native handling of categorical variables (country, regulation). However, <b>Ridge dominates</b> on fintech penetration (R²=0.99), a variable with strong temporal trends.',
        'interp_explain_fr': 'Les variables les plus influentes en fintech sont l\'<b>adoption du mobile money</b>, le <b>PIB par habitant</b> et la <b>pénétration d\'Internet</b>. Le taux de littératie et l\'indice de réglementation jouent un rôle modérateur. Les pays avec une forte instabilité politique montrent des profils de risque plus élevés.',
        'interp_explain_en': 'The most influential variables in fintech are <b>mobile money adoption</b>, <b>GDP per capita</b> and <b>Internet penetration</b>. Literacy rate and regulation index play a moderating role. Countries with high political instability show higher risk profiles.',
        'interp_certify_fr': 'Le domaine fintech obtient <b>1 Grade A et 2 Grades B</b>. Les Grades B reflètent les biais structurels des données bancaires ouest-africaines : hétérogénéité des systèmes financiers, données limitées, et effets confondants de la réglementation. L\'adaptation <b>wT=0,40</b> renforce l\'exigence d\'équité entre pays.',
        'interp_certify_en': 'The fintech domain achieves <b>1 Grade A and 2 Grades B</b>. The Grade Bs reflect structural biases in West African banking data: heterogeneity of financial systems, limited data, and confounding effects of regulation. The <b>wT=0.40</b> adaptation strengthens the equity requirement between countries.',
    },
    'sante': {
        'nom_fr': 'Santé — Afrique de l\'Est',
        'nom_en': 'Health — East Africa',
        'icone': 'fa-heart-pulse',
        'couleur': '#27ae60',
        'emoji': '🏥',
        'fichier': 'dataset_sante.csv',
        'zone_col': 'Province',
        'annee_col': 'Annee',
        'cibles': {
            'Mortalite_infantile_p1000': {'fr': 'Mortalité Infantile (‰)', 'en': 'Infant Mortality (‰)', 'unit': '‰'},
            'Vaccination_complete_pct': {'fr': 'Vaccination Complète (%)', 'en': 'Complete Vaccination (%)', 'unit': '%'},
            'Accouchements_qualified_pct': {'fr': 'Accouchements Qualifiés (%)', 'en': 'Skilled Births (%)', 'unit': '%'},
        },
        'description_fr': 'Données de santé publique couvrant 14 provinces d\'Afrique de l\'Est (2014-2024) : couverture vaccinale, accouchements qualifiés, eau potable, paludisme, anémie, dépenses santé.',
        'description_en': 'Public health data covering 14 East African provinces (2014-2024): vaccination coverage, skilled births, safe water, malaria, anemia, health spending.',
        'adaptation_certification_fr': '<b>AUC</b> (Aire sous la courbe ROC) au lieu de R² — les cibles de santé sont des indicateurs de santé publique. AUC=1 (parfait), 0,5 (aléatoire).',
        'adaptation_certification_en': '<b>AUC</b> (Area Under the ROC Curve) instead of R² — health targets are public health indicators. AUC=1 (perfect), 0.5 (random).',
        'interp_predict_fr': 'Le domaine santé montre que <b>Ridge domine sur les 3 cibles</b> (R² de 0,97 à 0,99), confirmant que les indicateurs de santé publique suivent des tendances temporelles fortes et prévisibles. Les modèles boîte noire n\'apportent pas d\'amélioration significative sur ces séries lisses.',
        'interp_predict_en': 'The health domain shows that <b>Ridge dominates on all 3 targets</b> (R² from 0.97 to 0.99), confirming that public health indicators follow strong and predictable temporal trends. Black-box models do not bring significant improvement on these smooth series.',
        'interp_explain_fr': 'Les variables les plus influentes en santé sont les <b>moyennes mobiles (ma3)</b>, les <b>valeurs retardées (lag)</b>, la <b>couverture vaccinale antérieure</b> et les <b>dépenses de santé</b>. Le paludisme et l\'anémie contribuent négativement. Les provinces avec le plus faible accès à l\'eau potable montrent les pires indicateurs.',
        'interp_explain_en': 'The most influential variables in health are <b>moving averages (ma3)</b>, <b>lagged values</b>, <b>previous vaccination coverage</b> and <b>health spending</b>. Malaria and anemia contribute negatively. Provinces with the lowest access to safe water show the worst indicators.',
        'interp_certify_fr': 'Le domaine santé obtient <b>3 Grades A</b> (scores de 97,7 à 99,2/100). Ces résultats élevés reflètent la forte prévisibilité des indicateurs de santé publique et la stabilité des explications entre provinces. L\'adaptation AUC confirme que les modèles discriminent bien les provinces à haut risque.',
        'interp_certify_en': 'The health domain achieves <b>3 Grades A</b> (scores from 97.7 to 99.2/100). These high results reflect the strong predictability of public health indicators and the stability of explanations between provinces. The AUC adaptation confirms that models discriminate well between high-risk provinces.',
    },
}

# ============================================================================
# FONCTIONS DE CHARGEMENT
# ============================================================================
@st.cache_data
def load_dataset(domaine_key):
    path = os.path.join(DATA_DIR, DOMAINES[domaine_key]['fichier'])
    return pd.read_csv(path)

@st.cache_data
def load_rapport(domaine_key):
    path = os.path.join(RESULTS_DIR, domaine_key, 'rapport_final.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def load_models_domaine(domaine_key):
    mdir = os.path.join(MODELS_DIR, domaine_key)
    models = {}
    feature_cols = []
    feature_stats = {}
    scaler = None
    if os.path.exists(os.path.join(mdir, 'feature_cols.pkl')):
        with open(os.path.join(mdir, 'feature_cols.pkl'), 'rb') as f:
            feature_cols = pickle.load(f)
    if os.path.exists(os.path.join(mdir, 'feature_stats.pkl')):
        with open(os.path.join(mdir, 'feature_stats.pkl'), 'rb') as f:
            feature_stats = pickle.load(f)
    if os.path.exists(os.path.join(mdir, 'scaler.pkl')):
        with open(os.path.join(mdir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
    for mn in ['Ridge', 'Random_Forest', 'XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting']:
        for target_info in DOMAINES[domaine_key]['cibles']:
            path = os.path.join(mdir, f"{mn}_{target_info}.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    models[(mn, target_info)] = pickle.load(f)
    return feature_cols, feature_stats, scaler, models

def load_image(path):
    if os.path.exists(path):
        return Image.open(path)
    return None

# ============================================================================
# SIDEBAR — UNIFORME AVEC LES 3 AUTRES PLATEFORMES
# ============================================================================
with st.sidebar:
    lang = st.radio('', ['fr','en'], format_func=lambda x: 'Français' if x=='fr' else 'English', horizontal=True, key='lg')
    is_fr = lang == 'fr'
    st.divider()

    # Navigation inter-plateformes — IDENTIQUE aux 3 autres apps
    st.divider()
    st.markdown(f'#### 🔗 {"Plateformes PEC" if is_fr else "PEC Platforms"}')
    c1, c2 = st.columns(2)
    with c1: st.link_button(f'📊 {"PEC Mada" if is_fr else "PEC Mada"}', pec_url('mada'), use_container_width=True)
    with c2: st.link_button(f'📊 {"PEC Aide" if is_fr else "PEC Aid"}', pec_url('aide'), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3: st.link_button(f'🔬 {"PEC What" if is_fr else "PEC What"}', pec_url('whatif'), use_container_width=True)
    with c4: st.link_button(f'🌍 {"PEC Gen" if is_fr else "PEC Gen"}', pec_url('gen'), use_container_width=True)
    st.divider()

    st.markdown(f"**{'Auteur' if is_fr else 'Author'}**")
    st.markdown(f'<a href="https://orcid.org/0009-0003-3048-1765" target="_blank" style="text-decoration:none;display:inline-flex;align-items:center;gap:4px;font-size:.9rem;color:#a6ce39;"><img src="https://orcid.org/assets/vectors/orcid.logo.icon.svg" width="16" height="16" alt="ORCID" style="vertical-align:middle;"/> orcid.org/0009-0003-3048-1765</a>', unsafe_allow_html=True)
    st.divider()

    domaine_choisi = st.selectbox(
        "Domaine" if is_fr else "Domain",
        list(DOMAINES.keys()),
        format_func=lambda k: f"{DOMAINES[k]['emoji']} {DOMAINES[k]['nom_fr'] if is_fr else DOMAINES[k]['nom_en']}"
    )
    st.divider()

    # Section sera geree par tabs dans le corps principal

# ============================================================================
# DONNÉES DU DOMAINE CHOISI
# ============================================================================
dom = DOMAINES[domaine_choisi]
df = load_dataset(domaine_choisi)
rapport = load_rapport(domaine_choisi)
feature_cols, feature_stats, scaler, all_models = load_models_domaine(domaine_choisi)

zone_col = dom['zone_col']
annee_col = dom['annee_col']
zones = sorted(df[zone_col].unique())
annees = sorted(df[annee_col].unique())

# ============================================================================
# EN-TÊTE
# ============================================================================
st.markdown(f"""
<h1 class="mt" style="color:{dom['couleur']}">
    <i class="fa-solid {dom['icone']}"></i> 
    {'Généralisation PEC — ' if is_fr else 'PEC Generalization — '}{dom['nom_fr'] if is_fr else dom['nom_en']}
</h1>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="pec-card" style="border-left:5px solid {dom['couleur']}">
<p>{dom['description_fr'] if is_fr else dom['description_en']}</p>
<p><strong>{'Observations' if is_fr else 'Observations'}:</strong> {len(df)} | 
<strong>{'Variables' if is_fr else 'Variables'}:</strong> {len(df.columns)} | 
<strong>{'Zones' if is_fr else 'Zones'}:</strong> {len(zones)} | 
<strong>{'Période' if is_fr else 'Period'}:</strong> {annees[0]}–{annees[-1]}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# VUE D'ENSEMBLE
# ============================================================================
# TABS
tabs = st.tabs(["Vue d'ensemble" if is_fr else "Overview", "PREDICT", "EXPLAIN", "CERTIFY", "À propos" if is_fr else "About"])
with tabs[0]:
    st.markdown(f"<h2 class='st'>{'📊 Vue d\'ensemble — 3 Domaines' if is_fr else '📊 Overview — 3 Domains'}</h2>", unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, (dkey, dinfo) in enumerate(DOMAINES.items()):
        rp = load_rapport(dkey)
        with cols[idx]:
            grades = []
            for t, cdata in rp.get('certify', {}).items():
                grades.append(cdata.get('grade', '?'))
            n_A = sum(1 for g in grades if g == 'A')
            n_B = sum(1 for g in grades if g == 'B')

            st.markdown(f"""
            <div class="pec-card" style="border-left:5px solid {dinfo['couleur']}; text-align:center">
                <h3 style="color:{dinfo['couleur']}">{dinfo['emoji']} {dinfo['nom_fr'] if is_fr else dinfo['nom_en']}</h3>
                <p style="font-size:2rem; font-weight:800; color:{dinfo['couleur']}">{n_A}A + {n_B}B</p>
                <p style="font-size:0.9rem; color:#6b7280">
                    {rp['meta']['observations']} {'obs' if is_fr else 'obs'} | 
                    {rp['meta']['variables']} {'vars' if is_fr else 'vars'} | 
                    {rp['meta']['regions']} {'zones' if is_fr else 'zones'}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Tableau comparatif
    st.markdown(f"<h2 class='st'>{'📋 Résultats comparatifs' if is_fr else '📋 Comparative Results'}</h2>", unsafe_allow_html=True)

    rows = []
    for dkey, dinfo in DOMAINES.items():
        rp = load_rapport(dkey)
        domaine_nom = dinfo['nom_fr'] if is_fr else dinfo['nom_en']
        for tkey, cdata in rp.get('certify', {}).items():
            target_nom = dinfo['cibles'].get(tkey, {}).get('fr' if is_fr else 'en', tkey)
            modele = cdata.get('model', '?').replace('_', ' ')
            r2 = cdata.get('R2_test', 0)
            score = cdata.get('certification_score', 0)
            grade = cdata.get('grade', '?')
            rows.append({
                'Domaine' if is_fr else 'Domain': domaine_nom,
                'Cible' if is_fr else 'Target': target_nom,
                'Modèle' if is_fr else 'Model': modele,
                'R²': f"{r2:.4f}",
                'Score PEC': f"{score:.1f}/100",
                'Grade': grade,
            })

    df_comp = pd.DataFrame(rows)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # Interprétation
    st.markdown(f"""
    <div class="interp-box">
        <strong><i class="fa-solid fa-lightbulb"></i> {'Interprétation' if is_fr else 'Interpretation'} :</strong><br>
        {'<b>Ridge</b> (modèle interprétable) gagne <b>8 cibles sur 9</b>. L\'interprétabilité n\'est pas un sacrifice — elle est un avantage. La persistance temporelle (MA3, Lag1) domine dans les 9 cibles. La certification est transférable : Grade A ou B sur toutes les cibles (87,4 à 99,2/100).'
        if is_fr else
        '<b>Ridge</b> (interpretable model) wins <b>8 out of 9 targets</b>. Interpretability is not a sacrifice — it is an advantage. Temporal persistence (MA3, Lag1) dominates in all 9 targets. Certification is transferable: Grade A or B on all targets (87.4 to 99.2/100).'}
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PREDICT
# ============================================================================
with tabs[1]:
    st.markdown(f"<h2 class='st'>{'📈 PREDICT — ' if is_fr else '📈 PREDICT — '}{dom['nom_fr'] if is_fr else dom['nom_en']}</h2>", unsafe_allow_html=True)

    # KPIs
    cols_kpi = st.columns(len(dom['cibles']))
    for idx, (tkey, tinfo) in enumerate(dom['cibles'].items()):
        t_nom = tinfo['fr'] if is_fr else tinfo['en']
        predict_data = rapport.get('predict', {}).get(tkey, {})
        meilleur = predict_data.get('meilleur_modele', '?').replace('_', ' ')
        r2 = predict_data.get('R2_test', 0)
        with cols_kpi[idx]:
            st.metric(t_nom, f"{r2:.4f}", f"{'Modèle' if is_fr else 'Model'}: {meilleur}")

    # Comparaison des modèles (image)
    img_path = os.path.join(RESULTS_DIR, domaine_choisi, 'figures', 'comparaison_modeles.png')
    img = load_image(img_path)
    if img:
        st.image(img, caption="Comparaison des modèles" if is_fr else "Model comparison", use_container_width=True)

    # Interprétation PREDICT
    interp_key = 'interp_predict_fr' if is_fr else 'interp_predict_en'
    if interp_key in dom:
        st.markdown(f"""
        <div class="interp-box">
            <strong><i class="fa-solid fa-lightbulb"></i> {'Interprétation' if is_fr else 'Interpretation'} :</strong><br>
            {dom[interp_key]}
        </div>
        """, unsafe_allow_html=True)

    # Détail par cible
    for tkey, tinfo in dom['cibles'].items():
        t_nom = tinfo['fr'] if is_fr else tinfo['en']
        st.markdown(f"<h3 style='color:{dom['couleur']}'>{t_nom}</h3>", unsafe_allow_html=True)

        predict_data = rapport.get('predict', {}).get(tkey, {})
        if predict_data:
            tous = predict_data.get('tous_modeles', {})
            if tous:
                rows_m = []
                for mn, mdata in tous.items():
                    rows_m.append({
                        'Modèle' if is_fr else 'Model': mn.replace('_', ' '),
                        'R²': f"{mdata.get('R2', 0):.4f}",
                        'RMSE': f"{mdata.get('RMSE', 0):.2f}",
                        'MAE': f"{mdata.get('MAE', 0):.2f}",
                    })
                st.dataframe(pd.DataFrame(rows_m), use_container_width=True, hide_index=True)

# ============================================================================
# EXPLAIN
# ============================================================================
with tabs[2]:
    st.markdown(f"<h2 class='st'>{'🔬 EXPLAIN — ' if is_fr else '🔬 EXPLAIN — '}{dom['nom_fr'] if is_fr else dom['nom_en']}</h2>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pec-card card-explain">
        <p>{'<b>Triangulation multi-méthodes</b> : SHAP + LIME + Permutation Importance + ALE. Une explication est recevable si τ de Kendall > 0,75 entre les classements.'
        if is_fr else
        '<b>Multi-method triangulation</b>: SHAP + LIME + Permutation Importance + ALE. An explanation is acceptable if Kendall τ > 0.75 between rankings.'}</p>
    </div>
    """, unsafe_allow_html=True)

    for tkey, tinfo in dom['cibles'].items():
        t_nom = tinfo['fr'] if is_fr else tinfo['en']
        st.markdown(f"<h3 style='color:{dom['couleur']}'>{t_nom}</h3>", unsafe_allow_html=True)

        # SHAP summary
        img_summary = load_image(os.path.join(RESULTS_DIR, domaine_choisi, 'shap', f'shap_summary_{tkey}.png'))
        if img_summary:
            st.image(img_summary, caption=f"SHAP Summary — {t_nom}", use_container_width=True)

        # SHAP bar
        img_bar = load_image(os.path.join(RESULTS_DIR, domaine_choisi, 'shap', f'shap_bar_{tkey}.png'))
        if img_bar:
            st.image(img_bar, caption=f"SHAP Bar — {t_nom}", use_container_width=True)

        # Interprétation locale par zone
        zone_sel = st.selectbox(
            f"{'Zone' if is_fr else 'Zone'} — {t_nom}",
            zones,
            key=f"expl_{domaine_choisi}_{tkey}"
        )

        # Données SHAP pour cette zone si disponibles
        shap_data = rapport.get('explain', {}).get(tkey, {}).get(zone_sel, {})
        if shap_data:
            st.markdown(f"""
            <div class="interp-box">
                <strong><i class="fa-solid fa-lightbulb"></i> {'Interprétation locale' if is_fr else 'Local interpretation'} — {zone_sel} :</strong><br>
                {shap_data.get('interpretation_fr' if is_fr else 'interpretation_en', '')}
            </div>
            """, unsafe_allow_html=True)

    # Interprétation EXPLAIN globale
    interp_key = 'interp_explain_fr' if is_fr else 'interp_explain_en'
    if interp_key in dom:
        st.markdown(f"""
        <div class="interp-box">
            <strong><i class="fa-solid fa-lightbulb"></i> {'Interprétation globale' if is_fr else 'Global interpretation'} :</strong><br>
            {dom[interp_key]}
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# CERTIFY
# ============================================================================
with tabs[3]:
    st.markdown(f"<h2 class='st'>{'✅ CERTIFY — ' if is_fr else '✅ CERTIFY — '}{dom['nom_fr'] if is_fr else dom['nom_en']}</h2>", unsafe_allow_html=True)

    # Adaptation du domaine
    cert_adapt_key = 'adaptation_certification_fr' if is_fr else 'adaptation_certification_en'
    st.markdown(f"""
    <div class="pec-card card-certify">
        <h4>{'Adaptation du domaine' if is_fr else 'Domain adaptation'}</h4>
        <p>{dom.get(cert_adapt_key, '')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Certification composite
    img_cert = load_image(os.path.join(RESULTS_DIR, domaine_choisi, 'certification', 'certification_composite.png'))
    if img_cert:
        st.image(img_cert, caption="Certification composite" if is_fr else "Composite certification", use_container_width=True)

    # Détail par cible
    for tkey, tinfo in dom['cibles'].items():
        t_nom = tinfo['fr'] if is_fr else tinfo['en']
        cdata = rapport.get('certify', {}).get(tkey, {})

        if cdata:
            grade = cdata.get('grade', '?')
            score = cdata.get('certification_score', 0)
            perf = cdata.get('performance_score', 0)
            stab = cdata.get('stability_score', 0)
            modele = cdata.get('model', '?').replace('_', ' ')

            grade_color = '#27ae60' if grade == 'A' else '#f39c12' if grade == 'B' else '#c0392b'
            grade_desc_fr = f"Excellent — le modèle est fiable, stable et équitable entre zones." if grade == 'A' else f"Bon — quelques réserves sur la stabilité ou l'équité." if grade == 'B' else f"Acceptable — des améliorations sont possibles."
            grade_desc_en = f"Excellent — the model is reliable, stable and equitable across zones." if grade == 'A' else f"Good — some reservations on stability or equity." if grade == 'B' else f"Acceptable — improvements are possible."

            st.markdown(f"""
            <div class="pec-card card-certify">
                <h3 style="color:{grade_color}">{t_nom}</h3>
                <p style="font-size:2.5rem; font-weight:800; color:{grade_color}; text-align:center">
                    Grade {grade} — {score:.1f}/100
                </p>
                <p>
                    <b>{'Modèle' if is_fr else 'Model'}:</b> {modele} | 
                    <b>{'Performance' if is_fr else 'Performance'}:</strong> {perf:.1f} | 
                    <b>{'Stabilité' if is_fr else 'Stability'}:</b> {stab:.1f}
                </p>
                <p style="color:{grade_color}">{grade_desc_fr if is_fr else grade_desc_en}</p>
            </div>
            """, unsafe_allow_html=True)

    # Interprétation CERTIFY
    interp_key = 'interp_certify_fr' if is_fr else 'interp_certify_en'
    if interp_key in dom:
        st.markdown(f"""
        <div class="interp-box">
            <strong><i class="fa-solid fa-lightbulb"></i> {'Interprétation' if is_fr else 'Interpretation'} :</strong><br>
            {dom[interp_key]}
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# À PROPOS
# ============================================================================
with tabs[4]:
    st.markdown(f"<h1><i class='fa-solid fa-circle-info'></i> {'À propos' if is_fr else 'About'}</h1>", unsafe_allow_html=True)
    if is_fr:
        st.markdown('''<div class="interp-box">
        <strong>PEC Généralisé</strong> — Plateforme de certification multi-domaines.<br><br>
        Cette plateforme démontre la <strong>généralisation</strong> du framework PEC (Predict-Explain-Certify) sur <strong>3 domaines</strong> et <strong>9 cibles</strong> :
        <ul>
            <li><strong>Climat — Asie du Sud-Est</strong> : Dégâts inondations, Indice de risque, Population à risque</li>
            <li><strong>Fintech — Afrique de l'Ouest</strong> : Pénétration numérique, Ratio NPL, Taux de défaut</li>
            <li><strong>Santé — Afrique de l'Est</strong> : Accouchements qualifiés, Mortalité infantile, Vaccination</li>
        </ul>
        Pour chaque cible, le système <strong>PREDICT</strong> (3 modèles), <strong>EXPLAIN</strong> (SHAP) puis <strong>CERTIFY</strong> (score composite + Grade A-D).
        </div>''', unsafe_allow_html=True)
        st.markdown('<h3><i class="fa-solid fa-list-check"></i> Conditions d\'application</h3>', unsafe_allow_html=True)
        st.markdown('''<div class="pec-card"><ul style="line-height:2;">
        <li>Données tabulaires (CSV) avec au moins 20 observations et 3 variables numériques</li>
        <li>Variables cibles numériques (régression) ou catégorielles (classification)</li>
        <li>Aucune valeur manquante critique (>50% par colonne)</li>
        <li>Relations non purement aléatoires entre variables explicatives et cible</li>
        </ul></div>''', unsafe_allow_html=True)
    else:
        st.markdown('''<div class="interp-box">
        <strong>PEC Generalized</strong> — Multi-domain certification platform.<br><br>
        This platform demonstrates the <strong>generalization</strong> of the PEC (Predict-Explain-Certify) framework across <strong>3 domains</strong> and <strong>9 targets</strong>:
        <ul>
            <li><strong>Climate — Southeast Asia</strong>: Flood damage, Risk index, Population at risk</li>
            <li><strong>Fintech — West Africa</strong>: Digital penetration, NPL ratio, Default rate</li>
            <li><strong>Health — East Africa</strong>: Qualified deliveries, Infant mortality, Vaccination</li>
        </ul>
        For each target, the system <strong>PREDICTS</strong> (3 models), <strong>EXPLAINS</strong> (SHAP) then <strong>CERTIFIES</strong> (composite score + Grade A-D).
        </div>''', unsafe_allow_html=True)
        st.markdown('<h3><i class="fa-solid fa-list-check"></i> Application conditions</h3>', unsafe_allow_html=True)
        st.markdown('''<div class="pec-card"><ul style="line-height:2;">
        <li>Tabular data (CSV) with at least 20 observations and 3 numeric variables</li>
        <li>Numeric targets (regression) or categorical targets (classification)</li>
        <li>No critical missing values (>50% per column)</li>
        <li>Non-random relationships between explanatory variables and target</li>
        </ul></div>''', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown('---')
st.markdown("""<div style="text-align:center;color:#6b7280;font-size:.85rem;padding:1rem 0;">
    PEC Framework v4 | <i class="fa-solid fa-flask"></i> Predict • <i class="fa-solid fa-magnifying-glass"></i> Explain • <i class="fa-solid fa-shield-halved"></i> Certify |
    <a href="https://orcid.org/0009-0003-3048-1765" target="_blank" style="color:#a6ce39;"><i class="fa-brands fa-orcid"></i> Rosa Elysabeth Ralinirina</a>
</div>""", unsafe_allow_html=True)