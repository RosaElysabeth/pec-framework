# ============================================================================
# PEC FRAMEWORK — What-Ifgascar
# Auteur : Rosa Elysabeth Ralinirina | ORCID : 0009-0003-3048-1765
# ============================================================================
# PREDICT — EXPLAIN — CERTIFY : cas d'�tude Madagascar
# ============================================================================

import streamlit as st

def pec_url(app):
    """URL dynamique: localhost en local, secrets en cloud."""
    local = {'mada': 'http://localhost:8501', 'aide': 'http://localhost:8502',
             'whatif': 'http://localhost:8503', 'gen': 'http://localhost:8504'}
    try:
        # Essai 1: st.secrets['urls']['mada'] (avec section [urls])
        try:
            cloud = {'mada': st.secrets['urls']['mada'], 'aide': st.secrets['urls']['aide'],
                     'whatif': st.secrets['urls']['whatif'], 'gen': st.secrets['urls']['gen']}
        except Exception:
            # Essai 2: st.secrets['mada'] (sans section)
            cloud = {'mada': st.secrets['mada'], 'aide': st.secrets['aide'],
                     'whatif': st.secrets['whatif'], 'gen': st.secrets['gen']}
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

st.set_page_config(page_title="PEC What-If | Madagascar", page_icon=None, layout="wide", initial_sidebar_state="expanded")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data") if os.path.exists(os.path.join(BASE, "data")) else os.path.join(BASE, "..", "data")
MODELS_DIR = os.path.join(BASE, "models") if os.path.exists(os.path.join(BASE, "models")) else os.path.join(BASE, "..", "models")
RESULTS_DIR = os.path.join(BASE, "results") if os.path.exists(os.path.join(BASE, "results")) else os.path.join(BASE, "..", "results")
class _SklearnUnpickler(pickle.Unpickler):
    """Unpickler qui mappe les modules internes sklearn (ex: _loss -> sklearn._loss)."""
    def find_class(self, module, name):
        if module.startswith('_') and not module.startswith('sklearn'):
            # Mapper _loss -> sklearn._loss, _tree -> sklearn.tree._tree, etc.
            mapped = 'sklearn.' + module
            try:
                __import__(mapped)
            except ImportError:
                pass
            module = mapped
        return super().find_class(module, name)

def _load(p):
    """Charge un .pkl — Unpickler sklearn + fallback joblib."""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*InconsistentVersion.*')
    try:
        with open(p, 'rb') as f:
            return _SklearnUnpickler(f).load()
    except Exception:
        try:
            return joblib.load(p)
        except Exception:
            # Dernier recours: forcer l'import des modules sklearn
            import sklearn._loss, sklearn.tree._tree
            with open(p, 'rb') as f:
                return pickle.load(f)



# ============================================================================
# FONT AWESOME + CSS
# ============================================================================
st.markdown("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">""", unsafe_allow_html=True)
st.markdown("""<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">""", unsafe_allow_html=True)

_css_path = os.path.join(BASE, 'shared_pec.css')
if os.path.exists(_css_path):
    with open(_css_path, 'r', encoding='utf-8') as _f: st.markdown(f'<style>{_f.read()}</style>', unsafe_allow_html=True)
else:
    st.markdown("""<style>
:root {
  --primary: #1a5276; --primary-light: #2e86c1; --secondary: #27ae60;
  --purple: #8e44ad; --red: #c0392b; --orange: #f39c12;
  --bg: #f4f6f9; --card: #fff; --text: #1a1a2e; --muted: #6b7280;
  --border: #e5e7eb; --radius: 12px; --shadow: 0 4px 24px rgba(0,0,0,.06);
}
</style>""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_resource
def load_all():
    meta = _load(os.path.join(MODELS_DIR, 'meta.pkl'))
    scaler = _load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    feature_cols = _load(os.path.join(MODELS_DIR, 'feature_cols.pkl'))
    le = _load(os.path.join(MODELS_DIR, 'label_encoder_region.pkl'))
    stats = _load(os.path.join(MODELS_DIR, 'feature_stats.pkl'))
    models = {}
    for mn in meta['models']:
        for target in meta['targets']:
            path = os.path.join(MODELS_DIR, f"{mn}_{target}.pkl")
            if os.path.exists(path):
                models[(mn, target)] = _load(path)
    return meta, scaler, feature_cols, le, stats, models

@st.cache_data
def load_ref(): return pd.read_csv(os.path.join(DATA_DIR, "dataset_pec_v3.csv"))

@st.cache_data
def load_report():
    with open(os.path.join(RESULTS_DIR, 'rapport_final.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

meta, scaler, feature_cols, le, feature_stats, models = load_all()
df_ref = load_ref()
rp = load_report()

RTK = {'Insuffisance_ponderale':'Insuffisance Pond\u00e9rale',
       'Malnutrition_chronique':'Malnutrition Chronique',
       'Malnutrition_aigue':'Malnutrition Aig\u00fce'}

TARGETS = {
    'Insuffisance_ponderale': {'short':'IP','fr':'Insuffisance Pond\u00e9rale','en':'Underweight','color':'#2e86c1','unit':'%','icon':'fa-weight-scale'},
    'Malnutrition_chronique': {'short':'MC','fr':'Malnutrition Chronique','en':'Chronic Malnutrition','color':'#8e44ad','unit':'%','icon':'fa-ruler-vertical'},
    'Malnutrition_aigue': {'short':'MA','fr':'Malnutrition Aig\u00fce','en':'Acute Malnutrition','color':'#c0392b','unit':'%','icon':'fa-triangle-exclamation'},
}

FRIENDLY_FEATURES = {
    'fr': {
        'Pauvrete_regionale_pct': 'Pauvret\u00e9 r\u00e9gionale (%)',
        'Precipitation_annuelle_mm': 'Pr\u00e9cipitations annuelles (mm)',
        'Temp_max_moy': 'Temp\u00e9rature maximale (\u00b0C)',
        'Riz_par_habitant_kg': 'Riz par habitant (kg/an)',
        'Croissance_PIB_pct': 'Croissance du PIB (%)',
        'Acces_eau_potable_pct': 'Acc\u00e8s \u00e0 l\u2019eau potable (%)',
        'Couverture_sanitaire_CSB2_pct': 'Couverture sanitaire CSB2 (%)',
        'Annee': 'Ann\u00e9e',
        'Region_enc': 'R\u00e9gion (encod\u00e9e)',
        'ma3_Insuffisance_ponderale': 'Tendance 3 ans (IP)',
        'ma3_Malnutrition_chronique': 'Tendance 3 ans (MC)',
        'ma3_Malnutrition_aigue': 'Tendance 3 ans (MA)',
        'lag1_Insuffisance_ponderale': 'Valeur ann\u00e9e pr\u00e9c\u00e9dente (IP)',
        'lag1_Malnutrition_chronique': 'Valeur ann\u00e9e pr\u00e9c\u00e9dente (MC)',
        'lag1_Malnutrition_aigue': 'Valeur ann\u00e9e pr\u00e9c\u00e9dente (MA)',
    },
    'en': {
        'Pauvrete_regionale_pct': 'Regional poverty (%)',
        'Precipitation_annuelle_mm': 'Annual precipitation (mm)',
        'Temp_max_moy': 'Max temperature (\u00b0C)',
        'Riz_par_habitant_kg': 'Rice per capita (kg/yr)',
        'Croissance_PIB_pct': 'GDP growth (%)',
        'Acces_eau_potable_pct': 'Safe water access (%)',
        'Couverture_sanitaire_CSB2_pct': 'Health coverage CSB2 (%)',
        'Annee': 'Year',
        'Region_enc': 'Region (encoded)',
    }
}

REGIONS = sorted(df_ref['Region'].unique().tolist())

# ============================================================================
# TRANSLATIONS
# ============================================================================
T = {
'fr': {
    'app_title':'What-If PEC — Madagascar','app_subtitle':'Pr\u00e9dire \u2022 Expliquer \u2022 Certifier',
    'tab_overview':'Vue d\u2019ensemble','tab_predict':'Pr\u00e9dire','tab_explain':'Expliquer','tab_certify':'Certifier','tab_scenarios':'Sc\u00e9narios','tab_about':'\u00c0 propos','nav_mada':'PEC Mada','nav_aide':'PEC Aide','nav_what':'PEC What','nav_gen':'PEC Gen',
    'target_label':'Cible nutritionnelle','region_label':'R\u00e9gion','year_label':'Ann\u00e9e de pr\u00e9diction',
    'btn_predict':'Lancer la pr\u00e9diction','btn_scenario':'G\u00e9n\u00e9rer les sc\u00e9narios',
    'prediction':'Pr\u00e9diction','predicted_value':'Valeur pr\u00e9dite','national_avg':'Moyenne nationale','variation':'Variation',
    'higher':'sup\u00e9rieur','lower':'inf\u00e9rieur','than_avg':'\u00e0 la moyenne nationale',
    'interpretation':'Interpr\u00e9tation',
    'explain_title':'Pourquoi cette pr\u00e9diction ?','explain_sub':'Facteurs les plus influents (SHAP)',
    'explain_note':'Plus la barre est longue, plus le facteur influence la pr\u00e9diction. Les valeurs positives augmentent la cible, les n\u00e9gatives la diminuent.',
    'certify_title':'Peut-on faire confiance \u00e0 cette pr\u00e9diction ?',
    'cert_score':'Score de confiance','cert_r2':'Performance (R\u00b2)','cert_grade':'Grade','cert_fairness':'\u00c9quit\u00e9 r\u00e9gionale','cert_stability':'Stabilit\u00e9',
    'cert_excellent':'Excellent \u2014 Syst\u00e8me certifi\u00e9, pr\u00eat pour le d\u00e9ploiement op\u00e9rationnel.',
    'cert_good':'Bon \u2014 Syst\u00e8me fiable, am\u00e9liorations mineures recommand\u00e9es.',
    'cert_warning':'Attention \u2014 Risques identifi\u00e9s, audit approfondi n\u00e9cessaire.',
    'cert_critical':'Critique \u2014 Syst\u00e8me non certifi\u00e9, restructuration n\u00e9cessaire.',
    'scenario':'Sc\u00e9nario','scenario_desc':'Simulation de sc\u00e9narios futurs pour 2026\u20132035.',
    'statuquo':'Statu quo (tendances actuelles)','amelioration':'Am\u00e9lioration climatique','aggravation':'Aggravation climatique',
    'reduction_pauvrete':'R\u00e9duction de la pauvret\u00e9','augmentation_pauvrete':'Augmentation de la pauvret\u00e9',
    'amelioration_sanitaire':'Am\u00e9lioration sanitaire','favorable':'Sc\u00e9nario favorable','defavorable':'Sc\u00e9nario d\u00e9favorable',
    'all_regions':'Toutes les r\u00e9gions (moyenne)','historique':'historique','scenario_label':'sc\u00e9nario',
    'download_csv':'T\u00e9l\u00e9charger CSV','no_data':'Aucune donn\u00e9e pour cette r\u00e9gion.',
    'best_model':'Meilleur mod\u00e8le','model_label':'Mod\u00e8le','overview_title':'Vue d\u2019ensemble PEC',
    'overview_desc':'Le framework PEC (Predict-Explain-Certify) appliqu\u00e9 \u00e0 la malnutrition \u00e0 Madagascar.',
    'key_vars':'Variables cl\u00e9s influentes','shap_interp':'Interpr\u00e9tation SHAP','method_comp':'Accord inter-m\u00e9thodes',
    'ref_val':'Valeur de r\u00e9f\u00e9rence',
},
'en': {
    'app_title':'What-If PEC — Madagascar','app_subtitle':'Predict \u2022 Explain \u2022 Certify',
    'tab_overview':'Overview','tab_predict':'Predict','tab_explain':'Explain','tab_certify':'Certify','tab_scenarios':'Scenarios','tab_about':'About','nav_mada':'PEC Mada','nav_aide':'PEC Aid','nav_what':'PEC What','nav_gen':'PEC Gen',
    'target_label':'Nutritional target','region_label':'Region','year_label':'Prediction year',
    'btn_predict':'Run prediction','btn_scenario':'Generate scenarios',
    'prediction':'Prediction','predicted_value':'Predicted value','national_avg':'National average','variation':'Variation',
    'higher':'higher','lower':'lower','than_avg':'than national average',
    'interpretation':'Interpretation',
    'explain_title':'Why this prediction?','explain_sub':'Most influential factors (SHAP)',
    'explain_note':'The longer the bar, the more influential the factor. Positive values increase the target, negative values decrease it.',
    'certify_title':'Can we trust this prediction?',
    'cert_score':'Trust score','cert_r2':'Performance (R\u00b2)','cert_grade':'Grade','cert_fairness':'Regional fairness','cert_stability':'Stability',
    'cert_excellent':'Excellent \u2014 Certified system, ready for operational deployment.',
    'cert_good':'Good \u2014 Reliable system, minor improvements recommended.',
    'cert_warning':'Warning \u2014 Risks identified, deeper audit needed.',
    'cert_critical':'Critical \u2014 Uncertified system, restructuring needed.',
    'scenario':'Scenario','scenario_desc':'Simulation of future scenarios for 2026\u20132035.',
    'statuquo':'Status quo (current trends)','amelioration':'Climate improvement','aggravation':'Climate degradation',
    'reduction_pauvrete':'Poverty reduction','augmentation_pauvrete':'Poverty increase',
    'amelioration_sanitaire':'Health improvement','favorable':'Combined favorable','defavorable':'Combined unfavorable',
    'all_regions':'All regions (average)','historique':'historical','scenario_label':'scenario',
    'download_csv':'Download CSV','no_data':'No data for this region.',
    'best_model':'Best model','model_label':'Model','overview_title':'PEC Overview',
    'overview_desc':'The PEC (Predict-Explain-Certify) framework applied to malnutrition in Madagascar.',
    'key_vars':'Key influential variables','shap_interp':'SHAP Interpretation','method_comp':'Inter-method agreement',
    'ref_val':'Reference value',
}}

def t(key): return T.get(st.session_state.lang, T['fr']).get(key, key)

def friendly_feat(feat, lang='fr'):
    d = FRIENDLY_FEATURES.get(lang, {})
    return d.get(feat, feat.replace('_',' ').replace(' pct','%').replace(' mm',' mm'))

def grade_info(score):
    if score >= 90: return 'A', 'badge-a', t('cert_excellent')
    elif score >= 80: return 'B', 'badge-b', t('cert_good')
    elif score >= 60: return 'C', 'badge-c', t('cert_warning')
    else: return 'D', 'badge-d', t('cert_critical')

def predict_one(target_key, region, year, model_name='Ridge'):
    model = models.get((model_name, target_key))
    if model is None: return None
    ref_rows = df_ref[df_ref['Region'] == region]
    ref_row = ref_rows.iloc[-1].copy() if not ref_rows.empty else df_ref.iloc[0].copy()
    row = ref_row.copy()
    row['Annee'] = year
    if 'Region_enc' in feature_cols and region in le.classes_:
        row['Region_enc'] = le.transform([region])[0]
    X = pd.DataFrame([[row.get(c, feature_stats.get(c, {}).get('mean', 0)) for c in feature_cols]], columns=feature_cols)
    for c in feature_cols:
        if X[c].isnull().any(): X[c] = X[c].fillna(feature_stats.get(c, {}).get('mean', X[c].mean()))
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)
    return float(model.predict(X_scaled)[0])

def get_shap_top(target_key, n=8):
    rk = RTK.get(target_key, target_key)
    ed = rp.get('explain', {}).get(rk, {})
    top5 = ed.get('top5_SHAP', {})
    if isinstance(top5, dict):
        items = list(top5.items())[:n]
        return [(friendly_feat(f, st.session_state.lang), v) for f, v in items]
    return []

def get_cert_detail(target_key):
    rk = RTK.get(target_key, target_key)
    cd = rp.get('certify', {}).get(rk, {})
    pd_data = rp.get('predict', {}).get(rk, {})
    return {
        'score': cd.get('certification_score', 0),
        'r2': pd_data.get('R2_test', 0),
        'best': pd_data.get('meilleur_modele', pd_data.get('model', 'N/A')),
        'fairness': cd.get('fairness', cd.get('equite', 0)),
        'stability': cd.get('stability', cd.get('stabilite', 0)),
        'ip_score': cd.get('ip_score', cd.get('IP', 0)),
        'mc_score': cd.get('mc_score', cd.get('MC', 0)),
        'ma_score': cd.get('ma_score', cd.get('MA', 0)),
    }

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    # Language toggle
    lang = st.radio('', ['fr','en'], format_func=lambda x: 'Français' if x=='fr' else 'English', horizontal=True, label_visibility='collapsed', key='sidebar_lang')
    st.session_state.lang = lang

    # Navigation inter-plateformes
    st.divider()
    st.markdown('#### 🔗 Plateformes PEC')
    c1, c2 = st.columns(2)
    with c1: st.link_button(f'📊 {t("nav_mada")}', pec_url('mada'), use_container_width=True)
    with c2: st.link_button(f'📊 {t("nav_aide")}', pec_url('aide'), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3: st.link_button(f'🔬 {t("nav_what")}', pec_url('whatif'), use_container_width=True)
    with c4: st.link_button(f'🌍 {t("nav_gen")}', pec_url('gen'), use_container_width=True)
    st.markdown(f"### <i class='fa-solid fa-flask'></i> PEC What-If", unsafe_allow_html=True)
    st.caption(t('app_subtitle'))
    st.divider()

    # Target selection
    target_key = st.selectbox(
        t('target_label'),
        list(TARGETS.keys()),
        format_func=lambda x: f"{TARGETS[x]['short']} — {TARGETS[x][st.session_state.lang]}",
        key='sidebar_target'
    )

    # Region
    region = st.selectbox(
        t('region_label'),
        REGIONS,
        key='sidebar_region'
    )

    # Year
    year = st.slider(
        t('year_label'),
        2026, 2035, 2026,
        key='sidebar_year'
    )

    # Model
    model_choice = st.selectbox(
        t('model_label'),
        meta['models'] if meta else [],
        index=meta['models'].index('Ridge') if meta else 0,
        key='sidebar_model'
    )

    st.divider()
    # Summary
    info = TARGETS[target_key]
    rk = RTK.get(target_key, target_key)
    cd = rp.get('certify', {}).get(rk, {})
    pd_i = rp.get('predict', {}).get(rk, {})
    cert = cd.get('certification_score', 0)
    r2 = pd_i.get('R2_test', 0)
    grade, _, _ = grade_info(cert)
    st.markdown(f"""
    <div style="background:rgba(255,255,255,.1);border-radius:10px;padding:.8rem;margin-bottom:.5rem;">
        <div style="font-weight:700;font-size:.95rem;">{info['short']} — {info[st.session_state.lang]}</div>
        <div style="font-size:.85rem;color:rgba(255,255,255,.8);">R\u00b2 = {r2:.4f} | <span class="badge {grade.lower()}" style="font-size:.8rem;">{grade}</span></div>
        <div style="font-size:.8rem;color:rgba(255,255,255,.7);">{t('best_model')}: {pd_i.get('meilleur_modele','N/A')}</div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""<div style="text-align:center;color:rgba(255,255,255,.7);font-size:.8rem;">
        Rosa Elysabeth Ralinirina<br>
        <a href="https://orcid.org/0009-0003-3048-1765" target="_blank" style="color:#a6ce39;">
            <i class="fa-brands fa-orcid"></i> 0009-0003-3048-1765
        </a>
    </div>""", unsafe_allow_html=True)

# ============================================================================
# PEC TABS
# ============================================================================
tab_overview, tab_predict, tab_explain, tab_certify, tab_scenarios, tab_about = st.tabs([
    t('tab_overview'),
    t('tab_predict'),
    t('tab_explain'),
    t('tab_certify'),
    t('tab_scenarios'),
    t('tab_about'),
])

# ============================================================================
# OVERVIEW
# ============================================================================
with tab_overview:
    lang = st.session_state.lang
    info = TARGETS[target_key]
    color = info['color']

    st.markdown(f'<h1><i class="fa-solid fa-chart-pie"></i> {t("overview_title")}</h1>', unsafe_allow_html=True)
    st.caption(t('overview_desc'))

    # Overview cards for all 3 targets
    cols = st.columns(3)
    for i, (tk, info_t) in enumerate(TARGETS.items()):
        with cols[i]:
            rk_t = RTK.get(tk, tk)
            pd_t = rp.get('predict', {}).get(rk_t, {})
            cd_t = rp.get('certify', {}).get(rk_t, {})
            r2_t = pd_t.get('R2_test', 0)
            cert_t = cd_t.get('certification_score', 0)
            best_t = pd_t.get('meilleur_modele', 'N/A')
            grade_t, badge_t, _ = grade_info(cert_t)
            st.markdown(f"""
            <div class="pec-card card-{'predict' if i==0 else 'explain' if i==1 else 'certify'}">
                <h4 style="color:{info_t['color']};margin:0;"><i class="fa-solid {info_t['icon']}"></i> {info_t[lang]}</h4>
                <div style="font-size:.9rem;color:var(--muted);">{info_t['short']} | {best_t}</div>
                <div style="margin-top:.8rem;">
                    <div class="metric-val" style="color:{info_t['color']};">{cert_t:.1f}<span style="font-size:1rem;">/100</span></div>
                    <div class="metric-lbl">{t('cert_score')}</div>
                </div>
                <div style="margin-top:.5rem;">
                    <div style="font-size:1.1rem;font-weight:600;">R\u00b2 = {r2_t:.4f}</div>
                    <span class="{badge_t}">Grade {grade_t}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # Interpretation box with dynamic data
    st.markdown(f'<h3><i class="fa-solid fa-lightbulb"></i> {t("interpretation")}</h3>', unsafe_allow_html=True)

    mean_vals = {tk: df_ref[tk].mean() for tk in TARGETS}
    worst_best = {}
    for tk in TARGETS:
        rs = df_ref.groupby('Region')[tk].mean().sort_values(ascending=False)
        worst_best[tk] = (rs.index[0], rs.iloc[0], rs.index[-1], rs.iloc[-1])

    if lang == 'fr':
        interp = f"Le framework PEC analyse <strong>{len(TARGETS)} cibles nutritionnelles</strong> dans <strong>{len(REGIONS)} r\u00e9gions</strong> de Madagascar. "
        interp += f"Le mod\u00e8le le plus performant est <strong>Ridge</strong> avec des scores de certification allant de <strong>94.5</strong> \u00e0 <strong>98.0</strong> sur 100. "
        for tk, info_t in TARGETS.items():
            w_reg, w_val, b_reg, b_val = worst_best[tk]
            interp += f"{info_t['fr']} : moyenne {mean_vals[tk]:.1f}%, r\u00e9gion la plus touch\u00e9e <strong>{w_reg}</strong> ({w_val:.1f}%), moins touch\u00e9e <strong>{b_reg}</strong> ({b_val:.1f}%). "
    else:
        interp = f"The PEC framework analyzes <strong>{len(TARGETS)} nutritional targets</strong> across <strong>{len(REGIONS)} regions</strong> of Madagascar. "
        interp += f"The best performing model is <strong>Ridge</strong> with certification scores ranging from <strong>94.5</strong> to <strong>98.0</strong> out of 100. "
        for tk, info_t in TARGETS.items():
            w_reg, w_val, b_reg, b_val = worst_best[tk]
            interp += f"{info_t['en']}: mean {mean_vals[tk]:.1f}%, most affected <strong>{w_reg}</strong> ({w_val:.1f}%), least affected <strong>{b_reg}</strong> ({b_val:.1f}%). "

    st.markdown(f'<div class="interp-box">{interp}</div>', unsafe_allow_html=True)

    # Historical evolution
    st.markdown(f'<h3><i class="fa-solid fa-timeline"></i> {"\u00c9volution historique" if lang=="fr" else "Historical evolution"}</h3>', unsafe_allow_html=True)
    fig_ov = go.Figure()
    for tk, info_t in TARGETS.items():
        mean_by_year = df_ref.groupby('Annee')[tk].mean()
        fig_ov.add_trace(go.Scatter(x=mean_by_year.index, y=mean_by_year.values, mode='lines+markers',
                                    name=f"{info_t[lang]} ({info_t['short']})",
                                    line=dict(color=info_t['color'], width=2.5), marker=dict(size=6)))
    fig_ov.update_layout(height=400, xaxis_title=lang=='fr' and 'Ann\u00e9e' or 'Year',
                         yaxis_title='%', hovermode='x unified',
                         legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    st.plotly_chart(fig_ov, use_container_width=True)

# ============================================================================
# PREDICT
# ============================================================================
with tab_predict:
    lang = st.session_state.lang
    info = TARGETS[target_key]
    color = info['color']
    rk = RTK.get(target_key, target_key)

    st.markdown(f'<h1><i class="fa-solid fa-bullseye"></i> {t("prediction")} — {info[lang]}</h1>', unsafe_allow_html=True)

    if st.button(t('btn_predict'), type='primary', use_container_width=True, key='predict_btn'):
        st.session_state.pred_done = True

    if st.session_state.get('pred_done', False):
        pred_val = predict_one(target_key, region, year, model_choice)
        if pred_val is not None:
            nat_avg = df_ref[target_key].mean()
            diff = pred_val - nat_avg
            direction = t('higher') if diff > 0 else t('lower')

            # Prediction card
            st.markdown(f"""
            <div class="pec-card card-predict">
                <h3 style="color:{color};"><i class="fa-solid {info['icon']}"></i> {info[lang]} — {region} ({year})</h3>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:1rem;">
                    <div style="text-align:center;padding:1rem;border-radius:var(--radius);background:#f0f7ff;">
                        <div class="metric-val" style="color:{color};">{pred_val:.1f}{info['unit']}</div>
                        <div class="metric-lbl">{t('predicted_value')}</div>
                    </div>
                    <div style="text-align:center;padding:1rem;border-radius:var(--radius);background:#f0f0f0;">
                        <div class="metric-val" style="color:var(--muted);">{nat_avg:.1f}{info['unit']}</div>
                        <div class="metric-lbl">{t('national_avg')}</div>
                    </div>
                    <div style="text-align:center;padding:1rem;border-radius:var(--radius);background:{'#fff0f0' if diff>0 else '#f0fff0'};">
                        <div class="metric-val" style="color:{'#e74c3c' if diff>0 else '#27ae60'};">{diff:+.1f}{info['unit']}</div>
                        <div class="metric-lbl">{direction} {t('than_avg')}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Key influential variables (from SHAP)
            top_feats = get_shap_top(target_key, n=8)
            if top_feats:
                st.markdown(f'<h4><i class="fa-solid fa-ranking-star"></i> {t("key_vars")}</h4>', unsafe_allow_html=True)
                feat_names = [f[0] for f in top_feats]
                feat_vals = [abs(f[1]) for f in top_feats]
                colors_bar = ['#27ae60' if f[1] > 0 else '#e74c3c' for f in top_feats]
                fig_f = go.Figure(go.Bar(x=feat_vals, y=feat_names, orientation='h',
                                         marker_color=colors_bar,
                                         text=[f'{v:+.4f}' for _, v in top_feats], textposition='outside'))
                fig_f.update_layout(height=350, showlegend=False,
                                    xaxis_title='SHAP value',
                                    margin=dict(l=20, r=60, t=20, b=20))
                st.plotly_chart(fig_f, use_container_width=True)

            # Dynamic interpretation based on real data
            region_mean = df_ref[df_ref['Region'] == region][target_key].mean()
            region_rank = df_ref.groupby('Region')[target_key].mean().sort_values(ascending=False)
            rank = list(region_rank.index).index(region) + 1 if region in region_rank.index else '?'

            if lang == 'fr':
                interp = f"Pour <strong>{region}</strong> en <strong>{year}</strong>, la pr\u00e9diction est <strong>{pred_val:.1f}%</strong> "
                interp += f"({diff:+.1f} pts par rapport \u00e0 la moyenne nationale de {nat_avg:.1f}%). "
                interp += f"{region} se classe <strong>{rank}\u00e8me</strong> sur {len(REGIONS)} r\u00e9gions. "
                if top_feats:
                    top3 = top_feats[:3]
                    interp += f"Les variables les plus influentes sont : "
                    interp += ", ".join([f"<strong>{n}</strong> ({'+' if v>0 else '\u2212'}{abs(v):.4f})" for n, v in top3])
                    interp += "."
            else:
                interp = f"For <strong>{region}</strong> in <strong>{year}</strong>, the prediction is <strong>{pred_val:.1f}%</strong> "
                interp += f"({diff:+.1f} pts vs national average of {nat_avg:.1f}%). "
                interp += f"{region} ranks <strong>{rank}</strong> out of {len(REGIONS)} regions. "
                if top_feats:
                    top3 = top_feats[:3]
                    interp += f"Most influential variables: "
                    interp += ", ".join([f"<strong>{n}</strong> ({'+' if v>0 else '\u2212'}{abs(v):.4f})" for n, v in top3])
                    interp += "."

            st.markdown(f'<div class="interp-box"><strong><i class="fa-solid fa-lightbulb"></i> {t("interpretation")} :</strong><br>{interp}</div>', unsafe_allow_html=True)

            # Gauge
            from plotly.subplots import make_subplots
            fig_g = go.Figure(go.Indicator(mode="gauge+number", value=pred_val,
                domain={'x':[0,1],'y':[0,1]}, title={'text':f'{info[lang]} — {region} {year}','font':{'size':16}},
                number={'suffix':info['unit'],'font':{'size':36,'color':color}},
                gauge={'axis':{'range':[None,max(100,pred_val*1.5)]},'bar':{'color':color},
                       'steps':[{'range':[0,max(100,pred_val*1.5)*0.4],'color':'#fde8e8'},
                                {'range':[max(100,pred_val*1.5)*0.4,max(100,pred_val*1.5)*0.6],'color':'#fef9e7'},
                                {'range':[max(100,pred_val*1.5)*0.6,max(100,pred_val*1.5)*0.8],'color':'#d5f5e3'},
                                {'range':[max(100,pred_val*1.5)*0.8,max(100,pred_val*1.5)],'color':'#a9dfbf'}]}))
            fig_g.update_layout(height=280, margin=dict(l=20,r=20,t=50,b=10))
            st.plotly_chart(fig_g, use_container_width=True)

# ============================================================================
# EXPLAIN
# ============================================================================
with tab_explain:
    lang = st.session_state.lang
    info = TARGETS[target_key]
    color = info['color']
    rk = RTK.get(target_key, target_key)

    st.markdown(f'<h1><i class="fa-solid fa-magnifying-glass-chart"></i> {t("explain_title")}</h1>', unsafe_allow_html=True)
    st.caption(f'{t("explain_sub")} — {info[lang]}')

    # SHAP top features
    top_feats = get_shap_top(target_key, n=10)
    if top_feats:
        feat_names = [f[0] for f in top_feats]
        feat_vals = [f[1] for f in top_feats]
        colors_shap = ['#27ae60' if v > 0 else '#e74c3c' for v in feat_vals]

        fig_shap = go.Figure(go.Bar(x=feat_vals, y=feat_names, orientation='h',
                                     marker_color=colors_shap,
                                     text=[f'{v:+.4f}' for v in feat_vals], textposition='outside'))
        fig_shap.update_layout(height=400, showlegend=False,
                               xaxis_title='SHAP value',
                               margin=dict(l=20,r=60,t=20,b=20))
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown(f'<div class="interp-box"><i class="fa-solid fa-circle-info"></i> {t("explain_note")}</div>', unsafe_allow_html=True)
    else:
        st.info(t('no_data'))

    # SHAP images
    for shap_type in ['summary', 'bar']:
        shap_path = os.path.join(RESULTS_DIR, 'shap', f'shap_{shap_type}_{target_key}.png')
        if os.path.exists(shap_path):
            st.image(Image.open(shap_path), use_container_width=True)

    # XAI comparison
    xai_path = os.path.join(RESULTS_DIR, 'comparaison_xai', f'xai_comparison_{target_key}.png')
    if os.path.exists(xai_path):
        st.markdown(f'<h4><i class="fa-solid fa-code-compare"></i> {t("method_comp")}</h4>', unsafe_allow_html=True)
        st.image(Image.open(xai_path), use_container_width=True)

# ============================================================================
# CERTIFY
# ============================================================================
with tab_certify:
    lang = st.session_state.lang
    info = TARGETS[target_key]
    color = info['color']
    rk = RTK.get(target_key, target_key)

    cd = rp.get('certify', {}).get(rk, {})
    pd_data = rp.get('predict', {}).get(rk, {})
    cert = cd.get('certification_score', 0)
    r2 = pd_data.get('R2_test', 0)
    best = pd_data.get('meilleur_modele', 'N/A')
    grade, badge, desc = grade_info(cert)

    st.markdown(f'<h1><i class="fa-solid fa-shield-halved"></i> {t("certify_title")}</h1>', unsafe_allow_html=True)

    # Main certification card
    st.markdown(f"""
    <div class="pec-card card-certify">
        <h3 style="color:#27ae60;"><i class="fa-solid fa-shield-halved"></i> {info['icon_fa' if False else 'icon']} {info[lang]}</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:1rem;">
            <div style="text-align:center;padding:1rem;border-radius:var(--radius);background:#f0fff0;border-top:4px solid #27ae60;">
                <div class="metric-val" style="color:#27ae60;">{cert:.1f}<span style="font-size:1rem;">/100</span></div>
                <div class="metric-lbl">{t('cert_score')}</div>
            </div>
            <div style="text-align:center;padding:1rem;border-radius:var(--radius);background:#f0f7ff;border-top:4px solid #2e86c1;">
                <div class="metric-val" style="color:#2e86c1;">{r2:.4f}</div>
                <div class="metric-lbl">{t('cert_r2')}</div>
            </div>
            <div style="text-align:center;padding:1rem;border-radius:var(--radius);background:#f8f0ff;border-top:4px solid #8e44ad;">
                <span class="{badge}" style="font-size:2rem;">{grade}</span>
                <div class="metric-lbl">{t('cert_grade')}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="interp-box"><strong>Grade {grade} :</strong> {desc}</div>', unsafe_allow_html=True)

    # Gauge
    fig_cert = go.Figure(go.Indicator(mode="gauge+number", value=cert,
        domain={'x':[0,1],'y':[0,1]}, title={'text':f'{t("cert_score")} — {info[lang]}','font':{'size':16}},
        number={'suffix':'/100','font':{'size':36,'color':'#27ae60'}},
        gauge={'axis':{'range':[None,100]},'bar':{'color':'#27ae60'},
               'steps':[{'range':[0,60],'color':'#fde8e8'},{'range':[60,80],'color':'#fef9e7'},
                        {'range':[80,90],'color':'#d5f5e3'},{'range':[90,100],'color':'#a9dfbf'}]}))
    fig_cert.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=10))
    st.plotly_chart(fig_cert, use_container_width=True)

    # All 3 targets comparison
    st.markdown(f'<h3><i class="fa-solid fa-scale-balanced"></i> {"Certification compar\u00e9e" if lang=="fr" else "Comparative certification"}</h3>', unsafe_allow_html=True)
    cert_data = []
    for tk2, info2 in TARGETS.items():
        rk2 = RTK.get(tk2, tk2)
        cd2 = rp.get('certify', {}).get(rk2, {})
        pd2 = rp.get('predict', {}).get(rk2, {})
        c2 = cd2.get('certification_score', 0)
        r22 = pd2.get('R2_test', 0)
        g2, _, _ = grade_info(c2)
        cert_data.append({'Cible': info2[lang], 'Score': c2, 'R\u00b2': r22, 'Mod\u00e8le': pd2.get('meilleur_modele','N/A'), 'Grade': g2})

    fig_all = go.Figure()
    for row in cert_data:
        clr = '#27ae60' if row['Grade']=='A' else '#f39c12' if row['Grade']=='B' else '#e74c3c'
        fig_all.add_trace(go.Bar(x=[row['Cible']], y=[row['Score']], marker_color=clr,
                                 text=f"{row['Score']:.1f}/100 ({row['Grade']})", textposition='outside', name=row['Cible']))
    fig_all.add_hline(y=90, line_dash='dash', line_color='#27ae60', annotation_text='Grade A')
    fig_all.add_hline(y=80, line_dash='dash', line_color='#f39c12', annotation_text='Grade B')
    fig_all.update_layout(height=350, showlegend=False, yaxis_title=t('cert_score'))
    st.plotly_chart(fig_all, use_container_width=True)
    st.dataframe(pd.DataFrame(cert_data), use_container_width=True, hide_index=True)

    # Certification image
    cert_img = os.path.join(RESULTS_DIR, 'certification', 'certification_composite.png')
    if os.path.exists(cert_img):
        st.image(Image.open(cert_img), use_container_width=True)

# ============================================================================
def _apply_scenario(row, si, ref_last):
    """Apply scenario modifier to a row dict. Returns modified row."""
    r = row.copy()
    if si == 1:  # Climate improvement
        if pd.notna(ref_last.get('Precipitation_annuelle_mm')):
            r['Precipitation_annuelle_mm'] = ref_last['Precipitation_annuelle_mm'] * 1.10
            if 'Precip_saison_cyclonique_mm' in r: r['Precip_saison_cyclonique_mm'] = ref_last.get('Precip_saison_cyclonique_mm', ref_last['Precipitation_annuelle_mm']*0.6) * 1.10
            if 'Precip_saison_seche_mm' in r: r['Precip_saison_seche_mm'] = ref_last.get('Precip_saison_seche_mm', ref_last['Precipitation_annuelle_mm']*0.15) * 1.10
        if pd.notna(ref_last.get('Temp_max_moy')):
            r['Temp_max_moy'] = ref_last['Temp_max_moy'] - 0.5
            if 'Temp_min_moy' in r: r['Temp_min_moy'] = ref_last.get('Temp_min_moy', ref_last['Temp_max_moy']-7) - 0.3
        if 'Humidite_max_moy' in r: r['Humidite_max_moy'] = min(100, ref_last.get('Humidite_max_moy', 80) + 2)
    elif si == 2:  # Climate degradation
        if pd.notna(ref_last.get('Precipitation_annuelle_mm')):
            r['Precipitation_annuelle_mm'] = ref_last['Precipitation_annuelle_mm'] * 0.80
            if 'Precip_saison_cyclonique_mm' in r: r['Precip_saison_cyclonique_mm'] = ref_last.get('Precip_saison_cyclonique_mm', ref_last['Precipitation_annuelle_mm']*0.6) * 0.80
            if 'Precip_saison_seche_mm' in r: r['Precip_saison_seche_mm'] = ref_last.get('Precip_saison_seche_mm', ref_last['Precipitation_annuelle_mm']*0.15) * 0.80
        if pd.notna(ref_last.get('Temp_max_moy')):
            r['Temp_max_moy'] = ref_last['Temp_max_moy'] + 1.0
            if 'Temp_min_moy' in r: r['Temp_min_moy'] = ref_last.get('Temp_min_moy', ref_last['Temp_max_moy']-7) + 0.5
        if 'Humidite_max_moy' in r: r['Humidite_max_moy'] = max(30, ref_last.get('Humidite_max_moy', 80) - 3)
    elif si == 3:  # Poverty reduction
        for col in ['Pauvrete_region_2021_pct', 'Pauvrete_region_est_pct']:
            if pd.notna(ref_last.get(col)): r[col] = max(0, ref_last[col] - 10)
    elif si == 4:  # Poverty increase
        for col in ['Pauvrete_region_2021_pct', 'Pauvrete_region_est_pct']:
            if pd.notna(ref_last.get(col)): r[col] = min(100, ref_last[col] + 10)
    elif si == 5:  # Health improvement
        if pd.notna(ref_last.get('Acces_eau_potable_pct')): r['Acces_eau_potable_pct'] = min(100, ref_last['Acces_eau_potable_pct'] + 15)
        if pd.notna(ref_last.get('Assainissement_ameliore_pct')): r['Assainissement_ameliore_pct'] = min(100, ref_last['Assainissement_ameliore_pct'] + 10)
        if pd.notna(ref_last.get('Couverture_sanitaire_CSB2_10000hab')): r['Couverture_sanitaire_CSB2_10000hab'] = min(3.0, ref_last['Couverture_sanitaire_CSB2_10000hab'] + 0.3)
    elif si == 6:  # Favorable combined
        for col in ['Pauvrete_region_2021_pct', 'Pauvrete_region_est_pct']:
            if pd.notna(ref_last.get(col)): r[col] = max(0, ref_last[col] - 10)
        if pd.notna(ref_last.get('Precipitation_annuelle_mm')): r['Precipitation_annuelle_mm'] = ref_last['Precipitation_annuelle_mm'] * 1.10
        if pd.notna(ref_last.get('Temp_max_moy')): r['Temp_max_moy'] = ref_last['Temp_max_moy'] - 0.5
        if pd.notna(ref_last.get('Acces_eau_potable_pct')): r['Acces_eau_potable_pct'] = min(100, ref_last['Acces_eau_potable_pct'] + 15)
        if pd.notna(ref_last.get('Assainissement_ameliore_pct')): r['Assainissement_ameliore_pct'] = min(100, ref_last['Assainissement_ameliore_pct'] + 10)
    elif si == 7:  # Unfavorable combined
        for col in ['Pauvrete_region_2021_pct', 'Pauvrete_region_est_pct']:
            if pd.notna(ref_last.get(col)): r[col] = min(100, ref_last[col] + 10)
        if pd.notna(ref_last.get('Precipitation_annuelle_mm')): r['Precipitation_annuelle_mm'] = ref_last['Precipitation_annuelle_mm'] * 0.80
        if pd.notna(ref_last.get('Temp_max_moy')): r['Temp_max_moy'] = ref_last['Temp_max_moy'] + 1.0
    return r


def _build_feature_row(row, yr, fut_region, feature_cols, feature_stats, le):
    """Build a feature vector from a row dict for model prediction."""
    r = row.copy()
    r['Annee'] = yr
    r['Annee_sin'] = np.sin(2 * np.pi * (yr - 2012) / 11)
    r['Annee_cos'] = np.cos(2 * np.pi * (yr - 2012) / 11)
    if 'Region_enc' in feature_cols and hasattr(le, 'classes_'):
        r['Region_enc'] = le.transform([fut_region])[0] if fut_region in le.classes_ else 0
    X = pd.DataFrame([[r.get(c, feature_stats.get(c, {}).get('mean', 0)) for c in feature_cols]], columns=feature_cols)
    for c in feature_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna(feature_stats.get(c, {}).get('mean', 0))
    return X


def _predict_row(X, feature_cols, scaler, models, model_choice, meta, TARGETS):
    """Run prediction for all targets and return result dict."""
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)
    result = {}
    for tk in meta['targets']:
        m = models.get((model_choice, tk))
        if m:
            result[f'{TARGETS[tk]["short"]}_pred'] = round(float(m.predict(X_scaled)[0]), 2)
    return result


def _get_scenario_vars(si, lang):
    """Return a human-readable description of what variables the scenario modifies."""
    si_map = {
        0: {'fr': 'Aucune modification (tendances actuelles)', 'en': 'No modification (current trends)'},
        1: {'fr': 'Pr\u00e9cipitations +10%, Temp\u00e9rature -0.5\u00b0C, Humidit\u00e9 +2%', 'en': 'Rainfall +10%, Temperature -0.5\u00b0C, Humidity +2%'},
        2: {'fr': 'Pr\u00e9cipitations -20%, Temp\u00e9rature +1\u00b0C, Humidit\u00e9 -3%', 'en': 'Rainfall -20%, Temperature +1\u00b0C, Humidity -3%'},
        3: {'fr': 'Pauvret\u00e9 r\u00e9gionale -10 points (2021 + estim\u00e9e)', 'en': 'Regional poverty -10 points (2021 + estimated)'},
        4: {'fr': 'Pauvret\u00e9 r\u00e9gionale +10 points (2021 + estim\u00e9e)', 'en': 'Regional poverty +10 points (2021 + estimated)'},
        5: {'fr': 'Acc\u00e8s eau +15%, Assainissement +10%, CSB2 +0.3', 'en': 'Water access +15%, Sanitation +10%, CSB2 +0.3'},
        6: {'fr': 'Pauvret\u00e9 -10, Pr\u00e9cipitations +10%, Temp -0.5\u00b0C, Eau +15%', 'en': 'Poverty -10, Rainfall +10%, Temp -0.5\u00b0C, Water +15%'},
        7: {'fr': 'Pauvret\u00e9 +10, Pr\u00e9cipitations -20%, Temp +1\u00b0C', 'en': 'Poverty +10, Rainfall -20%, Temp +1\u00b0C'},
    }
    return si_map.get(si, si_map[0]).get(lang, si_map[0]['fr'])


def _get_scenario_impact(si, diff):
    """Return (emoji, color) for scenario impact."""
    if si == 0: return '\u2194', '#6b7280'
    if diff < -0.5: return '\u2705', '#27ae60'
    elif diff > 0.5: return '\u26a0\ufe0f', '#e74c3c'
    else: return '\u2194', '#f39c12'


# SCENARIOS
# ============================================================================
with tab_scenarios:
    lang = st.session_state.lang
    info = TARGETS[target_key]
    color = info['color']

    st.markdown(f'<h1><i class="fa-solid fa-wand-magic-sparkles"></i> {t("scenario")}</h1>', unsafe_allow_html=True)
    st.caption(f'{t("scenario_desc")} \u2014 {info[lang]}')

    if lang == 'fr':
        st.markdown('<div class="interp-box"><strong><i class="fa-solid fa-circle-question"></i> Que sont les sc\u00e9narios ?</strong><br>Les sc\u00e9narios simulent <strong>\u00ab et si ? \u00bb</strong> \u2014 que se passerait-il si certaines variables changeaient ?<br><em>Exemple :</em> \u00ab Et si la pauvret\u00e9 diminuait de 10 points \u00e0 Androy ? \u00bb<br>Cela permet aux d\u00e9cideurs de <strong>mesurer l&#39;impact concret</strong> d&#39;une politique avant sa mise en \u0153uvre.<br><small>\u2139\ufe0f Le mod\u00e8le pr\u00e9dit les 3 cibles nutritionnelles en modifiant les variables <strong>r\u00e9gionales r\u00e9elles</strong> du dataset (pauvret\u00e9 2021/estim\u00e9e, eau, assainissement, climat).</small></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="interp-box"><strong><i class="fa-solid fa-circle-question"></i> What are scenarios?</strong><br>Scenarios simulate <strong>"what if?"</strong> \u2014 what would happen if certain variables changed?<br><em>Example:</em> "What if poverty decreased by 10 points in Androy?"<br>This allows decision-makers to <strong>measure the concrete impact</strong> of a policy before implementing it.<br><small>\u2139\ufe0f The model predicts 3 nutritional targets by modifying <strong>actual regional</strong> variables from the dataset (poverty 2021/estimated, water, sanitation, climate).</small></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fut_region = st.selectbox(t('region_label'), REGIONS, key='fut_region')
    with c2:
        fut_horizon = st.selectbox("Horizon" if lang=='en' else "Horizon", ['2026\u20132028','2026\u20132030','2026\u20132035'], key='fut_horizon')

    end_year = int(fut_horizon.split('\u2013')[-1])
    years = list(range(2026, end_year + 1))

    if lang == 'fr':
        scenario_options = [
            'Statu quo (tendances actuelles)',
            'Am\u00e9lioration climatique (+10% pr\u00e9cipitations, -0.5\u00b0C)',
            'Aggravation climatique (-20% pr\u00e9cipitations, +1\u00b0C)',
            'R\u00e9duction de la pauvret\u00e9 (-10 points)',
            'Augmentation de la pauvret\u00e9 (+10 points)',
            'Am\u00e9lioration sanitaire (+15% acc\u00e8s eau, +10% assainissement)',
            'Sc\u00e9nario favorable (pauvret\u00e9 \u2193 + climat \u2191 + sant\u00e9 \u2191)',
            'Sc\u00e9nario d\u00e9favorable (pauvret\u00e9 \u2191 + climat \u2193 + chaleur \u2191)'
        ]
    else:
        scenario_options = [
            'Status quo (current trends)',
            'Climate improvement (+10% rainfall, -0.5\u00b0C)',
            'Climate degradation (-20% rainfall, +1\u00b0C)',
            'Poverty reduction (-10 points)',
            'Poverty increase (+10 points)',
            'Health improvement (+15% water access, +10% sanitation)',
            'Combined favorable (poverty \u2193 + climate \u2191 + health \u2191)',
            'Combined unfavorable (poverty \u2191 + climate \u2193 + heat \u2191)'
        ]

    scenario = st.selectbox(t('scenario'), scenario_options, key='fut_scenario')
    si = scenario_options.index(scenario) if scenario in scenario_options else 0

    st.info(f"{'Variables modifi\u00e9es' if lang=='fr' else 'Modified variable(s)'}: {_get_scenario_vars(si, lang)}")

    # Show reference values for the selected region
    ref_rows = df_ref[df_ref['Region'] == fut_region]
    if not ref_rows.empty:
        ref_last = ref_rows.sort_values('Annee').iloc[-1]
        if lang == 'fr':
            ref_text = f"**{fut_region} (r\u00e9f\u00e9rence {int(ref_last['Annee'])})** \u2014 Pauvret\u00e9: {ref_last.get('Pauvrete_region_2021_pct','\u2014')}% | Pr\u00e9cipitations: {ref_last.get('Precipitation_annuelle_mm','\u2014')}mm | Eau: {ref_last.get('Acces_eau_potable_pct','\u2014')}%"
        else:
            ref_text = f"**{fut_region} (ref {int(ref_last['Annee'])})** \u2014 Poverty: {ref_last.get('Pauvrete_region_2021_pct','\u2014')}% | Rainfall: {ref_last.get('Precipitation_annuelle_mm','\u2014')}mm | Water: {ref_last.get('Acces_eau_potable_pct','\u2014')}%"
        st.caption(ref_text)

    if st.button(t('btn_scenario'), type='primary', use_container_width=True, key='scenario_btn'):
        if ref_rows.empty:
            st.error(t('no_data'))
            st.stop()

        ref_last_dict = ref_rows.sort_values('Annee').iloc[-1].to_dict()

        # Compute status quo predictions (baseline)
        sq_results = []
        for yr in years:
            X_sq = _build_feature_row(ref_last_dict, yr, fut_region, feature_cols, feature_stats, le)
            res_sq = {'Annee': yr, 'Region': fut_region}
            res_sq.update(_predict_row(X_sq, feature_cols, scaler, models, model_choice, meta, TARGETS))
            sq_results.append(res_sq)
        df_sq = pd.DataFrame(sq_results)

        # Compute scenario predictions
        if si == 0:
            df_fut = df_sq.copy()
        else:
            fut_results = []
            running_vals = {}
            for tk in meta['targets']:
                tk_val = ref_last_dict.get(tk)
                running_vals[tk] = [tk_val] if tk_val is not None and pd.notna(tk_val) else []

            for yr in years:
                modified_row = _apply_scenario(ref_last_dict, si, ref_last_dict)
                X_fut = _build_feature_row(modified_row, yr, fut_region, feature_cols, feature_stats, le)

                # Update lag/ma3/delta features with running predictions
                for tk in meta['targets']:
                    for suffix in ['lag1', 'delta', 'ma3']:
                        col = f'{tk}_{suffix}'
                        if col in feature_cols and len(running_vals.get(tk, [])) > 0:
                            if suffix == 'lag1':
                                X_fut[col] = running_vals[tk][-1]
                            elif suffix == 'delta':
                                X_fut[col] = running_vals[tk][-1] - running_vals[tk][-2] if len(running_vals[tk]) >= 2 else 0.0
                            elif suffix == 'ma3':
                                X_fut[col] = float(np.mean(running_vals[tk][-3:])) if len(running_vals[tk]) >= 1 else feature_stats.get(col, {}).get('mean', 0)

                for c in feature_cols:
                    if X_fut[c].isnull().any():
                        X_fut[c] = X_fut[c].fillna(feature_stats.get(c, {}).get('mean', 0))

                res_fut = {'Annee': yr, 'Region': fut_region}
                res_fut.update(_predict_row(X_fut, feature_cols, scaler, models, model_choice, meta, TARGETS))

                # Store predictions for next iteration lag/ma3
                for tk in meta['targets']:
                    short = TARGETS[tk]['short']
                    pred_key = f'{short}_pred'
                    if pred_key in res_fut:
                        running_vals[tk].append(res_fut[pred_key])

                fut_results.append(res_fut)
            df_fut = pd.DataFrame(fut_results)

        # Display results
        st.markdown(f'<h3>{"R\u00e9sultats du sc\u00e9nario" if lang=="fr" else "Scenario results"}</h3>', unsafe_allow_html=True)

        # Impact metrics comparing scenario vs status quo
        if si != 0 and len(df_sq) > 0:
            st.markdown(f'<h4>{"Impact vs statu quo" if lang=="fr" else "Impact vs status quo"}</h4>', unsafe_allow_html=True)
            imp_cols = st.columns(len(TARGETS))
            for i, (tk, tinfo) in enumerate(TARGETS.items()):
                short = tinfo['short']
                pred_key = f'{short}_pred'
                if pred_key in df_fut.columns and pred_key in df_sq.columns:
                    last_sc = df_fut[pred_key].iloc[-1]
                    last_sq = df_sq[pred_key].iloc[-1]
                    diff = last_sc - last_sq
                    emoji_m, _ = _get_scenario_impact(si, diff)
                    with imp_cols[i]:
                        st.metric(tinfo[lang], f"{last_sc:.1f}%", f"{diff:+.1f}pp {'vs statu quo' if lang=='fr' else 'vs status quo'} {emoji_m}")

        # Results table with readable column names
        cols_display = ['Annee', 'Region'] + [f'{TARGETS[tk]["short"]}_pred' for tk in meta['targets'] if f'{TARGETS[tk]["short"]}_pred' in df_fut.columns]
        df_show = df_fut[cols_display].copy()
        rename_map = {'Annee': ('Ann\u00e9e' if lang=='fr' else 'Year'), 'Region': ('R\u00e9gion' if lang=='fr' else 'Region')}
        for tk in meta['targets']:
            short = TARGETS[tk]['short']
            rename_map[f'{short}_pred'] = TARGETS[tk][lang]
        df_show = df_show.rename(columns=rename_map)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        # Chart: historical + status quo + scenario
        fig = go.Figure()
        for tk in meta['targets']:
            short = TARGETS[tk]['short']
            clr = TARGETS[tk]['color']
            label = TARGETS[tk][lang]
            hist_data = df_ref[df_ref['Region'] == fut_region].sort_values('Annee')
            if tk in hist_data.columns:
                fig.add_trace(go.Scatter(x=hist_data['Annee'], y=hist_data[tk], mode='lines+markers', name=f'{label} ({t("historique")})', line=dict(color=clr, dash='solid'), marker=dict(size=5)))
            if f'{short}_pred' in df_sq.columns:
                fig.add_trace(go.Scatter(x=df_sq['Annee'], y=df_sq[f'{short}_pred'], mode='lines+markers', name=f'{label} (statu quo)' if lang=='fr' else f'{label} (status quo)', line=dict(color=clr, dash='dot'), marker=dict(size=5, symbol='diamond')))
            if si != 0 and f'{short}_pred' in df_fut.columns:
                fig.add_trace(go.Scatter(x=df_fut['Annee'], y=df_fut[f'{short}_pred'], mode='lines+markers', name=f'{label} ({t("scenario_label")})', line=dict(color=clr, dash='dash', width=3), marker=dict(size=8)))

        # Calculate Y-axis range from actual data to zoom in and make differences visible
        all_y_vals = []
        for tk in meta['targets']:
            hist_data = df_ref[df_ref['Region'] == fut_region].sort_values('Annee')
            if tk in hist_data.columns:
                all_y_vals.extend(hist_data[tk].dropna().tolist())
            short = TARGETS[tk]['short']
            if f'{short}_pred' in df_sq.columns:
                all_y_vals.extend(df_sq[f'{short}_pred'].dropna().tolist())
            if f'{short}_pred' in df_fut.columns:
                all_y_vals.extend(df_fut[f'{short}_pred'].dropna().tolist())
        if all_y_vals:
            y_min = max(0, min(all_y_vals) - 2)
            y_max = max(all_y_vals) + 2
            y_range = [y_min, y_max]
        else:
            y_range = None

        chart_title = f'{fut_region} \u2014 {scenario}'
        fig.update_layout(title=chart_title, height=560, hovermode='x unified', xaxis_title='Ann\u00e9e' if lang=='fr' else 'Year', yaxis_title='%', legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5, font=dict(size=11)), margin=dict(t=60, b=80), yaxis=dict(range=y_range, dtick=2))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(t('download_csv'), data=df_fut.to_csv(index=False).encode('utf-8'), file_name=f'pec_scenario_{fut_region}.csv', mime='text/csv')



# ---- A PROPOS ----
with tab_about:
    lang=st.session_state.lang
    st.markdown(f'<h1><i class="fa-solid fa-circle-info"></i> {t("tab_about")}</h1>',unsafe_allow_html=True)
    if lang=='fr':
        st.markdown('''<div class="interp-box">
        <strong>What-If PEC — Madagascar</strong> — Exploration interactive de scénarios.<br><br>
        Cette plateforme permet de <strong>simuler des prédictions</strong> pour une région et une année données, d’<strong>expliquer</strong> les facteurs influents via SHAP, et de <strong>certifier</strong> la confiance dans le résultat.
        <ul>
            <li>Sélectionnez une <strong>région</strong>, une <strong>cible</strong> et une <strong>année</strong></li>
            <li>Observez la prédiction avec les <strong>facteurs SHAP</strong></li>
            <li>Simulez des <strong>scénarios futurs</strong> (2026–2035)</li>
            <li>Consultez le <strong>score de certification</strong> (Grade A-D)</li>
        </ul>
        </div>''',unsafe_allow_html=True)
    else:
        st.markdown('''<div class="interp-box">
        <strong>What-If PEC — Madagascar</strong> — Interactive scenario exploration.<br><br>
        This platform allows you to <strong>simulate predictions</strong> for a given region and year, <strong>explain</strong> influential factors via SHAP, and <strong>certify</strong> confidence in the result.
        <ul>
            <li>Select a <strong>region</strong>, a <strong>target</strong> and a <strong>year</strong></li>
            <li>Observe the prediction with <strong>SHAP factors</strong></li>
            <li>Simulate <strong>future scenarios</strong> (2026–2035)</li>
            <li>Check the <strong>certification score</strong> (Grade A-D)</li>
        </ul>
        </div>''',unsafe_allow_html=True)

# FOOTER
# ============================================================================
st.markdown('---')
st.markdown("""<div style="text-align:center;color:#6b7280;font-size:.85rem;padding:1rem 0;">
    PEC Framework v4 | <i class="fa-solid fa-flask"></i> Predict \u2022 <i class="fa-solid fa-magnifying-glass"></i> Explain \u2022 <i class="fa-solid fa-shield-halved"></i> Certify |
    <a href="https://orcid.org/0009-0003-3048-1765" target="_blank" style="color:#a6ce39;"><i class="fa-brands fa-orcid"></i> Rosa Elysabeth Ralinirina</a>
</div>""", unsafe_allow_html=True)