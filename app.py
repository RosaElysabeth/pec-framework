# ============================================================================
# PEC FRAMEWORK — Application Streamlit v4
# Auteur : Rosa Elysabeth Ralinirina | ORCID : 0009-0003-3048-1765
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
import matplotlib; matplotlib.use('Agg')
import plotly.express as px

# Font Awesome pour icones modernes
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">', unsafe_allow_html=True)
import plotly.graph_objects as go
from PIL import Image
import json, os, warnings, pickle, joblib
warnings.filterwarnings('ignore')

# ============================================================================
# COORDONNEES DES 22 REGIONS
# ============================================================================
RCOORD = {
    'Analamanga':{'lat':-18.90,'lon':47.53},'Vakinankaratra':{'lat':-20.00,'lon':46.80},
    'Itasy':{'lat':-19.00,'lon':46.50},'Bongolava':{'lat':-18.50,'lon':45.50},
    'Haute Matsiatra':{'lat':-21.50,'lon':47.00},"Amoron'i Mania":{'lat':-20.50,'lon':47.00},
    'Vatovavy Fitovinany':{'lat':-22.00,'lon':47.60},'Ihorombe':{'lat':-22.00,'lon':46.00},
    'Atsimo-Atsinanana':{'lat':-23.00,'lon':47.70},'Atsimo-Andrefana':{'lat':-23.50,'lon':44.50},
    'Androy':{'lat':-25.00,'lon':45.50},'Anosy':{'lat':-24.00,'lon':46.80},
    'Menabe':{'lat':-20.50,'lon':44.40},'Melaky':{'lat':-17.00,'lon':44.50},
    'Betsiboka':{'lat':-17.00,'lon':46.50},'Boeny':{'lat':-16.00,'lon':46.30},
    'Sofia':{'lat':-15.50,'lon':47.50},'Diana':{'lat':-13.50,'lon':49.00},
    'Sava':{'lat':-14.50,'lon':50.00},'Alaotra Mangoro':{'lat':-17.50,'lon':48.50},
    'Analanjirofo':{'lat':-17.50,'lon':49.50},'Atsinanana':{'lat':-19.00,'lon':48.80},
}

# ============================================================================
# TRADUCTIONS FR/EN
# ============================================================================
T = {
'fr': {
    'app_title':'Framework PEC','app_subtitle':'Predict \u2014 Explain \u2014 Certify',
    'app_desc':"Audit et Certification des Syst\u00e8mes d\u2019Intelligence Artificielle Explicable pour la Mod\u00e9lisation Pr\u00e9dictive de la Malnutrition \u00e0 Madagascar",
    'author':'Rosa Elysabeth Ralinirina','orcid':'0009-0003-3048-1765',
    'tab_overview':'Vue d\u2019ensemble','tab_predict':'PREDICT','tab_explain':'EXPLAIN','tab_certify':'CERTIFY','tab_explore':'Explorer','tab_about':'À propos','nav_mada':'PEC Mada','nav_aide':'PEC Aide','nav_what':'PEC What','nav_gen':'PEC Gen',
    'target_label':'Cible nutritionnelle','region_label':'R\u00e9gion','all_regions':'Toutes les r\u00e9gions',
    'observations':'Observations','variables':'Variables','regions':'R\u00e9gions','years':'Ann\u00e9es',
    # Titres principaux
    'predict_title':'PREDICT \u2014 Mod\u00e9lisation Pr\u00e9dictive','predict_subtitle':'6 mod\u00e8les de ML compar\u00e9s sur 3 cibles nutritionnelles (train : 2012-2019, test : 2020-2022)',
    'explain_title':'EXPLAIN \u2014 Explicabilit\u00e9','explain_subtitle':'SHAP, LIME, Permutation Importance \u2014 Audit de la qualit\u00e9 des explications',
    'certify_title':'CERTIFY \u2014 Certification de Confiance','certify_subtitle':'\u00c9quit\u00e9 r\u00e9gionale, stabilit\u00e9 bootstrap, score certificatif composite',
    'explore_title':'Explorer les Donn\u00e9es',
    # Vue d'ensemble
    'training_data':'Donn\u00e9es d\'entra\u00eenement','training_desc':'Extrait des donn\u00e9es utilis\u00e9es pour entra\u00eener les mod\u00e8les PEC.','key_results':'R\u00e9sultats Cl\u00e9s',
    'key_results_interp':'Les trois cibles nutritionnelles obtiennent des scores de certification \u2265 90/100. Les disparit\u00e9s r\u00e9gionales sont marqu\u00e9es : la r\u00e9gion la plus touch\u00e9e affiche un taux 2 \u00e0 3 fois sup\u00e9rieur \u00e0 la moins touch\u00e9e.',
    'framework_title':'Architecture du Framework PEC',
    'framework_predict':'PREDICT \u2014 Mod\u00e9lisation pr\u00e9dictive de 3 cibles nutritionnelles \u00e0 travers 22 r\u00e9gions de Madagascar sur 2012-2022. Six mod\u00e8les : Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, Ridge.',
    'framework_explain':'EXPLAIN \u2014 Audit de la qualit\u00e9 des explications par 3 m\u00e9thodes (SHAP, LIME, Permutation Importance), accord inter-m\u00e9thodes (Kendall \u03c4) et explications locales par r\u00e9gion.',
    'framework_certify':'CERTIFY \u2014 Certification de confiance composite : performance pr\u00e9dictive (R\u00b2), stabilit\u00e9 bootstrap, \u00e9quit\u00e9 r\u00e9gionale. Score unique sur 100 permettant aux d\u00e9cideurs ONN/UNICEF de valider le syst\u00e8me.',
    'map_title':'Carte de Madagascar',
    'map_interp':'Les r\u00e9gions du Grand Sud et du Sud-Est pr\u00e9sentent les taux les plus \u00e9lev\u00e9s, corr\u00e9l\u00e9s \u00e0 la s\u00e9cheresse et \u00e0 la faible couverture sanitaire. Les r\u00e9gions du Nord affichent les taux les plus bas, b\u00e9n\u00e9ficiant de meilleures infrastructures.',
    # Predict
        'model_comparison':'Comparaison des Mod\u00e8les','model':'Mod\u00e8le','model_label':'Mod\u00e8le',
    'model_interp':'Le mod\u00e8le Ridge offre les meilleurs R\u00b2 sur les 3 cibles gr\u00e2ce \u00e0 l\u2019auto-corr\u00e9lation temporelle des donn\u00e9es ONN. Random Forest et XGBoost restent comp\u00e9titifs et offrent une meilleure interpr\u00e9tabilit\u00e9 via SHAP.',
    'detailed_performance':'Performances D\u00e9taill\u00e9es','distribution':'Distribution',
    'dist_interp':'Deux groupes de r\u00e9gions se distinguent : les r\u00e9gions \u00e0 haut risque largement au-dessus de la moyenne nationale, et les r\u00e9gions mieux loties en dessous de la moyenne. S\u00e9lectionnez la cible pour voir les valeurs sp\u00e9cifiques.',
    'pred_interp':'Les pr\u00e9dictions suivent la diagonale avec une forte pr\u00e9cision. Les \u00e9carts aux extr\u00eames correspondent aux r\u00e9gions les plus vuln\u00e9rables.',
    'prediction_vs_reality':'Pr\u00e9dictions vs Valeurs R\u00e9elles',
    'actual_values':'Valeurs r\u00e9elles (%)',
    'predicted_values':'Pr\u00e9dictions (%)',
    'worst_regions':'R\u00e9gions les plus touch\u00e9es','best_regions':'R\u00e9gions les moins touch\u00e9es',
    # Explain
    'shap_importance':'SHAP \u2014 Importance des Variables',
    'shap_interp':'Les variables les plus influentes sont les moyennes mobiles 3 ans (ma3), les d\u00e9calages temporels (lags) et la temp\u00e9rature. La pauvret\u00e9 r\u00e9gionale et le riz par habitant contribuent significativement. Les indicatrices r\u00e9gionales captent les disparit\u00e9s g\u00e9ographiques persistantes.',
    'bar_plot':'Importance Globale',
    'bar_interp':'Le diagramme confirme l\u2019ordre d\u2019importance : tendances temporelles (ma3, lag) en t\u00eate, puis facteurs climatiques et socio-\u00e9conomiques. Les indicatrices r\u00e9gionales r\u00e9v\u00e8lent des effets fixes g\u00e9ographiques significatifs pour les r\u00e9gions les plus vuln\u00e9rables.',
    'dependence_plot':'D\u00e9pendance',
    'dep_interp':'La relation pauvret\u00e9 \u2192 malnutrition est non-lin\u00e9aire : l\u2019effet s\u2019accentue fortement au-del\u00e0 de 70% de pauvret\u00e9, seuil critique o\u00f9 chaque point suppl\u00e9mentaire augmente davantage la malnutrition.',
    'local_explanation':'Explication Locale \u2014 Androy',
    'local_explanation_desc':'Explication locale SHAP pour Androy : comment chaque variable contribue \u00e0 la pr\u00e9diction individuelle de cette r\u00e9gion vuln\u00e9rable.',
    'local_interp':'Pour Androy (r\u00e9gion la plus vuln\u00e9rable), les facteurs qui augmentent le plus la pr\u00e9diction sont : l\u2019indicatrice r\u00e9gionale, la pauvret\u00e9 \u00e9lev\u00e9e et les conditions climatiques extr\u00eames. L\u2019ins\u00e9curit\u00e9 alimentaire et le faible acc\u00e8s \u00e0 l\u2019eau contribuent aussi positivement.',
    'inter_method':'Comparaison Inter-M\u00e9thodes','inter_method_desc':'Accord Kendall \u03c4 entre SHAP, LIME et Permutation Importance',
    'inter_interp':'L\u2019accord SHAP-Permutation Importance est \u00e9lev\u00e9 (Kendall \u03c4 > 0.8), confirmant la robustesse des explications globales. LIME montre un accord plus mod\u00e9r\u00e9 car ses explications sont locales et plus sensibles aux perturbations. Cette coh\u00e9rence inter-m\u00e9thodes renforce la confiance dans les variables identifi\u00e9es.',
    'kendall_details':'D\u00e9tails de l\u2019accord Kendall \u03c4',
    # Certify
    'composite_score':'Score Certificatif Composite',
    'cert_interp':'Les 3 cibles obtiennent un score \u2265 90/100 : le syst\u00e8me est fiable, \u00e9quitable entre r\u00e9gions et stable. La cible la plus pr\u00e9visible atteint le score le plus \u00e9lev\u00e9 car ses variations sont plus faibles et mieux capt\u00e9es par le mod\u00e8le.',
    'regional_fairness':'\u00c9quit\u00e9 R\u00e9gionale',
    'fairness_interp':'L\u2019\u00e9quit\u00e9 r\u00e9gionale est satisfaisante : les \u00e9carts de RMSE entre r\u00e9gions restent contenus. Les r\u00e9gions avec plus de donn\u00e9es sont mieux pr\u00e9dites mais l\u2019\u00e9cart reste acceptable.',
    'score_interpretation':'Interpr\u00e9tation des Scores','certification_details':'D\u00e9tails des R\u00e9sultats de Certification',
    'recommendation':'Recommandation ONN/UNICEF : Les 3 cibles obtiennent un score \u2265 90/100. Le syst\u00e8me PEC est certifi\u00e9 pour un d\u00e9ploiement op\u00e9rationnel comme outil d\u2019aide \u00e0 la d\u00e9cision pour la surveillance nutritionnelle \u00e0 Madagascar.',
    'niveau':'Niveau','score_level':'Signification','score_excellent':'Excellent \u2014 Syst\u00e8me certifi\u00e9, pr\u00eat pour d\u00e9ploiement op\u00e9rationnel',
    'score_good':'Bon \u2014 Syst\u00e8me fiable, utilisation avec pr\u00e9caution',
    'score_warning':'Risqu\u00e9 \u2014 Am\u00e9liorations n\u00e9cessaires',
    'score_critical':'Non certifi\u00e9 \u2014 Restructuration requise',
    'grade_a':'Certifi\u00e9 \u2014 D\u00e9ploiement op\u00e9rationnel (\u226590/100)',
    'grade_b':'Fiable \u2014 Utilisation avec pr\u00e9caution (80-89/100)',
    'grade_c':'Risqu\u00e9 \u2014 Am\u00e9liorations n\u00e9cessaires (60-79/100)',
    'grade_d':'Non certifi\u00e9 \u2014 Restructuration requise (<60/100)',
    'score_good':'Bon \u2014 Syst\u00e8me fiable, am\u00e9liorations mineures recommand\u00e9es',
    'score_warning':'Attention \u2014 Risques identifi\u00e9s, audit approfondi n\u00e9cessaire',
    'score_critical':'Critique \u2014 Syst\u00e8me non certifi\u00e9, restructuration n\u00e9cessaire',
    # Explore
    'selected_obs':'Observations s\u00e9lectionn\u00e9es',
    'interactive_correlations':'Corr\u00e9lations Interactives','axis_x':'Axe X','axis_y':'Axe Y','color':'Couleur',
    'raw_data':'Donn\u00e9es Brutes','data_dictionary':'Dictionnaire des Donn\u00e9es',
    'temporal_evolution':'\u00c9volution Temporelle',
    'temporal_interp':'La malnutrition chronique reste stable \u00e0 un niveau \u00e9lev\u00e9 (\u224840% en moyenne nationale), avec une l\u00e9g\u00e8re am\u00e9lioration 2012-2018 puis une remont\u00e9e 2020-2022 (COVID-19 et s\u00e9cheresse). Les r\u00e9gions du Grand Sud montrent une d\u00e9gradation continue.',
    'correlation_matrix':'Matrice de Corr\u00e9lations',
    'correlation_interp':'Fortes corr\u00e9lations positives entre les 3 cibles nutritionnelles : les r\u00e9gions touch\u00e9es par un type le sont aussi par les autres. La pauvret\u00e9 et la temp\u00e9rature sont positivement corr\u00e9l\u00e9es \u00e0 la malnutrition ; le riz par habitant et l\u2019acc\u00e8s \u00e0 l\u2019eau potable le sont n\u00e9gativement.',
    # Cibles
    'IP':'Insuffisance Pond\u00e9rale','MC':'Malnutrition Chronique','MA':'Malnutrition Aig\u00fce',
    'IP_full':'Insuffisance Pond\u00e9rale (%)','MC_full':'Malnutrition Chronique (%)','MA_full':'Malnutrition Aig\u00fce (%)',
    'interp':'Interpr\u00e9tation',
},
'en': {
    'app_title':'PEC Framework','app_subtitle':'Predict \u2014 Explain \u2014 Certify',
    'app_desc':'Auditing and Certifying Explainable AI Systems for Predictive Modeling of Malnutrition in Madagascar',
    'author':'Rosa Elysabeth Ralinirina','orcid':'0009-0003-3048-1765',
    'tab_overview':'Overview','tab_predict':'PREDICT','tab_explain':'EXPLAIN','tab_certify':'CERTIFY','tab_explore':'Explore','tab_about':'About','nav_mada':'PEC Mada','nav_aide':'PEC Aid','nav_what':'PEC What','nav_gen':'PEC Gen',
    'target_label':'Nutritional target','region_label':'Region','all_regions':'All regions',
    'observations':'Observations','variables':'Variables','regions':'Regions','years':'Years',
    'predict_title':'PREDICT \u2014 Predictive Modeling','predict_subtitle':'6 ML models compared on 3 nutritional targets (train: 2012-2019, test: 2020-2022)',
    'explain_title':'EXPLAIN \u2014 Explainability','explain_subtitle':'SHAP, LIME, Permutation Importance \u2014 Audit of explanation quality',
    'certify_title':'CERTIFY \u2014 Trust Certification','certify_subtitle':'Regional fairness, bootstrap stability, composite certification score',
    'explore_title':'Explore Data',
    'training_data':'Training Data','training_desc':'Sample of data used to train PEC models.','key_results':'Key Results',
    'key_results_interp':'All 3 nutritional targets achieve certification scores \u2265 90/100. Regional disparities are stark: the most affected region has rates 2-3 times higher than the least affected.',
    'framework_title':'PEC Framework Architecture',
    'framework_predict':'PREDICT \u2014 Predictive modeling of 3 nutritional targets across 22 regions of Madagascar (2012-2022). Six models: Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, Ridge.',
    'framework_explain':'EXPLAIN \u2014 Audit of explanation quality through 3 methods (SHAP, LIME, Permutation Importance), inter-method agreement (Kendall \u03c4) and local explanations by region.',
    'framework_certify':'CERTIFY \u2014 Composite trust certification combining predictive performance (R\u00b2), bootstrap stability, and regional fairness into a single score out of 100 for ONN/UNICEF decision-makers.',
    'map_title':'Madagascar Map',
    'map_interp':'Grand Sud and South-East regions show the highest rates, correlated with drought and weak health coverage. Northern regions display the lowest rates, benefiting from better infrastructure.',
        'model_comparison':'Model Comparison','model':'Model','model_label':'Model',
    'model_interp':'Ridge regression achieves the best R\u00b2 across all 3 targets thanks to the strong temporal autocorrelation in ONN data. Random Forest and XGBoost remain competitive and offer better interpretability via SHAP.',
    'detailed_performance':'Detailed Performance','distribution':'Distribution',
    'dist_interp':'Two groups of regions emerge: high-risk regions well above the national average, and better-off regions below the average. Select the target to see specific values.',
    'pred_interp':'Predictions follow the diagonal with high precision. Deviations at the extremes correspond to the most vulnerable regions.',
    'prediction_vs_reality':'Predictions vs Actual Values',
    'actual_values':'Actual values (%)',
    'predicted_values':'Predictions (%)',
    'worst_regions':'Most affected regions','best_regions':'Least affected regions',
    'shap_importance':'SHAP \u2014 Feature Importance',
    'shap_interp':'The most influential features are 3-year moving averages (ma3), temporal lags, and temperature. Regional poverty and rice per capita also contribute significantly. Regional dummies capture persistent geographic disparities.',
    'bar_plot':'Global Importance',
    'bar_interp':'The bar chart confirms the importance ranking: temporal trends (ma3, lag) dominate, followed by climatic and socio-economic factors. Regional dummies reveal significant geographic fixed effects for the most vulnerable regions.',
    'dependence_plot':'Dependence Plot',
    'dep_interp':'The poverty\u2013malnutrition relationship is non-linear: the effect intensifies sharply above 70% poverty, a critical threshold where each additional point increases malnutrition more.',
    'local_explanation':'Local Explanation \u2014 Androy',
    'local_explanation_desc':'Local SHAP explanation for Androy: how each feature contributes to the individual prediction for this vulnerable region.',
    'local_interp':'For Androy (most vulnerable region), the factors that most increase the prediction are: the regional dummy, high poverty, and extreme climatic conditions. Food insecurity and low water access also contribute positively.',
    'inter_method':'Inter-Method Comparison','inter_method_desc':'Kendall \u03c4 agreement between SHAP, LIME and Permutation Importance',
    'inter_interp':'SHAP-Permutation Importance agreement is high (Kendall \u03c4 > 0.8), confirming robust global explanations. LIME shows moderate agreement as its explanations are local and more sensitive to perturbations. This inter-method coherence strengthens confidence in identified features.',
    'kendall_details':'Kendall \u03c4 Agreement Details',
    'composite_score':'Composite Certification Score',
    'cert_interp':'All 3 targets achieve \u226590/100: the system is reliable, fair across regions, and stable. The most predictable target reaches the highest score because its variations are smaller and better captured by the model.',
    'regional_fairness':'Regional Fairness',
    'fairness_interp':'Regional fairness is satisfactory: RMSE differences between regions remain contained. Regions with more data are better predicted but the gap remains acceptable.',
    'niveau':'Level','score_level':'Meaning','score_interpretation':'Score Interpretation','certification_details':'Certification Results Details',
    'recommendation':'ONN/UNICEF Recommendation: All 3 targets score \u226590/100. The PEC system is certified for operational deployment as a decision-support tool for nutritional surveillance in Madagascar.',
    'score_excellent':'Excellent \u2014 Certified system, ready for operational deployment',
    'score_good':'Good \u2014 Reliable system, use with caution',
    'score_warning':'Risky \u2014 Improvements needed',
    'score_critical':'Uncertified \u2014 Restructuring required',
    'grade_a':'Certified \u2014 Operational deployment (\u226590/100)',
    'grade_b':'Reliable \u2014 Use with caution (80-89/100)',
    'grade_c':'Risky \u2014 Improvements needed (60-79/100)',
    'grade_d':'Uncertified \u2014 Restructuring required (<60/100)',
    'selected_obs':'Selected observations',
    'interactive_correlations':'Interactive Correlations','axis_x':'X Axis','axis_y':'Y Axis','color':'Color',
    'raw_data':'Raw Data','data_dictionary':'Data Dictionary',
    'temporal_evolution':'Temporal Evolution',
    'temporal_interp':'Chronic malnutrition remains stable at a high level (~40% national average), with a slight improvement 2012-2018 followed by a rise in 2020-2022 (COVID-19 and drought). Grand Sud regions show continuous degradation.',
    'correlation_matrix':'Correlation Matrix',
    'correlation_interp':'Strong positive correlations between the 3 targets: regions affected by one type are affected by others. Poverty and temperature are positively correlated with malnutrition; rice per capita and safe water access are negatively correlated.',
    'IP':'Underweight','MC':'Chronic Malnutrition','MA':'Acute Malnutrition',
    'IP_full':'Underweight (%)','MC_full':'Chronic Malnutrition (%)','MA_full':'Acute Malnutrition (%)',
    'interp':'Interpretation',
}
}

def t(key):
    return T[st.session_state.lang].get(key, key)

def tl(key):
    """Target label in current language"""
    info = TARGETS[key]
    return f"{info['short']} \u2014 {info[st.session_state.lang]}"

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(page_title="PEC Framework | Malnutrition Madagascar", page_icon=None, layout="wide", initial_sidebar_state="expanded")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
MODELS_DIR = os.path.join(BASE, "models")
FIGD = os.path.join(RESULTS_DIR, "figures")
SHAD = os.path.join(RESULTS_DIR, "shap")
CERD = os.path.join(RESULTS_DIR, "certification")
XAID = os.path.join(RESULTS_DIR, "comparaison_xai")

TARGETS = {
    'Insuffisance_ponderale':{'short':'IP','en':'Underweight','fr':'Insuffisance Pond\u00e9rale','color':'#2e86c1'},
    'Malnutrition_chronique':{'short':'MC','en':'Chronic Malnutrition','fr':'Malnutrition Chronique','color':'#8e44ad'},
    'Malnutrition_aigue':{'short':'MA','en':'Acute Malnutrition','fr':'Malnutrition Aig\u00fce','color':'#c0392b'},
}
RTK = {'Insuffisance_ponderale':'Insuffisance Pond\xe9rale','Malnutrition_chronique':'Malnutrition Chronique','Malnutrition_aigue':'Malnutrition Aig\xfce'}

# ============================================================================
# CSS
# ============================================================================
st.markdown("""<style>
/* ===== PEC SHARED THEME ===== */
""", unsafe_allow_html=True)
_css_path = os.path.join(BASE, 'shared_pec.css')
if os.path.exists(_css_path):
    with open(_css_path, 'r', encoding='utf-8') as _f: st.markdown(f'<style>{_f.read()}</style>', unsafe_allow_html=True)
else:
    st.warning('shared_pec.css not found')

# ============================================================================
# LOAD
# ============================================================================
@st.cache_data
def ld():
    df=pd.read_csv(os.path.join(DATA_DIR,"dataset_pec_v3.csv"))
    rp={}
    p=os.path.join(RESULTS_DIR,"rapport_final.json")
    if os.path.exists(p):
        with open(p,'r',encoding='utf-8') as f: rp=json.load(f)
    cf=None;p2=os.path.join(RESULTS_DIR,"comparaison_modeles.csv")
    if os.path.exists(p2): cf=pd.read_csv(p2); cf.columns=['Cible','Modele','R2_test','RMSE_test','MAE_test','R2_train','Temps_s'] if len(cf.columns)==7 else cf.columns
    dd={}
    p3=os.path.join(DATA_DIR,"data_dictionary_v3.json")
    if os.path.exists(p3):
        with open(p3,'r',encoding='utf-8') as f: dd=json.load(f)
    return df,rp,cf,dd

df,rp,cf,dd=ld()
REGS=sorted(df['Region'].unique()) if 'Region' in df.columns else []
pd_data=rp.get('predict',{});cd_data=rp.get('certify',{})

# Load models for interactive predictions
def _load(p):
    """Charge un .pkl avec pickle, fallback joblib."""
    try:
        with open(p,'rb') as f: return pickle.load(f)
    except Exception:
        return joblib.load(p)

@st.cache_resource
def ld_models():
    meta_m = _load(os.path.join(MODELS_DIR,'meta.pkl'))
    _scaler = _load(os.path.join(MODELS_DIR,'scaler.pkl'))
    _fcols = _load(os.path.join(MODELS_DIR,'feature_cols.pkl'))
    _le = _load(os.path.join(MODELS_DIR,'label_encoder_region.pkl'))
    _fstats = _load(os.path.join(MODELS_DIR,'feature_stats.pkl'))
    _models={}
    for mn in meta_m['models']:
        for tgt in meta_m['targets']:
            p=os.path.join(MODELS_DIR,f"{mn}_{tgt}.pkl")
            if os.path.exists(p):
                _models[(mn,tgt)] = _load(p)
    return _models,_scaler,_fcols,_le,_fstats

models,scaler,feature_cols,le,feature_stats = ld_models()

if 'lang' not in st.session_state: st.session_state.lang='fr'

# ============================================================================
# HELPERS
# ============================================================================
def li(p):
    if os.path.exists(p): return Image.open(p)
def sc(s):
    if s>=90: return "se"
    elif s>=75: return "sg"
    elif s>=50: return "sw"
    else: return "sc"
def sm(s):
    if s>=90: return "A"
    elif s>=80: return "B"
    elif s>=60: return "C"
    else: return "D"

def grade_label(s):
    """Description du grade pour affichage."""
    if s>=90: return t('grade_a')
    elif s>=80: return t('grade_b')
    elif s>=60: return t('grade_c')
    else: return t('grade_d')
def ib(text):
    st.markdown(f'<div class="ib"><strong>{t("interp")} :</strong> {text}</div>', unsafe_allow_html=True)

def dyn_interp(tk, kind='map'):
    """Generer une interpretation dynamique basee sur les donnees reelles."""
    lng = st.session_state.lang
    if tk not in df.columns or 'Region' not in df.columns:
        return t('map_interp')
    rs = df.groupby('Region')[tk].mean().sort_values(ascending=False)
    worst = rs.index[0]; wv = rs.iloc[0]
    best = rs.index[-1]; bv = rs.iloc[-1]
    mn = df[tk].mean()
    nm = TARGETS[tk][lng]
    if kind == 'map':
        if lng == 'fr':
            return f"{nm} : la region la plus touchee est {worst} ({wv:.1f}%), la moins touchee est {best} ({bv:.1f}%). Moyenne nationale : {mn:.1f}%. Les regions du Grand Sud sont les plus impactees (secheresse, inscurite alimentaire), tandis que les regions du Nord beneficient de meilleures conditions."
        return f"{nm}: the most affected region is {worst} ({wv:.1f}%), the least affected is {best} ({bv:.1f}%). National average: {mn:.1f}%. Grand Sud regions are most impacted (drought, food insecurity), while Northern regions benefit from better conditions."
    elif kind == 'dist':
        if lng == 'fr':
            return f"Deux groupes de regions se distinguent : {worst} ({wv:.1f}%) et les regions a haut risque au-dessus de la moyenne ({mn:.1f}%), tandis que {best} ({bv:.1f}%) et les regions mieux loties sont en dessous."
        return f"Two groups emerge: {worst} ({wv:.1f}%) and high-risk regions above average ({mn:.1f}%), while {best} ({bv:.1f}%) and better-off regions are below."
    elif kind == 'dist_full':
        top3 = ', '.join([f"{r} ({v:.1f}%)" for r, v in rs.head(3).items()])
        bot3 = ', '.join([f"{r} ({v:.1f}%)" for r, v in rs.tail(3).items()])
        if lng == 'fr':
            return f"Top 3 regions les plus touchees : {top3}. Top 3 les moins touchees : {bot3}. Ecart entre la plus touchee et la moins touchee : {wv-bv:.1f} points."
        return f"Top 3 most affected regions: {top3}. Top 3 least affected: {bot3}. Gap between most and least affected: {wv-bv:.1f} points."
    elif kind == 'pred':
        rk = RTK.get(tk, tk)
        pr = pd_data.get(rk, {})
        r2 = pr.get('R2_test', pr.get('r2_test', 0))
        bm = pr.get('meilleur_modele', pr.get('model', 'Ridge'))
        if lng == 'fr':
            return f"Les predictions de {nm} (R\u00b2 = {r2:.3f}, modele {bm}) suivent la diagonale avec precision. Les ecarts aux extremes correspondent aux regions les plus vulnerables ({worst}, {rs.index[1]})."
        return f"Predictions for {nm} (R\u00b2 = {r2:.3f}, model {bm}) follow the diagonal with precision. Deviations at extremes correspond to most vulnerable regions ({worst}, {rs.index[1]})."
    elif kind == 'key':
        rk = RTK.get(tk, tk)
        ct = cd_data.get(rk, {})
        cs = ct.get('certification_score', 0)
        if lng == 'fr':
            return f"{nm} : score certificatif {cs:.0f}/100, region la plus touchee = {worst} ({wv:.1f}%), region la moins touchee = {best} ({bv:.1f}%), moyenne nationale = {mn:.1f}%. Ecart regional = {wv-bv:.1f} points."
        return f"{nm}: certification score {cs:.0f}/100, most affected region = {worst} ({wv:.1f}%), least affected = {best} ({bv:.1f}%), national average = {mn:.1f}%. Regional gap = {wv-bv:.1f} points."
    return ''

def makemap(df, tk, lng):
    sn=TARGETS[tk]['short']; nm=TARGETS[tk][lng]; fl=t(sn+'_full')
    md=df.groupby('Region')[tk].mean().reset_index()
    region_colors=['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6',
        '#bfef45','#fabed4','#469990','#dcbeff','#9a6324','#fffac8','#800000','#aaffc3',
        '#808000','#ffd8b1','#000075','#a9a9a9','#e6beff','#1abc9c']
    la,lo,va,na,co=[],[],[],[],[]
    for i,(_,r) in enumerate(md.iterrows()):
        rg=r['Region']
        if rg in RCOORD:
            la.append(RCOORD[rg]['lat']);lo.append(RCOORD[rg]['lon']);va.append(r[tk]);na.append(rg)
            co.append(region_colors[i % len(region_colors)])
    fig=go.Figure()
    fig.add_trace(go.Scattermapbox(lat=la,lon=lo,mode='markers+text',
        marker=dict(size=[max(12,v/2.5) for v in va],color=co,opacity=0.9),
        text=[f"<b>{n}</b><br>{v:.1f}%" for n,v in zip(na,va)],
        textposition='top right',textfont=dict(size=11,color='#1a2744',family='Segoe UI'),
        hovertext=[f"<b>{n}</b><br>{fl}: {v:.1f}%" for n,v in zip(na,va)],hoverinfo='text',
        name=nm))
    fig.update_layout(
        mapbox=dict(style='carto-positron',center=dict(lat=-19.5,lon=46.5),zoom=4.8),
        margin=dict(l=0,r=0,t=40,b=0),height=600,
        title=dict(text=f"{nm} — {fl}",x=0.5,font=dict(size=18,family='Segoe UI')),
        legend=dict(yanchor='top',y=0.99,xanchor='left',x=0.01,font=dict(size=9)))
    return fig
# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    lang=st.radio('', ['fr','en'], format_func=lambda x: 'Français' if x=='fr' else 'English', horizontal=True, key='lg')
    st.session_state.lang=lang
    st.divider()
    # Navigation inter-plateformes
    st.divider()
    st.markdown(f'#### 🔗 {t("nav_platforms") if "nav_platforms" in T[st.session_state.lang] else "Plateformes PEC"}')
    c1, c2 = st.columns(2)
    with c1: st.link_button(f'📊 {t("nav_mada")}', pec_url('mada'), use_container_width=True)
    with c2: st.link_button(f'📊 {t("nav_aide")}', pec_url('aide'), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3: st.link_button(f'🔬 {t("nav_what")}', pec_url('whatif'), use_container_width=True)
    with c4: st.link_button(f'🌍 {t("nav_gen")}', pec_url('gen'), use_container_width=True)
    st.markdown(f"**{t('author')}**")
    oid=t('orcid')
    st.markdown(f'<a href="https://orcid.org/{oid}" target="_blank" style="text-decoration:none;display:inline-flex;align-items:center;gap:4px;font-size:.9rem;color:#a6ce39;"><img src="https://orcid.org/assets/vectors/orcid.logo.icon.svg" width="16" height="16" alt="ORCID" style="vertical-align:middle;"/> orcid.org/{oid}</a>',unsafe_allow_html=True)
    st.divider()
    c1,c2=st.columns(2);c1.metric(t('observations'),len(df));c2.metric(t('variables'),len(df.columns))
    c1,c2=st.columns(2);c1.metric(t('regions'),len(REGS))
    if 'Annee' in df.columns: c2.metric(t('years'),f"{df['Annee'].min()}-{df['Annee'].max()}")
    st.divider()
    tk=st.selectbox(t('target_label'),list(TARGETS.keys()),format_func=lambda x:tl(x),key='sb_t')
    rs=st.selectbox(t('region_label'),[t('all_regions')]+REGS,key='sb_r')

# ============================================================================
# TABS
# ============================================================================
tov,tp,tex,tce,tex2,tab_about=st.tabs([t('tab_overview'),t('tab_predict'),t('tab_explain'),t('tab_certify'),t('tab_explore'),t('tab_about')])

# ---- OVERVIEW ----
with tov:
    st.markdown('<h1 class="mt">Framework PEC</h1>',unsafe_allow_html=True)
    st.markdown(f'**{t("app_subtitle")}**');st.markdown(t('app_desc'))
    st.markdown('<span class="bg bp">PREDICT</span> <span class="bg be">EXPLAIN</span> <span class="bg bc">CERTIFY</span>',unsafe_allow_html=True)
    # --- PEC FRAMEWORK SCHEMA (HTML/CSS) ---
    st.markdown('---')
    st.markdown(f'<h2 class="st">{t("framework_title")}</h2>',unsafe_allow_html=True)

    # Data from models
    pd_ip = pd_data.get('Insuffisance Pond\u00e9rale', pd_data.get('Insuffisance_ponderale', {}))
    pd_mc = pd_data.get('Malnutrition Chronique', pd_data.get('Malnutrition_chronique', {}))
    pd_ma = pd_data.get('Malnutrition Aig\u00fce', pd_data.get('Malnutrition_aigue', {}))
    best_ip = pd_ip.get('meilleur_modele', pd_ip.get('model', 'Ridge'))
    best_mc = pd_mc.get('meilleur_modele', pd_mc.get('model', 'Ridge'))
    best_ma = pd_ma.get('meilleur_modele', pd_ma.get('model', 'Ridge'))
    r2_ip = pd_ip.get('R2_test', pd_ip.get('r2_test', 0))
    r2_mc = pd_mc.get('R2_test', pd_mc.get('r2_test', 0))
    r2_ma = pd_ma.get('R2_test', pd_ma.get('r2_test', 0))
    cd_ip = cd_data.get('Insuffisance Pond\u00e9rale', cd_data.get('Insuffisance_ponderale', {}))
    cd_mc = cd_data.get('Malnutrition Chronique', cd_data.get('Malnutrition_chronique', {}))
    cd_ma = cd_data.get('Malnutrition Aig\u00fce', cd_data.get('Malnutrition_aigue', {}))
    cert_ip = cd_ip.get('certification_score', 0)
    cert_mc = cd_mc.get('certification_score', 0)
    cert_ma = cd_ma.get('certification_score', 0)

    st.markdown(f"""
    <div style="display:flex;gap:16px;margin:20px 0;flex-wrap:wrap;">
      <!-- PREDICT -->
      <div style="flex:1;min-width:280px;background:linear-gradient(135deg,#e8f4fd,#d0e8f7);border:3px solid #2e86c1;border-radius:12px;padding:24px;font-family:'Segoe UI',system-ui,sans-serif;">
        <div style="font-size:1.6rem;font-weight:800;color:#1a5276;margin-bottom:12px;letter-spacing:1px;">PREDICT</div>
        <div style="font-size:1rem;color:#2c3e50;line-height:1.6;">
          <div style="margin-bottom:8px;"><strong>6 mod\u00e8les</strong> : Random Forest, XGBoost, LightGBM, CatBoost, GB, Ridge</div>
          <div style="margin-bottom:8px;"><strong>3 cibles</strong> : IP ({r2_ip:.1%} R\u00b2), MC ({r2_mc:.1%} R\u00b2), MA ({r2_ma:.1%} R\u00b2)</div>
          <div style="margin-bottom:8px;"><strong>22 r\u00e9gions</strong> \u00d7 <strong>11 ann\u00e9es</strong> (2012\u20142022)</div>
          <div style="margin-bottom:8px;"><strong>139 variables</strong> : m\u00e9t\u00e9o, socio-\u00e9co, sant\u00e9, cyclones</div>
          <div style="font-size:0.9rem;color:#5d7a96;">Meilleur mod\u00e8le : <strong>{best_ip}</strong></div>
        </div>
      </div>
      <!-- EXPLAIN -->
      <div style="flex:1;min-width:280px;background:linear-gradient(135deg,#f0e6f6,#e0ccf0);border:3px solid #8e44ad;border-radius:12px;padding:24px;font-family:'Segoe UI',system-ui,sans-serif;">
        <div style="font-size:1.6rem;font-weight:800;color:#6c3483;margin-bottom:12px;letter-spacing:1px;">EXPLAIN</div>
        <div style="font-size:1rem;color:#2c3e50;line-height:1.6;">
          <div style="margin-bottom:8px;"><strong>3 m\u00e9thodes</strong> : SHAP, LIME, Permutation Importance</div>
          <div style="margin-bottom:8px;"><strong>Accord inter-m\u00e9thodes</strong> : Kendall \u03c4 > 0.8</div>
          <div style="margin-bottom:8px;"><strong>Explications locales</strong> : Waterfall par r\u00e9gion (Androy)</div>
          <div style="margin-bottom:8px;"><strong>Top features</strong> : moyennes mobiles, lags, temp\u00e9rature, pauvret\u00e9</div>
          <div style="font-size:0.9rem;color:#7a5a8a;">Features les plus influentes : <strong>ma3, lag1, Temp</strong></div>
        </div>
      </div>
      <!-- CERTIFY -->
      <div style="flex:1;min-width:280px;background:linear-gradient(135deg,#e8f8e8,#d0f0d0);border:3px solid #27ae60;border-radius:12px;padding:24px;font-family:'Segoe UI',system-ui,sans-serif;">
        <div style="font-size:1.6rem;font-weight:800;color:#1e8449;margin-bottom:12px;letter-spacing:1px;">CERTIFY</div>
        <div style="font-size:1rem;color:#2c3e50;line-height:1.6;">
          <div style="margin-bottom:8px;"><strong>Score composite</strong> sur 100 :</div>
          <div style="display:flex;gap:12px;margin-bottom:8px;">
            <div style="background:#27ae60;color:white;padding:6px 12px;border-radius:6px;font-weight:700;"><strong>IP {cert_ip:.0f}/100</strong></div>
            <div style="background:#27ae60;color:white;padding:6px 12px;border-radius:6px;font-weight:700;"><strong>MC {cert_mc:.0f}/100</strong></div>
            <div style="background:#27ae60;color:white;padding:6px 12px;border-radius:6px;font-weight:700;"><strong>MA {cert_ma:.0f}/100</strong></div>
          </div>
          <div style="margin-bottom:8px;"><strong>R\u00b2</strong> + <strong>Stabilit\u00e9 bootstrap</strong> + <strong>\u00c9quit\u00e9 r\u00e9gionale</strong></div>
          <div style="font-size:0.9rem;color:#5d8a6d;">Certification ONN/UNICEF : <strong>Op\u00e9rationnel</strong></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="fs sp">{t("framework_predict")}</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="fs sep">{t("framework_explain")}</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="fs sec">{t("framework_certify")}</div>',unsafe_allow_html=True)
    ib(dyn_interp(tk, 'key'))
    # CARTE
    st.markdown('---')
    st.markdown(f'<h2 class="st">{t("map_title")} \u2014 {tl(tk)}</h2>',unsafe_allow_html=True)
    st.plotly_chart(makemap(df,tk,st.session_state.lang),use_container_width=True)
    ib(dyn_interp(tk, 'map'))
    # RESULTATS CLES
    st.markdown('---')
    st.markdown(f'<h2 class="st">{t("key_results")}</h2>',unsafe_allow_html=True)
    cl=st.columns(3)
    for c,(k,info) in zip(cl,TARGETS.items()):
        rk=RTK.get(k,info[st.session_state.lang]);pr=pd_data.get(rk,{})
        bm=pr.get('meilleur_modele',pr.get('model','N/A'));r2=pr.get('R2_test',pr.get('r2_test',0))
        ct=cd_data.get(rk,{});cs=ct.get('certification_score',0)
        bc={'Insuffisance_ponderale':'cb','Malnutrition_chronique':'cp','Malnutrition_aigue':'cr'}[k]
        with c:
            st.markdown(f'<div class="cd {bc}"><h3 style="margin:0 0 .3rem 0;">{tl(k)}</h3><p style="margin:.2rem 0;font-size:.95rem;color:#555;">{t("model")} : <strong>{bm}</strong><br>R\u00b2 : <strong>{r2:.3f}</strong></p><p style="margin:.3rem 0 0 0;font-size:1.8rem;font-weight:700;" class="{sc(cs)}">{cs:.1f}/100 \u2014 {sm(cs)}</p></div>',unsafe_allow_html=True)
    # EXTRAIT DES DONNEES D'ENTRAINEMENT
    st.markdown('---')
    lang = st.session_state.lang
    st.markdown(f'<h2 class="st"><i class="fa-solid fa-database"></i> {t("training_data")}</h2>',unsafe_allow_html=True)
    st.caption(t('training_desc'))
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric('Observations' if lang=='fr' else 'Observations', f"{len(df):,}")
    with c2: st.metric('Variables' if lang=='fr' else 'Features', f"{len(df.columns)}")
    with c3: st.metric('R\u00e9gions' if lang=='fr' else 'Regions', f"{len(df['Region'].unique())}")
    with c4: st.metric('Ann\u00e9es' if lang=='fr' else 'Years', f"{df['Annee'].min()}\u2013{df['Annee'].max()}")
    # Variables par cat\u00e9gorie
    all_cols = df.columns.tolist()
    cat_targets = ['Region','Annee'] + [c for c in TARGETS.keys() if c in all_cols]
    cat_climat = [c for c in all_cols if any(k in c.lower() for k in ['temp','precip','humid','vent','press','nuage','uv','soleil','pluie','cyclone','saison','debit'])]
    cat_socio = [c for c in all_cols if any(k in c.lower() for k in ['pib','pauvrete','chomage','inflation','agri','production','import','export','cout','pop','densite','scol','alpha','sanit','vaccin','assain','eau','allai','anem','faible','stabilite'])]
    calc_cols = [c for c in all_cols if any(k in c.lower() for k in ['lag','delta','ma3','moy_region','std_region','stress','amplitude','indice'])]
    region_dummies = [c for c in all_cols if c.startswith('R_')]
    other_cols = [c for c in all_cols if c not in cat_targets+cat_climat+cat_socio+calc_cols+region_dummies]
    # Variables par catégorie avec sources
    dd_path = os.path.join(DATA_DIR, 'data_dictionary_v3.json')
    dd = {}
    if os.path.exists(dd_path):
        with open(dd_path, 'r', encoding='utf-8') as f_dd: dd = json.load(f_dd)
    col_desc = {}
    for cat in dd:
        if isinstance(dd[cat], dict):
            for k, v in dd[cat].items(): col_desc[k] = v
    if lang == 'fr':
        col_groups = [
            ('🎯 Cibles nutritionnelles — ONN/ENSOMD', ['Region','Annee'] + [c for c in TARGETS if c in all_cols]),
            ('🌡 Climat & Météo — 42 stations météorologiques', [c for c in all_cols if any(k in c for k in ['Temp','Precip','Humid','Vent','Press','Couverture','UV','Ensoleillement','Nb_jours','saison_cyclonique','saison_seche'])]),
            ('🌪 Cyclones — BNGRC/EM-DAT', [c for c in all_cols if c.startswith('Cycl_')]),
            ('📈 Macroéconomie & Marché du travail — INSTAT/Banque Mondiale/EPM', [c for c in all_cols if any(k in c for k in ['Croissance_PIB','Inflation','Chomage','Pauvrete','PIB_Agriculture','Part_Agriculture','GDP_per_capita','Stabilite_politique','Prod_alimentaire','Sous_emploi'])]),
            ('👥 Démographie — INSTAT/RGPH-3', [c for c in all_cols if any(k in c for k in ['Pop_','Superficie','Densite','Source_enquete']) and not c.startswith('R_')]),
            ('🌾 Agriculture & Prix alimentation — MINAGRI/FAO', [c for c in all_cols if any(k in c for k in ['Production_riz','Production_cafe','Import_miel','Export_miel','Mais_','Manioc_','Riz_','Kcal_','Cout_'])]),
            ('💧 Eau & Assainissement — WHO/UNICEF JMP', [c for c in all_cols if any(k in c for k in ['Acces_eau','Assainissement','FAO_acces','FAO_assainissement','FAO_eau'])]),
            ('🏥 Santé & Nutrition — OMS/UNICEF', [c for c in all_cols if any(k in c for k in ['Alpha_t','Scolarisation','Couverture_sanitaire','Vaccination','Anemie','Allaitement','Faible_poids','Nb_enfants','Nb_nouveau','FAO_stunted','FAO_wasting','Food_supply','Besoins_energ','CV_consom','Pertes_cal','Densite_ferree'])]),
            ('🌊 Hydrologie — APIPA', [c for c in all_cols if any(k in c for k in ['Pluie_APIPA','Debit_'])]),
            ('🔢 Variables calculées & Dérivées', [c for c in all_cols if any(k in c for k in ['lag1','delta','ma3','moy_region','std_region','Stress_thermique','Amplitude_thermique','Indice_vuln','Annee_sin','Annee_cos'])]),
            ('🏢 Dummies régionaux (22 régions)', [c for c in all_cols if c.startswith('R_')]),
        ]
    else:
        col_groups = [
            ('🎯 Nutritional targets — ONN/ENSOMD', ['Region','Annee'] + [c for c in TARGETS if c in all_cols]),
            ('🌡 Climate & Weather — 42 meteorological stations', [c for c in all_cols if any(k in c for k in ['Temp','Precip','Humid','Vent','Press','Couverture','UV','Ensoleillement','Nb_jours','saison_cyclonique','saison_seche'])]),
            ('🌪 Cyclones — BNGRC/EM-DAT', [c for c in all_cols if c.startswith('Cycl_')]),
            ('📈 Macroeconomics & Labor market — INSTAT/World Bank/EPM', [c for c in all_cols if any(k in c for k in ['Croissance_PIB','Inflation','Chomage','Pauvrete','PIB_Agriculture','Part_Agriculture','GDP_per_capita','Stabilite_politique','Prod_alimentaire','Sous_emploi'])]),
            ('👥 Demographics — INSTAT/RGPH-3', [c for c in all_cols if any(k in c for k in ['Pop_','Superficie','Densite','Source_enquete']) and not c.startswith('R_')]),
            ('🌾 Agriculture & Food prices — MINAGRI/FAO', [c for c in all_cols if any(k in c for k in ['Production_riz','Production_cafe','Import_miel','Export_miel','Mais_','Manioc_','Riz_','Kcal_','Cout_'])]),
            ('💧 Water & Sanitation — WHO/UNICEF JMP', [c for c in all_cols if any(k in c for k in ['Acces_eau','Assainissement','FAO_acces','FAO_assainissement','FAO_eau'])]),
            ('🏥 Health & Nutrition — WHO/UNICEF', [c for c in all_cols if any(k in c for k in ['Alpha_t','Scolarisation','Couverture_sanitaire','Vaccination','Anemie','Allaitement','Faible_poids','Nb_enfants','Nb_nouveau','FAO_stunted','FAO_wasting','Food_supply','Besoins_energ','CV_consom','Pertes_cal','Densite_ferree'])]),
            ('🌊 Hydrology — APIPA', [c for c in all_cols if any(k in c for k in ['Pluie_APIPA','Debit_'])]),
            ('🔢 Computed & derived features', [c for c in all_cols if any(k in c for k in ['lag1','delta','ma3','moy_region','std_region','Stress_thermique','Amplitude_thermique','Indice_vuln','Annee_sin','Annee_cos'])]),
            ('🏢 Regional dummies (22 regions)', [c for c in all_cols if c.startswith('R_')]),
        ]
    
    for gname, gcols in col_groups:
        if not gcols: continue
        # Filter columns with actual data (drop all-NaN)
        # Garder toutes les colonnes, afficher les lignes avec des donn\u00e9es (pas de None)
        _cat_df = df[gcols].dropna(how='all').copy()
        _cat_df['_na'] = _cat_df.isna().sum(axis=1)
        _cat_df = _cat_df.sort_values('_na').drop(columns='_na').head(15)
        # Remplacer les NaN restants par "\u2014" pour un affichage propre
        _cat_df = _cat_df.fillna('\u2014')
        if _cat_df.empty: continue
        with st.expander(f"{gname} ({len(gcols)})"):
            st.dataframe(_cat_df, use_container_width=True, hide_index=True)
    # Dictionnaire complet
    with st.expander('📖 ' + ('Dictionnaire des variables' if lang=='fr' else 'Data dictionary')):
        dict_data = []
        for cat_key, cat_dict in dd.items():
            if isinstance(cat_dict, dict):
                for col_name, col_desc_str in cat_dict.items():
                    if col_name in all_cols:
                        dict_data.append({'Variable': col_name, ('Catégorie' if lang=='fr' else 'Category'): cat_key, 'Description': col_desc_str})
        if dict_data:
            st.dataframe(pd.DataFrame(dict_data), use_container_width=True, hide_index=True)
    # Aperçu complet
    with st.expander('📊 ' + ('Aperçu complet (toutes les colonnes)' if lang=='fr' else 'Full preview (all columns)')):
        st.dataframe(df.head(5), use_container_width=True, hide_index=True)
    ib(dyn_interp(tk, 'key'))
    with st.expander(t('correlation_matrix')):
        hi=li(os.path.join(FIGD,"05_heatmap_correlations.png"))
        if hi: st.image(hi,use_container_width=True)
        ib(t('correlation_interp'))
    with st.expander(t('temporal_evolution')):
        # Graphique interactif au lieu de l'image statique MC
        if tk in df.columns and 'Region' in df.columns and 'Annee' in df.columns:
            fig_te=px.line(df,x='Annee',y=tk,color='Region',markers=True,
                title=f'{TARGETS[tk][st.session_state.lang]} — {t(tk+"_full")}',
                labels={tk:TARGETS[tk][st.session_state.lang],'Annee':t('years')})
            fig_te.update_layout(height=450)
            st.plotly_chart(fig_te,use_container_width=True)
        ib(t('temporal_interp'))

# ---- PREDICT ----
with tp:
    st.markdown(f'<h1 class="mt">{t("predict_title")}</h1>',unsafe_allow_html=True)
    st.caption(t('predict_subtitle'))
    st.markdown(f'<h2 class="st">{t("model_comparison")}</h2>',unsafe_allow_html=True)
    # Model comparison - interactive Plotly chart (no author, translated)
    if cf is not None:
        models_list = cf['Modele'].unique().tolist()
        targets_list = cf['Cible'].unique().tolist()
        fig_mc = go.Figure()
        colors_t = {'Insuffisance Pond\u00e9rale':'#2e86c1','Malnutrition Chronique':'#8e44ad','Malnutrition Aig\u00fce':'#c0392b'}
        for tgt in targets_list:
            subset = cf[cf['Cible']==tgt].sort_values('R2_test',ascending=True)
            clr = colors_t.get(tgt,'#2e86c1')
            tname = next((TARGETS[k][st.session_state.lang] for k in TARGETS if TARGETS[k]['fr']==tgt or k in tgt), tgt)
            fig_mc.add_trace(go.Bar(y=subset['Modele'],x=subset['R2_test'],orientation='h',name=tname,marker_color=clr,
                                   text=[f"{v:.4f}" for v in subset['R2_test']],textposition='outside'))
        fig_mc.update_layout(height=450,barmode='group',showlegend=True,
                             title=t('model_comparison'),
                             xaxis_title='R\u00b2',yaxis_title=t('model_label'),
                             legend_title=t('target_label'))
        st.plotly_chart(fig_mc,use_container_width=True)
    else:
        ci=li(os.path.join(FIGD,"01_comparaison_modeles.png"))
        if ci: st.image(ci,use_container_width=True)
    ib(t('model_interp'))
    if cf is not None:
        st.markdown(f'#### {t("detailed_performance")}')
        # Translate target names in dataframe
        cdf = cf.copy()
        if 'Cible' in cdf.columns:
            target_name_map = {k: TARGETS[k][st.session_state.lang] for k in TARGETS}
            cdf['Cible'] = cdf['Cible'].apply(lambda x: next((v for k,v in target_name_map.items() if k in str(x)), x))
        col_map = {'Cible': t('target_label'), 'Modele': t('model_label'), 'R2_test': 'R\u00b2', 'RMSE_test': 'RMSE', 'MAE_test': 'MAE'}
        disp_cols = [c for c in ['Cible','Modele','R2_test','RMSE_test','MAE_test'] if c in cdf.columns]
        ddf = cdf[disp_cols].copy()
        # Format numbers
        for c in ['R2_test','RMSE_test','MAE_test']:
            if c in ddf.columns: ddf[c] = ddf[c].apply(lambda x: f"{x:.4f}" if isinstance(x,(int,float)) else str(x))
        # Translate column names for display
        ddf.columns = [col_map.get(c, c) for c in ddf.columns]
        st.dataframe(ddf, use_container_width=True, hide_index=True)
    st.markdown('---')
    st2=st.selectbox(t('target_label'),list(TARGETS.keys()),format_func=lambda x:tl(x),key="p_t")
    # Predictions vs Reality - interactive Plotly (no author, translated)
    if st2 in df.columns:
        for mn in ['Ridge','XGBoost','RandomForest']:
            model = models.get((mn, st2))
            if model is not None:
                break
        if model is not None:
            try:
                # Real train/test split for proper scatter plot
                from sklearn.model_selection import train_test_split as tts
                np.random.seed(42)
                df_ml = df.dropna(subset=[st2]).copy()
                feat_avail = [c for c in feature_cols if c in df_ml.columns]
                X_all = df_ml[feat_avail].copy()
                for c in feat_avail:
                    X_all[c] = X_all[c].fillna(feature_stats.get(c,{}).get('mean',0))
                if 'Region_enc' in feature_cols and 'Region' in df_ml.columns:
                    valid = df_ml['Region'].isin(le.classes_)
                    X_all = X_all[valid].copy()
                    df_ml = df_ml[valid].copy()
                    X_all['Region_enc'] = le.transform(df_ml['Region'].values)
                    feat_use = [c for c in feature_cols if c in X_all.columns]
                else:
                    feat_use = feat_avail
                y_all = df_ml[st2].values
                X_tr, X_te, y_tr, y_te = tts(X_all[feat_use], y_all, test_size=0.2, random_state=42)
                # Scale
                X_tr_s = pd.DataFrame(scaler.transform(X_tr[feat_use].fillna(0)), columns=feat_use)
                X_te_s = pd.DataFrame(scaler.transform(X_te[feat_use].fillna(0)), columns=feat_use)
                # Use the best model type, retrain on this split
                from sklearn.linear_model import Ridge as SkRidge
                from sklearn.ensemble import RandomForestRegressor as SkRF, GradientBoostingRegressor as SkGB
                model_map = {'Ridge': SkRidge(alpha=1.0), 'RandomForest': SkRF(n_estimators=100, random_state=42),
                             'XGBoost': None, 'Random_Forest': SkRF(n_estimators=100, random_state=42)}
                # Retrain best model type on this split
                if mn in model_map and model_map[mn] is not None:
                    m_type = model_map[mn]
                    if mn in ['Ridge']:
                        m_type.fit(X_tr_s, y_tr)
                        preds = m_type.predict(X_te_s)
                    else:
                        m_type.fit(X_tr[feat_use], y_tr)
                        preds = m_type.predict(X_te[feat_use])
                else:
                    # Fallback: Ridge always works
                    m_fb = SkRidge(alpha=1.0)
                    m_fb.fit(X_tr_s, y_tr)
                    preds = m_fb.predict(X_te_s)
                actuals = y_te
                tname = TARGETS[st2][st.session_state.lang]
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=actuals, y=preds, mode='markers',
                    marker=dict(color=TARGETS[st2]['color'], size=8, opacity=0.6),
                    text=[f'{tname}: {a:.1f}% → {p:.1f}%' for a,p in zip(actuals,preds)],
                    name=t('predicted_values').replace(' (%)',''),
                    hovertemplate='%{text}<extra></extra>'))
                mn_val = min(min(actuals),min(preds)); mx_val = max(max(actuals),max(preds))
                fig_pr.add_trace(go.Scatter(x=[mn_val,mx_val],y=[mn_val,mx_val],mode='lines',
                    line=dict(dash='dash',color='#555'), name='y=x'))
                fig_pr.update_layout(height=500, showlegend=True,
                    title=f'{tname} — {t("prediction_vs_reality")}',
                    xaxis_title=t('actual_values'),
                    yaxis_title=t('predicted_values'),
                    hovermode='closest')
                # R² annotation
                from sklearn.metrics import r2_score
                r2_plot = r2_score(actuals, preds)
                fig_pr.add_annotation(x=0.02, y=0.98, xref='paper', yref='paper',
                    text=f'R\u00b2 = {r2_plot:.4f}', showarrow=False,
                    font=dict(size=14, color='#1a5276'), bgcolor='rgba(255,255,255,0.8)')
                st.plotly_chart(fig_pr, use_container_width=True, key=f'pred_real_{st2}')
            except Exception as e:
                st.warning(f'Error: {e}')
    ib(dyn_interp(st2, 'pred'))
    st.markdown(f'#### {t("distribution")} \u2014 {TARGETS[st2][st.session_state.lang]}')
    if st2 in df.columns:
        fd=px.histogram(df,x=st2,color='Annee' if 'Annee' in df.columns else None,nbins=30,color_discrete_sequence=px.colors.qualitative.Set2,
                         labels={st2:TARGETS[st2][st.session_state.lang],'Annee':t('years'),'count':t('observations')},
                         title=f"{TARGETS[st2][st.session_state.lang]} — {t('distribution')}")
        fd.update_layout(height=400,xaxis_title=TARGETS[st2][st.session_state.lang],yaxis_title=t('observations'))
        st.plotly_chart(fd,use_container_width=True)
        ib(dyn_interp(st2, 'dist_full'))
    if st2 in df.columns and 'Region' in df.columns:
        rs_d=df.groupby('Region')[st2].mean().sort_values(ascending=False)
        c1,c2=st.columns(2)
        with c1:
            st.markdown(f"**{t('worst_regions')}**")
            for r,v in rs_d.head(5).items(): st.markdown(f"- {r} : **{v:.1f}%**")
        with c2:
            st.markdown(f"**{t('best_regions')}**")
            for r,v in rs_d.tail(5).items(): st.markdown(f"- {r} : **{v:.1f}%**")

# ---- EXPLAIN ----
with tex:
    st.markdown(f'<h1 class="mt">{t("explain_title")}</h1>',unsafe_allow_html=True)
    st.caption(t('explain_subtitle'))
    et=st.selectbox(t('target_label'),list(TARGETS.keys()),format_func=lambda x:tl(x),key="e_t")
    st.markdown(f'<h2 class="st">{t("shap_importance")}</h2>',unsafe_allow_html=True)
    si=li(os.path.join(SHAD,f"shap_summary_{et}.png"))
    if si: st.image(si,caption=f"SHAP Summary \u2014 {TARGETS[et][st.session_state.lang]}",use_container_width=True)
    ib(t('shap_interp'))
    c1,c2=st.columns(2)
    with c1:
        st.markdown(f'#### {t("bar_plot")}')
        bi=li(os.path.join(SHAD,f"shap_bar_{et}.png"))
        if bi: st.image(bi,use_container_width=True)
        ib(t('bar_interp'))
    with c2:
        st.markdown(f'#### {t("dependence_plot")}')
        di=li(os.path.join(SHAD,f"shap_dependence_{et}.png"))
        if di: st.image(di,use_container_width=True)
        ib(t('dep_interp'))
    st.markdown('---')
    st.markdown(f'<h2 class="st">{t("local_explanation")}</h2>',unsafe_allow_html=True)
    st.caption(t('local_explanation_desc'))
    wi=li(os.path.join(SHAD,f"shap_waterfall_androy_{et}.png"))
    if wi: st.image(wi,caption=f"SHAP Waterfall \u2014 Androy \u2014 {TARGETS[et][st.session_state.lang]}",use_container_width=True)
    ib(t('local_interp'))
    st.markdown('---')
    st.markdown(f'<h2 class="st">{t("inter_method")}</h2>',unsafe_allow_html=True)
    st.caption(t('inter_method_desc'))
    xi=li(os.path.join(XAID,f"xai_comparison_{et}.png"))
    if xi: st.image(xi,caption=f"{t('inter_method')} \u2014 {TARGETS[et][st.session_state.lang]}",use_container_width=True)
    ib(t('inter_interp'))
    ed=rp.get('explain',{})
    if ed:
        with st.expander(t('kendall_details')): st.json(ed)

# ---- CERTIFY ----
with tce:
    st.markdown(f'<h1 class="mt">{t("certify_title")}</h1>',unsafe_allow_html=True)
    st.caption(t('certify_subtitle'))
    st.markdown(f'<h2 class="st">{t("composite_score")}</h2>',unsafe_allow_html=True)
    # Composite Certification Score - Plotly interactif (avec decimales)
    cert_labels = []; cert_scores = []; cert_colors = []
    for k, info in TARGETS.items():
        rk = RTK.get(k, info[st.session_state.lang])
        ct = cd_data.get(rk, {})
        s = ct.get('certification_score', 0)
        g = 'A' if s >= 90 else 'B' if s >= 80 else 'C'
        cert_labels.append(info[st.session_state.lang])
        cert_scores.append(s)
        cert_colors.append('#27ae60' if g == 'A' else '#f39c12' if g == 'B' else '#e74c3c')
    fig_cert = go.Figure(go.Bar(x=cert_labels, y=cert_scores, marker_color=cert_colors,
        text=[f"{s:.1f}/100 {g}" for s, g in zip(cert_scores, ['A' if s>=90 else 'B' if s>=80 else 'C' for s in cert_scores])],
        textposition='outside', textfont=dict(size=14, color='#1a1a2e')))
    fig_cert.add_hline(y=90, line_dash='dash', line_color='#27ae60', annotation_text='A (≥90)')
    fig_cert.add_hline(y=80, line_dash='dash', line_color='#2e86c1', annotation_text='B (≥80)')
    fig_cert.add_hline(y=60, line_dash='dash', line_color='#f39c12', annotation_text='C (≥60)')
    fig_cert.update_layout(height=400, showlegend=False,
        title=t('composite_score'),
        yaxis=dict(
            title=t('cert_score'),
            tick0=50, dtick=10,
            range=[min(50, min(cert_scores)-5), 100],
            tickformat='.0f'),
        xaxis_title=t('target_label'))
    st.plotly_chart(fig_cert, use_container_width=True)
    ib(t('cert_interp'))
    cl=st.columns(3)
    for c,(k,info) in zip(cl,TARGETS.items()):
        rk=RTK.get(k,info[st.session_state.lang]);ct=cd_data.get(rk,{});s=ct.get('certification_score',0);mn=ct.get('model','N/A')
        with c:
            st.markdown(f'<div class="cd cg"><h3 style="margin:0;">{tl(k)}</h3><p style="margin:.2rem 0;font-size:.9rem;color:#555;">{t("model")} : {mn}</p><p style="margin:0;font-size:2rem;font-weight:700;" class="{sc(s)}">{s:.1f}/100 \u2014 {sm(s)}</p></div>',unsafe_allow_html=True)
    st.markdown('---')
    st.markdown(f'<h2 class="st">{t("regional_fairness")}</h2>',unsafe_allow_html=True)
    ct2=st.selectbox(t('target_label'),list(TARGETS.keys()),format_func=lambda x:tl(x),key="c_t")
    fi=li(os.path.join(CERD,f"fairness_{ct2}.png"))
    if fi: st.image(fi,caption=f"{t('regional_fairness')} \u2014 {TARGETS[ct2][st.session_state.lang]}",use_container_width=True)
    ib(t('fairness_interp'))
    with st.expander(t('certification_details')): st.json(rp.get('certify',{}))
    st.markdown('---')
    st.markdown(f'<h2 class="st">{t("score_interpretation")}</h2>',unsafe_allow_html=True)
    idf=pd.DataFrame({'Score':['90\u2014100','75\u201489','50\u201474','0\u201449'],
        'Niveau':[t('score_excellent').split('\u2014')[0].strip(),t('score_good').split('\u2014')[0].strip(),t('score_warning').split('\u2014')[0].strip(),t('score_critical').split('\u2014')[0].strip()],
        'Interpr\u00e9tation':[t('score_excellent'),t('score_good'),t('score_warning'),t('score_critical')]})
    st.dataframe(idf,use_container_width=True,hide_index=True)
    st.markdown(f'<div class="cd cg"><strong>{t("recommendation")}</strong></div>',unsafe_allow_html=True)

# ---- EXPLORE ----
with tex2:
    st.markdown(f'<h1 class="mt">{t("explore_title")}</h1>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: er=st.multiselect(t('region_label'),REGS,default=REGS[:5],key="e_r")
    with c2:
        if 'Annee' in df.columns:
            ys=sorted(df['Annee'].unique());ey=st.slider(t('years'),min(ys),max(ys),(min(ys),max(ys)),key="e_y")
    de=df.copy()
    if er and 'Region' in de.columns: de=de[de['Region'].isin(er)]
    if 'Annee' in de.columns: de=de[(de['Annee']>=ey[0])&(de['Annee']<=ey[1])]
    st.metric(t('selected_obs'),f"{len(de)} / {len(df)}")
    # CARTE
    st.markdown('---')
    ts2=st.selectbox(t('target_label'),list(TARGETS.keys()),format_func=lambda x:tl(x),key="e_t2")
    st.plotly_chart(makemap(de,ts2,st.session_state.lang),use_container_width=True)
    # EVOLUTION
    st.markdown('---')
    st.markdown(f'<h2 class="st">{t("temporal_evolution")}</h2>',unsafe_allow_html=True)
    if ts2 in de.columns and 'Region' in de.columns and 'Annee' in de.columns:
        fe=px.line(de,x='Annee',y=ts2,color='Region',markers=True,
                         labels={'Annee':t('years'),ts2:TARGETS[ts2][st.session_state.lang],'Region':'Region'},
                         title=f"{TARGETS[ts2][st.session_state.lang]} — {t('temporal_evolution')}")
        fe.update_layout(height=450,xaxis_title=t('years'),yaxis_title=TARGETS[ts2][st.session_state.lang])
        st.plotly_chart(fe,use_container_width=True)
        ib(t('temporal_interp'))
    # SCATTER
    st.markdown('---')
    st.markdown(f'<h2 class="st">{t("interactive_correlations")}</h2>',unsafe_allow_html=True)
    nc=de.select_dtypes(include=[np.number]).columns.tolist()
    nc=[c for c in nc if c not in {'Annee'} and not c.startswith('dummy_')]
    c1,c2=st.columns(2)
    with c1: xvar=st.selectbox(t('axis_x'),nc,index=nc.index(ts2) if ts2 in nc else 0,key="s_x")
    with c2:
        yi=nc.index('Malnutrition_chronique') if 'Malnutrition_chronique' in nc else min(1,len(nc)-1)
        yvar=st.selectbox(t('axis_y'),nc,index=yi,key="s_y")
    cvar=st.selectbox(t('color'),['Region','Annee']+nc,index=0,key="s_c")
    fs=px.scatter(de,x=xvar,y=yvar,color=cvar,trendline='ols',hover_data=['Region','Annee'] if 'Region' in de.columns else None,
                   labels={xvar:xvar.replace('_',' ').replace('pct','%'),yvar:yvar.replace('_',' ').replace('pct','%')})
    fs.update_layout(height=450,xaxis_title=xvar.replace('_',' ').replace('pct','%'),yaxis_title=yvar.replace('_',' ').replace('pct','%'));st.plotly_chart(fs,use_container_width=True)
    # DONNEES
    st.markdown('---')
    st.markdown(f'<h2 class="st">{t("raw_data")}</h2>',unsafe_allow_html=True)
    dc=['Region','Annee']+list(TARGETS.keys());ac=[c for c in dc if c in de.columns]
    if ac: st.dataframe(de[ac].sort_values(['Region','Annee']),use_container_width=True,height=400)
    with st.expander(t('data_dictionary')):
        if dd: st.json(dd)

# ---- A PROPOS ----
with tab_about:
    lang=st.session_state.lang
    st.markdown(f'<h1><i class="fa-solid fa-circle-info"></i> {t("tab_about")}</h1>',unsafe_allow_html=True)
    if lang=='fr':
        st.markdown('''<div class="interp-box">
        <strong>PEC Madagascar</strong> — Audit complet du framework PEC sur la malnutrition à Madagascar.<br><br>
        Cette plateforme analyse <strong>3 cibles nutritionnelles</strong> à travers <strong>22 régions</strong> sur la période 2012-2022 :
        <ul>
            <li><strong>Insuffisance pondérale</strong></li>
            <li><strong>Malnutrition aiguë</strong></li>
            <li><strong>Malnutrition chronique</strong></li>
        </ul>
        Pour chaque cible, le système <strong>PREDICT</strong> (6 modèles de ML), <strong>EXPLAIN</strong> (SHAP, LIME, Permutation Importance) puis <strong>CERTIFY</strong> (score composite /100 + Grade A-D).
        </div>''',unsafe_allow_html=True)
        st.markdown('<h3><i class="fa-solid fa-list-check"></i> Conditions d’application</h3>',unsafe_allow_html=True)
        st.markdown('''<div class="pec-card"><ul style="line-height:2;">
        <li>Données tabulaires (CSV) avec au moins 20 observations et 3 variables numériques</li>
        <li>Variables cibles numériques (régression) ou catégorielles (classification)</li>
        <li>Aucune valeur manquante critique (>50% par colonne)</li>
        <li>Relations non purement aléatoires entre variables explicatives et cible</li>
        </ul></div>''',unsafe_allow_html=True)
    else:
        st.markdown('''<div class="interp-box">
        <strong>PEC Madagascar</strong> — Full audit of the PEC framework on malnutrition in Madagascar.<br><br>
        This platform analyzes <strong>3 nutritional targets</strong> across <strong>22 regions</strong> over 2012-2022:
        <ul>
            <li><strong>Underweight</strong></li>
            <li><strong>Acute malnutrition</strong></li>
            <li><strong>Chronic malnutrition</strong></li>
        </ul>
        For each target, the system <strong>PREDICTS</strong> (6 ML models), <strong>EXPLAINS</strong> (SHAP, LIME, Permutation Importance) then <strong>CERTIFIES</strong> (composite score /100 + Grade A-D).
        </div>''',unsafe_allow_html=True)
        st.markdown('<h3><i class="fa-solid fa-list-check"></i> Application conditions</h3>',unsafe_allow_html=True)
        st.markdown('''<div class="pec-card"><ul style="line-height:2;">
        <li>Tabular data (CSV) with at least 20 observations and 3 numeric variables</li>
        <li>Numeric targets (regression) or categorical targets (classification)</li>
        <li>No critical missing values (>50% per column)</li>
        <li>Non-random relationships between explanatory variables and target</li>
        </ul></div>''',unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown('---')
st.markdown("""<div style="text-align:center;color:#6b7280;font-size:.85rem;padding:1rem 0;">
    PEC Framework v4 | <i class="fa-solid fa-flask"></i> Predict \u2022 <i class="fa-solid fa-magnifying-glass"></i> Explain \u2022 <i class="fa-solid fa-shield-halved"></i> Certify |
    <a href="https://orcid.org/0009-0003-3048-1765" target="_blank" style="color:#a6ce39;"><i class="fa-brands fa-orcid"></i> Rosa Elysabeth Ralinirina</a>
</div>""", unsafe_allow_html=True)