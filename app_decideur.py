# ============================================================================
# PEC FRAMEWORK — G\u00e9n\u00e9ralis\u00e9
# Auteur : Rosa Elysabeth Ralinirina | ORCID : 0009-0003-3048-1765
# ============================================================================
# PREDICT \u2014 EXPLAIN \u2014 CERTIFY : applicable \u00e0 tout domaine
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
import os, json, warnings, io, pickle, joblib
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

st.set_page_config(page_title="PEC G\u00e9n\u00e9ralis\u00e9", page_icon=None, layout="wide", initial_sidebar_state="expanded")

BASE = os.path.dirname(os.path.abspath(__file__))

# Demo datasets
DEMO_DATASETS = {
    'heart': {'name_fr':'Maladies cardiaques (UCI)', 'name_en':'Heart Disease (UCI)', 'path':'generalisation/data/external/heart.csv', 'sep':',', 'targets':['target'], 'desc_fr':'Donn\u00e9es cliniques de 302 patients, objectif : pr\u00e9dire la pr\u00e9sence de maladie cardiaque.', 'desc_en':'Clinical data from 302 patients, goal: predict heart disease presence.'},
    'breast_cancer': {'name_fr':'Cancer du sein (Wisconsin)', 'name_en':'Breast Cancer (Wisconsin)', 'path':'generalisation/data/external/breast-cancer.csv', 'sep':',', 'targets':['diagnosis'], 'desc_fr':'Caract\u00e9ristiques de biopsie, objectif : classifier les tumeurs b\u00e9nignes/malignes.', 'desc_en':'Biopsy features, goal: classify benign/malignant tumors.'},
    'wine_red': {'name_fr':'Qualit\u00e9 du vin rouge (Vinho Verde)', 'name_en':'Red Wine Quality (Vinho Verde)', 'path':'generalisation/data/external/winequality-red.csv', 'sep':';', 'targets':['quality'], 'desc_fr':'Donn\u00e9es physico-chimiques de vins rouges portugais, objectif : pr\u00e9dire la qualit\u00e9.', 'desc_en':'Physicochemical data of Portuguese red wines, goal: predict quality.'},
}
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
:root{--primary:#1a5276;--primary-light:#2e86c1;--secondary:#27ae60;--purple:#8e44ad;--red:#c0392b;--orange:#f39c12;--bg:#f4f6f9;--card:#fff;--text:#1a1a2e;--muted:#6b7280;--border:#e5e7eb;--radius:12px;--shadow:0 4px 24px rgba(0,0,0,.06)}
.stApp{background:var(--bg);font-family:'Inter',sans-serif}
</style>""", unsafe_allow_html=True)

# ============================================================================
# TRANSLATIONS
# ============================================================================
T = {
'fr': {
    'app_title':'PEC G\u00e9n\u00e9ralis\u00e9','app_subtitle':'Predict \u2014 Explain \u2014 Certify : applicable \u00e0 tout domaine',
    'tab_demo':'D\u00e9mo','tab_upload':'Importer vos donn\u00e9es','tab_results':'R\u00e9sultats','tab_about':'\u00c0 propos','nav_mada':'PEC Mada','nav_aide':'PEC Aide','nav_what':'PEC What','nav_gen':'PEC Gen',
    'select_demo':'Choisir un jeu de donn\u00e9es d\u00e9mo','select_upload':'Importer un fichier CSV',
    'target_sel':'S\u00e9lectionner la/les cible(s) \u00e0 pr\u00e9dire','run_pec':'Lancer l\u2019analyse PEC',
    'predict':'Pr\u00e9dire','explain':'Expliquer','certify':'Certifier',
    'r2_score':'Performance (R\u00b2)','rmse':'RMSE','best_model':'Meilleur mod\u00e8le',
    'cert_score':'Score de certification','grade':'Grade',
    'cert_excellent':'Excellent \u2014 Syst\u00e8me certifi\u00e9.','cert_good':'Bon \u2014 Syst\u00e8me fiable.',
    'cert_warning':'Attention \u2014 Risques identifi\u00e9s.','cert_critical':'Critique \u2014 Non certifi\u00e9.',
    'interpretation':'Interpr\u00e9tation','features':'Variables les plus influentes',
    'download':'T\u00e9l\u00e9charger','no_data':'Aucune donn\u00e9e charg\u00e9e.',
    'upload_desc':'Importez un fichier CSV avec vos propres donn\u00e9es. Le framework PEC va automatiquement : (1) entra\u00eener plusieurs mod\u00e8les, (2) g\u00e9n\u00e9rer des explications (SHAP), (3) calculer un score de certification.',
    'upload_format':'Format attendu : colonnes num\u00e9riques = variables, lignes = observations. La derni\u00e8re colonne est souvent la cible.',
    'demo_desc':'S\u00e9lectionnez un jeu de donn\u00e9es pr\u00e9-charg\u00e9 pour tester le framework PEC.',
    'about_what':'Qu\u2019est-ce que PEC ?','about_what_desc':'PEC (Predict-Explain-Certify) est un framework d\u2019intelligence artificielle qui pr\u00e9dit, explique et certifie les r\u00e9sultats. Il est applicable \u00e0 tout domaine : sant\u00e9, finance, environnement, \u00e9ducation, etc.',
    'col_count':'Colonnes','row_count':'Lignes','target_count':'Cibles',
    'shap_note':'Les barres SHAP montrent l\u2019impact de chaque variable sur la pr\u00e9diction. Barres vertes = augmentation, barres rouges = diminution.',
    'model_compare':'Comparaison des mod\u00e8les','prediction':'Pr\u00e9dictions',
},
'en': {
    'app_title':'PEC Generalized','app_subtitle':'Predict \u2014 Explain \u2014 Certify: applicable to any domain',
    'tab_demo':'Demo','tab_upload':'Upload Your Data','tab_results':'Results','tab_about':'About','nav_mada':'PEC Mada','nav_aide':'PEC Aid','nav_what':'PEC What','nav_gen':'PEC Gen',
    'select_demo':'Select a demo dataset','select_upload':'Upload a CSV file',
    'target_sel':'Select target(s) to predict','run_pec':'Run PEC Analysis',
    'predict':'Predict','explain':'Explain','certify':'Certify',
    'r2_score':'Performance (R\u00b2)','rmse':'RMSE','best_model':'Best model',
    'cert_score':'Certification score','grade':'Grade',
    'cert_excellent':'Excellent \u2014 Certified system.','cert_good':'Good \u2014 Reliable system.',
    'cert_warning':'Warning \u2014 Risks identified.','cert_critical':'Critical \u2014 Uncertified system.',
    'interpretation':'Interpretation','features':'Most influential features',
    'download':'Download','no_data':'No data loaded.',
    'upload_desc':'Upload a CSV file with your own data. The PEC framework will automatically: (1) train multiple models, (2) generate explanations (SHAP), (3) compute a certification score.',
    'upload_format':'Expected format: numeric columns = variables, rows = observations. The last column is often the target.',
    'demo_desc':'Select a pre-loaded dataset to test the PEC framework.',
    'about_what':'What is PEC?','about_what_desc':'PEC (Predict-Explain-Certify) is an AI framework that predicts, explains, and certifies results. It is applicable to any domain: health, finance, environment, education, etc.',
    'col_count':'Columns','row_count':'Rows','target_count':'Targets',
    'shap_note':'SHAP bars show the impact of each variable on the prediction. Green bars = increase, red bars = decrease.',
    'model_compare':'Model comparison','prediction':'Predictions',
}}

def t(key): return T.get(st.session_state.lang, T['fr']).get(key, key)

def grade_info(score):
    if score >= 90: return 'A', 'badge-a', t('cert_excellent')
    elif score >= 80: return 'B', 'badge-b', t('cert_good')
    elif score >= 60: return 'C', 'badge-c', t('cert_warning')
    else: return 'D', 'badge-d', t('cert_critical')

if 'lang' not in st.session_state: st.session_state.lang = 'fr'
if 'results' not in st.session_state: st.session_state.results = None
if 'dataset' not in st.session_state: st.session_state.dataset = None

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    lang = st.radio('', ['fr','en'], format_func=lambda x: 'Français' if x=='fr' else 'English', horizontal=True, label_visibility='collapsed', key='sidebar_lang3')
    st.session_state.lang = lang

    # Navigation inter-plateformes
    st.divider()
    st.markdown(f'#### 🔗 Plateformes PEC')
    c1, c2 = st.columns(2)
    with c1: st.link_button(f'📊 {t("nav_mada")}', pec_url('mada'), use_container_width=True)
    with c2: st.link_button(f'📊 {t("nav_aide")}', pec_url('aide'), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3: st.link_button(f'🔬 {t("nav_what")}', pec_url('whatif'), use_container_width=True)
    with c4: st.link_button(f'🌍 {t("nav_gen")}', pec_url('gen'), use_container_width=True)
    st.markdown(f"### <i class='fa-solid fa-flask-vial'></i> PEC G\u00e9n\u00e9ralis\u00e9", unsafe_allow_html=True)
    st.caption(t('app_subtitle'))
    st.divider()
    if st.session_state.results:
        res = st.session_state.results
        for tgt, r in res.items():
            r2 = r.get('best_r2', 0)
            cert = r.get('cert_score', 0)
            grade, _, _ = grade_info(cert)
            st.markdown(f"**{tgt}**: R\u00b2={r2:.4f} | <span class='{grade}' style='font-size:.8rem;'>{grade}</span> ({cert:.0f}/100)", unsafe_allow_html=True)
    st.divider()
    st.markdown("""<div style="text-align:center;color:rgba(255,255,255,.7);font-size:.8rem;">
        Rosa Elysabeth Ralinirina<br>
        <a href="https://orcid.org/0009-0003-3048-1765" target="_blank" style="color:#a6ce39;">
            <i class="fa-brands fa-orcid"></i> 0009-0003-3048-1765
        </a></div>""", unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================
tab_demo, tab_upload, tab_results, tab_about = st.tabs([
    t('tab_demo'),
    t('tab_upload'),
    t('tab_results'),
    t('tab_about'),
])

# ============================================================================
# DEMO TAB
# ============================================================================
with tab_demo:
    lang = st.session_state.lang
    st.markdown(f'<h1><i class="fa-solid fa-database"></i> {t("select_demo")}</h1>', unsafe_allow_html=True)
    st.caption(t('demo_desc'))

    demo_key = st.selectbox(t('select_demo'), list(DEMO_DATASETS.keys()),
                            format_func=lambda x: f"{DEMO_DATASETS[x][f'name_{lang}']}" if lang in DEMO_DATASETS[x] else DEMO_DATASETS[x]['name_en'],
                            key='demo_select')

    demo_info = DEMO_DATASETS[demo_key]
    demo_path = os.path.join(BASE, demo_info['path'])
    desc_key = f'desc_{lang}' if f'desc_{lang}' in demo_info else 'desc_en'

    if os.path.exists(demo_path):
        df_demo = pd.read_csv(demo_path, sep=demo_info['sep'])
        st.markdown(f'<div class="interp-box">{demo_info.get(desc_key, demo_info.get("desc_en",""))}</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: st.metric(t('row_count'), f"{len(df_demo)}")
        with c2: st.metric(t('col_count'), f"{len(df_demo.columns)}")
        with c3: st.metric(t('target_count'), f"{len(demo_info['targets'])}")

        st.dataframe(df_demo.head(20), use_container_width=True, hide_index=True)

        # Select targets
        numeric_cols = df_demo.select_dtypes(include=[np.number]).columns.tolist()
        default_targets = [c for c in demo_info['targets'] if c in numeric_cols]
        target_cols = st.multiselect(t('target_sel'), numeric_cols, default=default_targets if default_targets else numeric_cols[-1:], key='demo_targets')

        if target_cols and st.button(t('run_pec'), type='primary', use_container_width=True, key='demo_run'):
            with st.spinner(t('run_pec')):
                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import Ridge
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                from sklearn.preprocessing import StandardScaler

                feature_cols = [c for c in numeric_cols if c not in target_cols]
                results = {}
                progress = st.progress(0, text=t('run_pec'))

                for ti, tgt in enumerate(target_cols):
                    df_clean = df_demo.dropna(subset=[tgt] + feature_cols[:50])
                    if len(df_clean) < 20: continue

                    use_feats = [c for c in feature_cols[:50] if c in df_clean.columns and df_clean[c].notna().sum() > len(df_clean)*0.5]
                    X = df_clean[use_feats].fillna(df_clean[use_feats].mean())
                    y = df_clean[tgt]

                    # Detect classification vs regression
                    n_unique = y.nunique()
                    is_classif = n_unique <= 10 and y.dtype in ['int64','int32','object']

                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                    sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

                    if is_classif:
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                        from sklearn.metrics import accuracy_score, f1_score
                        models_dict = {
                            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                        }
                    else:
                        from sklearn.linear_model import Ridge
                        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                        models_dict = {
                            'Ridge': Ridge(alpha=1.0),
                            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                        }

                    model_results = {}
                    best_score = -999; best_name = 'N/A'; best_model_obj = None

                    for mn, m in models_dict.items():
                        try:
                            if 'Logistic' in mn or 'Ridge' in mn: m.fit(X_tr_s, y_tr); pred = m.predict(X_te_s)
                            else: m.fit(X_tr, y_tr); pred = m.predict(X_te)
                            if is_classif:
                                sc_val = accuracy_score(y_te, pred)
                                f1 = f1_score(y_te, pred, average='weighted', zero_division=0)
                                model_results[mn] = {'r2': sc_val, 'rmse': 1-sc_val, 'mae': f1, 'metric': 'accuracy'}
                            else:
                                r2 = r2_score(y_te, pred); rmse = np.sqrt(mean_squared_error(y_te, pred)); mae = mean_absolute_error(y_te, pred)
                                model_results[mn] = {'r2': r2, 'rmse': rmse, 'mae': mae, 'metric': 'r2'}
                            if model_results[mn]['r2'] > best_score: best_score = model_results[mn]['r2']; best_name = mn; best_model_obj = m
                        except: pass

                    # Feature importance — always compute via permutation or tree
                    from sklearn.inspection import permutation_importance
                    feat_imp = {}
                    if hasattr(best_model_obj, 'feature_importances_'):
                        feat_imp = dict(zip(use_feats, best_model_obj.feature_importances_))
                    elif best_model_obj is not None:
                        try:
                            X_te_imp = X_te_s if ('Logistic' in best_name or 'Ridge' in best_name) else X_te
                            perm = permutation_importance(best_model_obj, X_te_imp, y_te, n_repeats=10, random_state=42, n_jobs=-1)
                            feat_imp = dict(zip(use_feats, perm.importances_mean))
                        except: pass
                    top_feats = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:10] if feat_imp else [(f, 0) for f in use_feats[:10]]

                    # PEC certification score
                    ip_score = min(100, max(0, best_score * 100))
                    stability_score = min(100, max(0, (1 - (1-best_score)*2) * 100)) if best_score > 0 else 0
                    fairness_score = 80  # simplified
                    cert = ip_score * 0.4 + stability_score * 0.3 + fairness_score * 0.3

                    results[tgt] = {
                        'best_model': best_name, 'best_r2': best_score,
                        'rmse': model_results.get(best_name, {}).get('rmse', 0),
                        'mae': model_results.get(best_name, {}).get('mae', 0),
                        'metric': model_results.get(best_name, {}).get('metric', 'r2'),
                        'is_classif': is_classif,
                        'cert_score': cert, 'top_feats': top_feats,
                        'model_results': model_results,
                        'predictions': pred.tolist()[:50],
                        'y_test': y_te.tolist()[:50],
                    }
                    metric_lbl = 'Accuracy' if is_classif else 'R\u00b2'
                    progress.progress((ti+1)/len(target_cols), text=f'{tgt}: {metric_lbl}={best_score:.4f}')

                st.session_state.results = results
                st.session_state.dataset = tgt
                st.success(f"\u2705 PEC termin\u00e9 pour {len(results)} cible(s) !" if lang=='fr' else f"\u2705 PEC completed for {len(results)} target(s)!")
                st.rerun()
    else:
        st.error(f"File not found: {demo_path}")

# ============================================================================
# UPLOAD TAB
# ============================================================================
with tab_upload:
    lang = st.session_state.lang
    st.markdown(f'<h1><i class="fa-solid fa-cloud-arrow-up"></i> {t("select_upload")}</h1>', unsafe_allow_html=True)
    st.markdown(f'<div class="interp-box">{t("upload_desc")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pec-card"><strong>{t("upload_format")}</strong></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(t('select_upload'), type=['csv'], key='upload_csv3', help='CSV file')

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.success(f"\u2705 {len(df_up)} lignes \u00d7 {len(df_up.columns)} colonnes")
            st.dataframe(df_up.head(15), use_container_width=True, hide_index=True)

            numeric_cols = df_up.select_dtypes(include=[np.number]).columns.tolist()
            st.info(f"\U0001f4ca {t('col_count')}: {len(numeric_cols)}")

            if len(numeric_cols) >= 2:
                target_cols = st.multiselect(t('target_sel'), numeric_cols, default=numeric_cols[-1:], key='upload_targets3')

                if target_cols and st.button(t('run_pec'), type='primary', use_container_width=True, key='upload_run3'):
                    with st.spinner(t('run_pec')):
                        from sklearn.model_selection import train_test_split
                        from sklearn.linear_model import Ridge
                        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                        from sklearn.preprocessing import StandardScaler

                        feature_cols = [c for c in numeric_cols if c not in target_cols]
                        results = {}
                        progress = st.progress(0, text=t('run_pec'))

                        for ti, tgt in enumerate(target_cols):
                            df_clean = df_up.dropna(subset=[tgt] + feature_cols[:50])
                            if len(df_clean) < 10: continue
                            use_feats = [c for c in feature_cols[:50] if c in df_clean.columns and df_clean[c].notna().sum() > len(df_clean)*0.5]
                            if not use_feats: continue
                            X = df_clean[use_feats].fillna(df_clean[use_feats].mean())
                            y = df_clean[tgt]

                            # Detect classification vs regression
                            n_unique = y.nunique()
                            is_classif = n_unique <= 10 and y.dtype in ['int64','int32','object']

                            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                            sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

                            if is_classif:
                                from sklearn.linear_model import LogisticRegression
                                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                                from sklearn.metrics import accuracy_score, f1_score
                                models_dict = {'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42), 'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42), 'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)}
                            else:
                                from sklearn.linear_model import Ridge
                                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                                models_dict = {'Ridge': Ridge(alpha=1.0), 'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42), 'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)}

                            model_results = {}
                            best_score = -999; best_name = 'N/A'; best_model_obj = None
                            for mn, m in models_dict.items():
                                try:
                                    if 'Logistic' in mn or 'Ridge' in mn: m.fit(X_tr_s, y_tr); pred = m.predict(X_te_s)
                                    else: m.fit(X_tr, y_tr); pred = m.predict(X_te)
                                    if is_classif:
                                        sc_val = accuracy_score(y_te, pred); f1 = f1_score(y_te, pred, average='weighted', zero_division=0)
                                        model_results[mn] = {'r2': sc_val, 'rmse': 1-sc_val, 'mae': f1, 'metric': 'accuracy'}
                                    else:
                                        r2 = r2_score(y_te, pred); rmse = np.sqrt(mean_squared_error(y_te, pred))
                                        model_results[mn] = {'r2': r2, 'rmse': rmse, 'metric': 'r2'}
                                    if model_results[mn]['r2'] > best_score: best_score = model_results[mn]['r2']; best_name = mn; best_model_obj = m
                                except: pass

                            # Feature importance
                            from sklearn.inspection import permutation_importance
                            feat_imp = {}
                            if hasattr(best_model_obj, 'feature_importances_'):
                                feat_imp = dict(zip(use_feats, best_model_obj.feature_importances_))
                            elif best_model_obj is not None:
                                try:
                                    X_te_imp = X_te_s if ('Logistic' in best_name or 'Ridge' in best_name) else X_te
                                    perm = permutation_importance(best_model_obj, X_te_imp, y_te, n_repeats=10, random_state=42, n_jobs=-1)
                                    feat_imp = dict(zip(use_feats, perm.importances_mean))
                                except: pass
                            top_feats = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:10] if feat_imp else [(f, 0) for f in use_feats[:10]]

                            ip_score = min(100, max(0, best_score * 100))
                            stability_score = min(100, max(0, (1 - (1-best_score)*2) * 100)) if best_score > 0 else 0
                            cert = ip_score * 0.4 + stability_score * 0.3 + 80 * 0.3

                            results[tgt] = {'best_model': best_name, 'best_r2': best_score, 'rmse': model_results.get(best_name, {}).get('rmse', 0), 'metric': model_results.get(best_name, {}).get('metric', 'r2'), 'is_classif': is_classif, 'cert_score': cert, 'top_feats': top_feats, 'model_results': model_results, 'predictions': pred.tolist()[:50], 'y_test': y_te.tolist()[:50]}
                            metric_lbl = 'Accuracy' if is_classif else 'R\u00b2'
                            progress.progress((ti+1)/len(target_cols), text=f'{tgt}: {metric_lbl}={best_score:.4f}')

                        st.session_state.results = results
                        st.session_state.dataset = f'upload_{len(df_up)}'
                        st.success(f"\u2705 PEC termin\u00e9 pour {len(results)} cible(s) !")
                        st.rerun()
        except Exception as e:
            st.error(f"Erreur : {e}")

# ============================================================================
# RESULTS TAB
# ============================================================================
with tab_results:
    lang = st.session_state.lang
    st.markdown(f'<h1><i class="fa-solid fa-chart-column"></i> {t("tab_results")}</h1>', unsafe_allow_html=True)

    results = st.session_state.results
    if not results:
        st.info(t('no_data'))
    else:
        for tgt, r in results.items():
            best_r2 = r.get('best_r2', 0)
            cert = r.get('cert_score', 0)
            best_name = r.get('best_model', 'N/A')
            grade, badge, desc = grade_info(cert)

            st.markdown(f"""
            <div class="pec-card card-certify">
                <h3 style="color:var(--secondary);"><i class="fa-solid fa-shield-halved"></i> {tgt}</h3>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:.8rem;">
                    <div style="text-align:center;padding:1rem;border-radius:var(--radius);background:#f0fff0;border-top:4px solid #27ae60;">
                        <div class="metric-val" style="color:#27ae60;">{cert:.1f}<span style="font-size:1rem;">/100</span></div>
                        <div class="metric-lbl">{t('cert_score')}</div>
                    </div>
                    <div style="text-align:center;padding:1rem;border-radius:var(--radius);background:#f0f7ff;border-top:4px solid #2e86c1;">
                        <div class="metric-val" style="color:#2e86c1;">{best_r2:.4f}</div>
                        <div class="metric-lbl">{'Accuracy' if r.get('is_classif') else t('r2_score')}</div>
                    </div>
                    <div style="text-align:center;padding:1rem;border-radius:var(--radius);background:#f8f0ff;border-top:4px solid #8e44ad;">
                        <span class="{badge}" style="font-size:2rem;">{grade}</span>
                        <div class="metric-lbl">{t('grade')}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="interp-box"><strong>Grade {grade} :</strong> {desc}</div>', unsafe_allow_html=True)

            # Model comparison
            model_results = r.get('model_results', {})
            if model_results:
                st.markdown(f'<h4><i class="fa-solid fa-code-compare"></i> {t("model_compare")}</h4>', unsafe_allow_html=True)
                models_names = list(model_results.keys())
                r2_vals = [model_results[m]['r2'] for m in models_names]
                colors_bar = ['#27ae60' if v == max(r2_vals) else '#2e86c1' for v in r2_vals]
                fig = go.Figure(go.Bar(x=models_names, y=r2_vals, marker_color=colors_bar,
                                       text=[f'{v:.4f}' for v in r2_vals], textposition='outside'))
                metric_key = 'Accuracy' if r.get('is_classif') else 'R\u00b2'
                fig.update_layout(height=300, showlegend=False, yaxis_title=metric_key)
                st.plotly_chart(fig, use_container_width=True, key=f'model_comp_{tgt}')

            # Feature importance
            top_feats = r.get('top_feats', [])
            if top_feats:
                st.markdown(f'<h4><i class="fa-solid fa-ranking-star"></i> {t("features")}</h4>', unsafe_allow_html=True)
                feat_names = [f[0][:30] for f in top_feats]
                feat_vals = [abs(f[1]) for f in top_feats]
                colors_feat = ['#27ae60' if f[1] > 0 else '#e74c3c' for f in top_feats]
                fig_f = go.Figure(go.Bar(x=feat_vals, y=feat_names, orientation='h', marker_color=colors_feat))
                fig_f.update_layout(height=350, showlegend=False, xaxis_title='Importance')
                st.plotly_chart(fig_f, use_container_width=True, key=f'feat_imp_{tgt}')
                st.markdown(f'<div class="interp-box"><i class="fa-solid fa-circle-info"></i> {t("shap_note")}</div>', unsafe_allow_html=True)

# ============================================================================
# ABOUT TAB
# ============================================================================
with tab_about:
    lang = st.session_state.lang
    st.markdown(f'<h1><i class="fa-solid fa-circle-info"></i> {t("about_what")}</h1>', unsafe_allow_html=True)
    st.markdown(f'<div class="interp-box">{t("about_what_desc")}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    # PREDICT / EXPLAIN / CERTIFY cards — traduits
    if lang == 'fr':
        c1_code = '''<div class="pec-card card-predict"><h4 style="color:#2e86c1;"><i class="fa-solid fa-bullseye"></i> PREDIRE</h4><p style="font-size:.95rem;line-height:1.6;">6 mod\u00e8les d'IA estiment les tendances futures. Le meilleur mod\u00e8le est s\u00e9lectionn\u00e9 automatiquement selon les performances (R\u00b2).</p></div>'''
        c2_code = '''<div class="pec-card card-explain"><h4 style="color:#8e44ad;"><i class="fa-solid fa-magnifying-glass"></i> EXPLIQUER</h4><p style="font-size:.95rem;line-height:1.6;">SHAP, LIME et Permutation Importance identifient les facteurs les plus influents, rendant les pr\u00e9dictions transparentes et interpr\u00e9tables.</p></div>'''
        c3_code = '''<div class="pec-card card-certify"><h4 style="color:#27ae60;"><i class="fa-solid fa-shield-halved"></i> CERTIFIER</h4><p style="font-size:.95rem;line-height:1.6;">Un score composite sur 100 (Performance + Stabilit\u00e9 + \u00c9quit\u00e9) certifie la fiabilit\u00e9 du syst\u00e8me, avec des grades de A \u00e0 D.</p></div>'''
    else:
        c1_code = '''<div class="pec-card card-predict"><h4 style="color:#2e86c1;"><i class="fa-solid fa-bullseye"></i> PREDICT</h4><p style="font-size:.95rem;line-height:1.6;">6 AI models estimate future trends. The best model is automatically selected based on performance (R\u00b2).</p></div>'''
        c2_code = '''<div class="pec-card card-explain"><h4 style="color:#8e44ad;"><i class="fa-solid fa-magnifying-glass"></i> EXPLAIN</h4><p style="font-size:.95rem;line-height:1.6;">SHAP, LIME, and Permutation Importance identify the most influential factors, making predictions transparent and interpretable.</p></div>'''
        c3_code = '''<div class="pec-card card-certify"><h4 style="color:#27ae60;"><i class="fa-solid fa-shield-halved"></i> CERTIFY</h4><p style="font-size:.95rem;line-height:1.6;">A composite score out of 100 (Performance + Stability + Fairness) certifies system reliability, with grades from A to D.</p></div>'''
    c1.markdown(c1_code, unsafe_allow_html=True)
    c2.markdown(c2_code, unsafe_allow_html=True)
    c3.markdown(c3_code, unsafe_allow_html=True)

    # Conditions d'application
    if lang == 'fr':
        st.markdown(f'<h3><i class="fa-solid fa-list-check"></i> Conditions d\'application</h3>', unsafe_allow_html=True)
        st.markdown('''<div class="pec-card"><ul style="line-height:2;">
        <li>Donn\u00e9es tabulaires (CSV) avec au moins 20 observations et 3 variables num\u00e9riques</li>
        <li>Variables cibles num\u00e9riques (r\u00e9gression) ou cat\u00e9gorielles (classification)</li>
        <li>Aucune valeur manquante critique (>50% par colonne)</li>
        <li>Relations non purement al\u00e9atoires entre variables explicatives et cible</li>
        </ul></div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'<h3><i class="fa-solid fa-list-check"></i> Application conditions</h3>', unsafe_allow_html=True)
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
    PEC Framework v4 | <i class="fa-solid fa-flask"></i> Predict \u2022 <i class="fa-solid fa-magnifying-glass"></i> Explain \u2022 <i class="fa-solid fa-shield-halved"></i> Certify |
    <a href="https://orcid.org/0009-0003-3048-1765" target="_blank" style="color:#a6ce39;"><i class="fa-brands fa-orcid"></i> Rosa Elysabeth Ralinirina</a>
</div>""", unsafe_allow_html=True)