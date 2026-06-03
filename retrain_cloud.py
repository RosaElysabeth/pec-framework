#!/usr/bin/env python3
"""
retrain_cloud.py — Régénère TOUS les modèles .pkl compatibles Python 3.14
À déployer UNE FOIS sur Streamlit Cloud, télécharger les .pkl, puis push sur GitHub.
Après usage, supprimer ce fichier du repo.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle, joblib, os, warnings, io, zipfile
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

st.set_page_config(page_title="PEC Retrain Cloud", layout="wide")
st.title("🔄 PEC — Régénération des modèles (Python 3.14)")

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")
GEN_MODELS_DIR = os.path.join(BASE, "generalisation", "models")
DATA_DIR = os.path.join(BASE, "data")
GEN_DATA_DIR = os.path.join(BASE, "generalisation", "data")

all_pkl_bytes = {}  # nom -> bytes du .pkl

def train_madagascar():
    """Entraîne les 6 modèles × 3 cibles Madagascar."""
    st.header("🇲🇬 Madagascar — 6 modèles × 3 cibles")
    csv_path = os.path.join(DATA_DIR, "dataset_pec_v3.csv")
    df = pd.read_csv(csv_path)
    targets = ['Insuffisance_ponderale', 'Malnutrition_aigue', 'Malnutrition_chronique']
    
    # Préparer les features
    drop_cols = targets + ['Region', 'Annee'] if 'Annee' in df.columns else targets + ['Region']
    feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in ['float64','int64','float32','int32']]
    
    # Label encoder pour Region
    le = LabelEncoder()
    df['Region_enc'] = le.fit(df['Region'].unique()).transform(df['Region'])
    feature_cols.append('Region_enc')
    
    # Scaler
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    
    # Feature stats
    feature_stats = {}
    for c in feature_cols:
        feature_stats[c] = {'mean': float(df[c].mean()), 'std': float(df[c].std())}
    
    models_dict = {}
    progress = st.progress(0, "Entraînement Madagascar...")
    
    model_classes = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=0),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Ridge': Ridge(alpha=1.0),
    }
    
    i = 0
    total = len(model_classes) * len(targets)
    for mn, model_template in model_classes.items():
        for tgt in targets:
            i += 1
            y = df[tgt].values
            X = df[feature_cols].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Clone le modèle
            import copy
            model = copy.deepcopy(model_template)
            model.fit(X_train, y_train)
            
            r2 = r2_score(y_test, model.predict(X_test))
            models_dict[(mn, tgt)] = model
            progress.progress(i/total, f"{mn} × {tgt}: R²={r2:.4f}")
    
    # Sauver en mémoire
    meta = {'models': list(model_classes.keys()), 'targets': targets}
    all_pkl_bytes['models/meta.pkl'] = pickle.dumps(meta, protocol=4)
    all_pkl_bytes['models/scaler.pkl'] = pickle.dumps(scaler, protocol=4)
    all_pkl_bytes['models/feature_cols.pkl'] = pickle.dumps(feature_cols, protocol=4)
    all_pkl_bytes['models/label_encoder_region.pkl'] = pickle.dumps(le, protocol=4)
    all_pkl_bytes['models/feature_stats.pkl'] = pickle.dumps(feature_stats, protocol=4)
    
    for (mn, tgt), model in models_dict.items():
        all_pkl_bytes[f'models/{mn}_{tgt}.pkl'] = pickle.dumps(model, protocol=4)
    
    # Corrélations
    corr = df[feature_cols + targets].corr()
    all_pkl_bytes['models/correlations.pkl'] = pickle.dumps(corr, protocol=4)
    
    progress.progress(1.0, "✅ Madagascar terminé !")
    st.success(f"✅ Madagascar : {len(models_dict)} modèles + 6 fichiers annexes")
    return models_dict

def train_generalisation():
    """Entraîne les 3 modèles × 3 cibles × 3 domaines."""
    st.header("🌍 Généralisation — 3 domaines × 3 cibles")
    
    domaines = {
        'climat': {
            'csv': 'dataset_climat.csv',
            'models': ['Random_Forest', 'XGBoost', 'CatBoost'],
            'cibles_col': None,  # auto-detect
        },
        'fintech': {
            'csv': 'dataset_fintech.csv',
            'models': ['Random_Forest', 'XGBoost', 'CatBoost'],
            'cibles_col': None,
        },
        'sante': {
            'csv': 'dataset_sante.csv',
            'models': ['Random_Forest', 'XGBoost', 'CatBoost'],
            'cibles_col': None,
        },
    }
    
    # Définir les cibles par domaine
    target_cols = {
        'climat': ['Degats_millions_USD', 'Indice_Risque_Inondation', 'Population_a_risque_M'],
        'fintech': ['Penetration_Fintech_pct', 'Ratio_NPL_pct', 'Taux_Defaut_pct'],
        'sante': ['Accouchements_qualified_pct', 'Mortalite_infantile_p1000', 'Vaccination_complete_pct'],
    }
    
    model_map = {
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=0),
    }
    
    total_models = 3 * 3 * 3  # 3 domaines × 3 cibles × 3 modèles
    count = 0
    progress = st.progress(0, "Entraînement Généralisation...")
    
    for dom_key, dom_info in domaines.items():
        st.subheader(f"  {dom_key.capitalize()}")
        csv_path = os.path.join(GEN_DATA_DIR, dom_info['csv'])
        df = pd.read_csv(csv_path)
        
        targets = target_cols[dom_key]
        drop_cols = targets + [c for c in df.columns if c in ['Pays', 'Region', 'Country', 'Year', 'Annee']]
        feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in ['float64','int64','float32','int32']]
        
        # Label encoder si colonne catégorielle
        le = None
        cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in drop_cols]
        if cat_cols:
            le = LabelEncoder()
            for cc in cat_cols:
                df[f'{cc}_enc'] = le.fit(df[cc].astype(str)).transform(df[cc].astype(str))
                feature_cols.append(f'{cc}_enc')
                drop_cols.append(cc)
        
        # Scaler
        scaler = StandardScaler()
        scaler.fit(df[feature_cols])
        
        # Feature stats
        feature_stats = {}
        for c in feature_cols:
            feature_stats[c] = {'mean': float(df[c].mean()), 'std': float(df[c].std())}
        
        # Sauver annexes
        all_pkl_bytes[f'generalisation/models/{dom_key}/feature_cols.pkl'] = pickle.dumps(feature_cols, protocol=4)
        all_pkl_bytes[f'generalisation/models/{dom_key}/feature_stats.pkl'] = pickle.dumps(feature_stats, protocol=4)
        all_pkl_bytes[f'generalisation/models/{dom_key}/scaler.pkl'] = pickle.dumps(scaler, protocol=4)
        if le:
            all_pkl_bytes[f'generalisation/models/{dom_key}/label_encoder.pkl'] = pickle.dumps(le, protocol=4)
        
        for tgt in targets:
            y = df[tgt].values
            X = df[feature_cols].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            for mn_template, model_template in model_map.items():
                count += 1
                import copy
                model = copy.deepcopy(model_template)
                model.fit(X_train, y_train)
                r2 = r2_score(y_test, model.predict(X_test))
                
                all_pkl_bytes[f'generalisation/models/{dom_key}/{mn_template}_{tgt}.pkl'] = pickle.dumps(model, protocol=4)
                progress.progress(count/total_models, f"{dom_key}/{mn_template}×{tgt}: R²={r2:.4f}")
    
    progress.progress(1.0, "✅ Généralisation terminée !")
    st.success(f"✅ Généralisation : {count} modèles + fichiers annexes")

# ============================================================================
# MAIN
# ============================================================================

if st.button("🚀 Lancer la régénération complète", type="primary", use_container_width=True):
    train_madagascar()
    train_generalisation()
    
    st.header("📦 Télécharger les .pkl")
    st.warning("Téléchargez le ZIP, extrayez dans votre repo local, puis `git push`.")
    
    # Créer un ZIP en mémoire
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, data in all_pkl_bytes.items():
            zf.writestr(name, data)
    zip_buffer.seek(0)
    
    st.download_button(
        "⬇️ Télécharger pec_models.zip",
        data=zip_buffer.getvalue(),
        file_name="pec_models.zip",
        mime="application/zip",
        use_container_width=True,
    )
    
    st.info(f"**{len(all_pkl_bytes)} fichiers .pkl** prêts à remplacer dans le repo")
    
    st.markdown("""
    ### Étapes après téléchargement :
    1. Extraire `pec_models.zip` dans `08-PROJET-PEC/`
    2. Les fichiers remplacent les anciens `.pkl`
    3. `git add . && git commit -m "fix: models regenerated for Python 3.14" && git push`
    4. Supprimer `retrain_cloud.py` du repo
    """)

st.divider()
st.caption("PEC Framework v4 — Régénération des modèles pour compatibilité Python 3.14")