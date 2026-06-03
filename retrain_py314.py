"""retrain_py314.py - Regénère les .pkl avec Python 3.14 (même environnement que Streamlit Cloud)"""
import pandas as pd, numpy as np, pickle, os, warnings, copy
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
BASE = '.'

# ====== MADAGASCAR ======
print('MADAGASCAR...')
MODELS_DIR = os.path.join(BASE, 'models')
df = pd.read_csv(os.path.join(BASE, 'data', 'dataset_pec_v3.csv'))
targets = ['Insuffisance_ponderale', 'Malnutrition_aigue', 'Malnutrition_chronique']
drop_cols = targets + ['Region', 'Annee']
feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in ['float64','int64','float32','int32']]

le = LabelEncoder()
df['Region_enc'] = le.fit_transform(df['Region'].astype(str))
feature_cols.append('Region_enc')

# Imputer pour les NaN
imputer = SimpleImputer(strategy='median')
X_all = imputer.fit_transform(df[feature_cols])
y_all = {t: df[t].values for t in targets}

scaler = StandardScaler()
scaler.fit(X_all)
feature_stats = {c: {'mean': float(df[c].mean()), 'std': float(df[c].std())} for c in feature_cols}

model_classes = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=0),
    'GradientBoosting': HistGradientBoostingRegressor(max_iter=100, random_state=42),
    'Ridge': Ridge(alpha=1.0),
}

models_dict = {}
for mn, tmpl in model_classes.items():
    for tgt in targets:
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all[tgt], test_size=0.2, random_state=42)
        model = copy.deepcopy(tmpl)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        models_dict[(mn, tgt)] = model
        print(f'  {mn} x {tgt}: R2={r2:.4f}')

meta = {'models': list(model_classes.keys()), 'targets': targets}
for name, obj in [('meta.pkl', meta), ('scaler.pkl', scaler), ('feature_cols.pkl', feature_cols),
                   ('label_encoder_region.pkl', le), ('feature_stats.pkl', feature_stats), ('imputer.pkl', imputer)]:
    with open(os.path.join(MODELS_DIR, name), 'wb') as f: pickle.dump(obj, f, protocol=4)
corr = pd.DataFrame(X_all, columns=feature_cols).join(pd.DataFrame(y_all)).corr()
with open(os.path.join(MODELS_DIR, 'correlations.pkl'), 'wb') as f: pickle.dump(corr, f, protocol=4)
for (mn, tgt), model in models_dict.items():
    with open(os.path.join(MODELS_DIR, f'{mn}_{tgt}.pkl'), 'wb') as f: pickle.dump(model, f, protocol=4)
print(f'MADAGASCAR OK: {len(models_dict)} modeles')

# ====== GENERALISATION ======
GEN_MODELS = os.path.join(BASE, 'generalisation', 'models')
GEN_DATA = os.path.join(BASE, 'generalisation', 'data')
target_cols = {
    'climat': ['Degats_millions_USD', 'Indice_Risque_Inondation', 'Population_a_risque_M'],
    'fintech': ['Penetration_Fintech_pct', 'Ratio_NPL_pct', 'Taux_Defaut_pct'],
    'sante': ['Accouchements_qualified_pct', 'Mortalite_infantile_p1000', 'Vaccination_complete_pct'],
}
gen_model_map = {
    'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=0),
}

count = 0
for dom_key, tgts in target_cols.items():
    print(f'{dom_key.upper()}...')
    df = pd.read_csv(os.path.join(GEN_DATA, f'dataset_{dom_key}.csv'))
    drop_cols = tgts + [c for c in df.columns if c in ['Pays', 'Region', 'Country', 'Year', 'Annee']]
    feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in ['float64','int64','float32','int32']]
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in drop_cols]
    le_d = None
    if cat_cols:
        le_d = LabelEncoder()
        for cc in cat_cols:
            df[f'{cc}_enc'] = le_d.fit_transform(df[cc].astype(str))
            feature_cols.append(f'{cc}_enc')
    
    imputer_d = SimpleImputer(strategy='median')
    X_all = imputer_d.fit_transform(df[feature_cols])
    y_all = {t: df[t].values for t in tgts}
    
    scaler_d = StandardScaler()
    scaler_d.fit(X_all)
    feature_stats_d = {c: {'mean': float(df[c].mean()), 'std': float(df[c].std())} for c in feature_cols}
    
    dom_dir = os.path.join(GEN_MODELS, dom_key)
    for name, obj in [('feature_cols.pkl', feature_cols), ('feature_stats.pkl', feature_stats_d),
                       ('scaler.pkl', scaler_d), ('imputer.pkl', imputer_d)]:
        with open(os.path.join(dom_dir, name), 'wb') as f: pickle.dump(obj, f, protocol=4)
    if le_d:
        with open(os.path.join(dom_dir, 'label_encoder.pkl'), 'wb') as f: pickle.dump(le_d, f, protocol=4)
    
    for tgt in tgts:
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all[tgt], test_size=0.2, random_state=42)
        for mn, tmpl in gen_model_map.items():
            model = copy.deepcopy(tmpl)
            model.fit(X_train, y_train)
            r2 = r2_score(y_test, model.predict(X_test))
            with open(os.path.join(dom_dir, f'{mn}_{tgt}.pkl'), 'wb') as f: pickle.dump(model, f, protocol=4)
            count += 1
            print(f'  {mn} x {tgt}: R2={r2:.4f}')

print(f'GENERALISATION OK: {count} modeles')
print(f'\nTOTAL: {len(models_dict) + count} modeles (Python 3.14 + sklearn {sklearn.__version__})')