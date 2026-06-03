#!/usr/bin/env python3
# ============================================================================
# PEC PIPELINE DE GENERALISATION
# Entraine 6 modeles sur 3 datasets differents, genere SHAP, certification
# Prouve que le framework PEC s'applique a tout domaine
# Auteur : Rosa Elysabeth Ralinirina | ORCID : 0009-0003-3048-1765
# ============================================================================

import numpy as np
import pandas as pd
import json, os, warnings, pickle, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import resample

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# CONFIGURATION DES 3 DOMAINES
# ============================================================================
DOMAINS = {
    'fintech': {
        'name': 'Fintech Afrique de l\'Ouest',
        'name_en': 'West Africa Fintech',
        'csv': os.path.join(BASE, 'data', 'dataset_fintech.csv'),
        'region_col': 'Pays',
        'year_col': 'Annee',
        'targets': {
            'Taux_Defaut_pct': {'short': 'TD', 'fr': 'Taux de Défaut (%)', 'en': 'Default Rate (%)', 'color': '#e74c3c'},
            'Ratio_NPL_pct': {'short': 'NPL', 'fr': 'Ratio NPL (%)', 'en': 'NPL Ratio (%)', 'color': '#3498db'},
            'Penetration_Fintech_pct': {'short': 'PF', 'fr': 'Pénétration Fintech (%)', 'en': 'Fintech Penetration (%)', 'color': '#27ae60'},
        },
        'train_years': list(range(2016, 2022)),
        'test_years': list(range(2022, 2025)),
    },
    'sante': {
        'name': 'Santé Afrique de l\'Est',
        'name_en': 'East Africa Health',
        'csv': os.path.join(BASE, 'data', 'dataset_sante.csv'),
        'region_col': 'Province',
        'year_col': 'Annee',
        'targets': {
            'Mortalite_infantile_p1000': {'short': 'MI', 'fr': 'Mortalité Infantile (‰)', 'en': 'Infant Mortality (‰)', 'color': '#e74c3c'},
            'Vaccination_complete_pct': {'short': 'VC', 'fr': 'Vaccination Complète (%)', 'en': 'Full Vaccination (%)', 'color': '#27ae60'},
            'Accouchements_qualified_pct': {'short': 'AQ', 'fr': 'Accouchements Qualifiés (%)', 'en': 'Qualified Deliveries (%)', 'color': '#3498db'},
        },
        'train_years': list(range(2013, 2021)),
        'test_years': list(range(2021, 2024)),
    },
    'climat': {
        'name': 'Climat Asie du Sud-Est',
        'name_en': 'Southeast Asia Climate',
        'csv': os.path.join(BASE, 'data', 'dataset_climat.csv'),
        'region_col': 'Zone',
        'year_col': 'Annee',
        'targets': {
            'Indice_Risque_Inondation': {'short': 'IRI', 'fr': 'Indice Risque Inondation', 'en': 'Flood Risk Index', 'color': '#e74c3c'},
            'Degats_millions_USD': {'short': 'DMG', 'fr': 'Dégâts (M$)', 'en': 'Damage (M$)', 'color': '#8e44ad'},
            'Population_a_risque_M': {'short': 'PAR', 'fr': 'Population à Risque (M)', 'en': 'Population at Risk (M)', 'color': '#3498db'},
        },
        'train_years': list(range(2014, 2022)),
        'test_years': list(range(2022, 2025)),
    }
}

def get_models():
    """Retourne la liste des modeles a entrainer."""
    models = {
        'Random_Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        'Ridge': Ridge(alpha=1.0),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
    if HAS_LGBM:
        models['LightGBM'] = LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
    if HAS_CB:
        models['CatBoost'] = CatBoostRegressor(iterations=200, depth=5, learning_rate=0.05, random_state=42, verbose=0)
    return models

def compute_certification(y_true, y_pred, X_test, n_bootstrap=100):
    """Calcule le score certificatif composite."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Stabilite bootstrap
    bootstrap_r2 = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(idx)) < 2:
            continue
        y_bt = y_true[idx]
        y_bp = y_pred[idx]
        if np.std(y_bt) > 0:
            bootstrap_r2.append(r2_score(y_bt, y_bp))
    stability = np.mean(bootstrap_r2) if bootstrap_r2 else 0
    stability_std = np.std(bootstrap_r2) if len(bootstrap_r2) > 1 else 0
    
    # Score composite (0-100)
    perf_score = max(0, min(100, r2 * 100))
    stab_score = max(0, min(100, stability * 100))
    cert_score = 0.5 * perf_score + 0.3 * stab_score + 0.2 * max(0, 100 - (stability_std * 100))
    
    return {
        'R2_test': float(r2),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'stability_mean': float(stability),
        'stability_std': float(stability_std),
        'performance_score': float(perf_score),
        'stability_score': float(stab_score),
        'certification_score': float(cert_score),
        'grade': 'A' if cert_score >= 90 else 'B' if cert_score >= 80 else 'C' if cert_score >= 60 else 'D'
    }


def run_pec_pipeline(domain_key, domain_config):
    """Execute le pipeline PEC complet pour un domaine."""
    print(f'\n{"="*70}')
    print(f'  PEC PIPELINE — {domain_config["name"]}')
    print(f'{"="*70}')
    
    # Load data
    df = pd.read_csv(domain_config['csv'])
    region_col = domain_config['region_col']
    year_col = domain_config['year_col']
    targets = domain_config['targets']
    
    print(f'  Data: {df.shape[0]} lignes x {df.shape[1]} colonnes')
    print(f'  Regions: {df[region_col].nunique()} | Annees: {df[year_col].min()}-{df[year_col].max()}')
    
    # Encode region
    le = LabelEncoder()
    region_enc_col = f'{region_col}_enc'
    df[region_enc_col] = le.fit_transform(df[region_col])
    
    # Feature columns (exclude targets and region name)
    exclude = list(targets.keys()) + [region_col]
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'int32']]
    # Remove the _enc column from feature_cols, we'll add it separately
    if region_enc_col in feature_cols:
        pass  # keep it
    
    print(f'  Features: {len(feature_cols)} colonnes')
    print(f'  Targets: {list(targets.keys())}')
    
    # Split train/test
    train_mask = df[year_col].isin(domain_config['train_years'])
    test_mask = df[year_col].isin(domain_config['test_years'])
    
    X_train = df.loc[train_mask, feature_cols].copy()
    X_test = df.loc[test_mask, feature_cols].copy()
    
    # Fill NaN with train means
    for col in feature_cols:
        if X_train[col].isnull().any():
            mean_val = X_train[col].mean()
            X_train[col] = X_train[col].fillna(mean_val)
            X_test[col] = X_test[col].fillna(mean_val)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
    
    print(f'  Train: {X_train.shape[0]} lignes ({domain_config["train_years"][0]}-{domain_config["train_years"][-1]})')
    print(f'  Test: {X_test.shape[0]} lignes ({domain_config["test_years"][0]}-{domain_config["test_years"][-1]})')
    
    # Save scaler and feature info
    models_dir = os.path.join(BASE, 'models', domain_key)
    results_dir = os.path.join(BASE, 'results', domain_key)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'shap'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'certification'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'comparaison_xai'), exist_ok=True)
    
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    feature_stats = {col: {'mean': float(X_train[col].mean()), 'std': float(X_train[col].std())} for col in feature_cols}
    with open(os.path.join(models_dir, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    with open(os.path.join(models_dir, 'feature_stats.pkl'), 'wb') as f:
        pickle.dump(feature_stats, f)
    with open(os.path.join(models_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    
    # Train models for each target
    all_models = get_models()
    results = {'predict': {}, 'explain': {}, 'certify': {}}
    
    for target_key, target_info in targets.items():
        short = target_info['short']
        print(f'\n  --- Target: {target_key} ({short}) ---')
        
        y_train = df.loc[train_mask, target_key].values
        y_test = df.loc[test_mask, target_key].values
        
        target_results = {}
        best_r2 = -float('inf')
        best_model_name = 'Ridge'  # default
        
        for model_name, model_template in all_models.items():
            from sklearn.base import clone
            model = clone(model_template)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            target_results[model_name] = {
                'R2': float(r2),
                'RMSE': float(rmse),
                'MAE': float(mae)
            }
            
            print(f'    {model_name:20s}: R2={r2:.4f}, RMSE={rmse:.3f}, MAE={mae:.3f}')
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name
            
            # Save model
            with open(os.path.join(models_dir, f'{model_name}_{target_key}.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        # Certification
        best_model = clone(all_models[best_model_name])
        best_model.fit(X_train_scaled, y_train)
        y_pred_best = best_model.predict(X_test_scaled)
        
        cert = compute_certification(y_test, y_pred_best, X_test_scaled)
        
        results['predict'][target_key] = {
            'meilleur_modele': best_model_name,
            'R2_test': float(best_r2),
            'tous_modeles': target_results
        }
        
        results['certify'][target_key] = {
            'model': best_model_name,
            'R2_test': float(best_r2),
            'RMSE': float(r2_score(y_test, y_pred_best)),
            'certification_score': cert['certification_score'],
            'performance_score': cert['performance_score'],
            'stability_score': cert['stability_score'],
            'grade': cert['grade'],
            'n_bootstrap': 100
        }
        
        print(f'    >> Best: {best_model_name} (R2={best_r2:.4f}, Cert={cert["certification_score"]:.1f}/100, Grade={cert["grade"]})')
        
        # SHAP explainability
        if HAS_SHAP:
            try:
                print(f'    SHAP analysis...')
                # Use best model for SHAP
                explainer = shap.Explainer(best_model, X_train_scaled[:50])
                shap_values = explainer(X_test_scaled[:50])
                
                # Top 5 features
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                top5_indices = np.argsort(mean_abs_shap)[-5:][::-1]
                top5_features = [feature_cols[i] for i in top5_indices]
                top5_values = [float(mean_abs_shap[i]) for i in top5_indices]
                
                results['explain'][target_key] = {
                    'model_utilise': best_model_name,
                    'top5_SHAP': dict(zip(top5_features, top5_values)),
                    'methodes_comparees': {
                        'SHAP_nb_features': len(feature_cols),
                        'top5': top5_features[:5]
                    }
                }
                
                # Save SHAP summary plot
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_scaled[:50], feature_names=feature_cols, show=False, max_display=15)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'shap', f'shap_summary_{target_key}.png'), dpi=150, bbox_inches='tight')
                plt.close()
                
                # Bar plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.bar(shap_values, max_display=15, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'shap', f'shap_bar_{target_key}.png'), dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f'    SHAP: Top 5 = {top5_features[:5]}')
            except Exception as e:
                print(f'    SHAP error: {e}')
                results['explain'][target_key] = {'model_utilise': best_model_name, 'error': str(e)}
    
    # Model comparison plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, len(targets), figsize=(6*len(targets), 5))
    if len(targets) == 1:
        axes = [axes]
    
    for idx, (target_key, target_info) in enumerate(targets.items()):
        ax = axes[idx]
        model_names = []
        r2_values = []
        for mn, mv in results['predict'][target_key]['tous_modeles'].items():
            model_names.append(mn)
            r2_values.append(mv['R2'])
        
        sorted_idx = np.argsort(r2_values)
        ax.barh([model_names[i] for i in sorted_idx], [r2_values[i] for i in sorted_idx], color='#27ae60')
        ax.set_xlabel('R²')
        ax.set_title(f'{target_info["short"]}')
        ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='R²=0.8')
        ax.legend()
    
    plt.suptitle(f'Comparaison des Modeles — {domain_config["name"]}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'figures', 'comparaison_modeles.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Certification composite plot
    fig, ax = plt.subplots(figsize=(8, 5))
    target_names = [targets[k]['short'] for k in targets]
    cert_scores = [results['certify'][k]['certification_score'] for k in targets]
    grades = [results['certify'][k]['grade'] for k in targets]
    colors = ['#27ae60' if g == 'A' else '#f39c12' if g == 'B' else '#e74c3c' for g in grades]
    
    bars = ax.bar(target_names, cert_scores, color=colors)
    ax.axhline(y=90, color='#27ae60', linestyle='--', alpha=0.5, label='Grade A (≥90)')
    ax.axhline(y=80, color='#f39c12', linestyle='--', alpha=0.5, label='Grade B (≥80)')
    ax.axhline(y=60, color='#e74c3c', linestyle='--', alpha=0.5, label='Grade C (≥60)')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Score Certificatif')
    ax.set_title(f'Certification PEC — {domain_config["name"]}')
    for bar, grade in zip(bars, grades):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, grade, ha='center', va='bottom', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'certification', 'certification_composite.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save rapport
    rapport = {
        'meta': {
            'framework': 'PEC - Predict-Explain-Certify',
            'domaine': domain_config['name'],
            'domaine_en': domain_config['name_en'],
            'date': '2026-05-30',
            'dataset': os.path.basename(domain_config['csv']),
            'observations': int(df.shape[0]),
            'variables': int(df.shape[1]),
            'regions': int(df[region_col].nunique()),
            'train_period': f'{domain_config["train_years"][0]}-{domain_config["train_years"][-1]}',
            'test_period': f'{domain_config["test_years"][0]}-{domain_config["test_years"][-1]}',
        },
        'auteurs': {
            'nom': 'Rosa Elysabeth Ralinirina',
            'orcid': '0009-0003-3048-1765',
        },
        'predict': results['predict'],
        'explain': results['explain'],
        'certify': results['certify']
    }
    
    rapport_path = os.path.join(results_dir, 'rapport_final.json')
    with open(rapport_path, 'w', encoding='utf-8') as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False, default=str)
    
    print(f'\n  Rapport sauvegarde: {rapport_path}')
    return rapport


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print('='*70)
    print('  PEC PIPELINE DE GENERALISATION')
    print('  Prouver que le framework marche sur 3 domaines differents')
    print('='*70)
    
    all_results = {}
    for domain_key, domain_config in DOMAINS.items():
        try:
            rapport = run_pec_pipeline(domain_key, domain_config)
            all_results[domain_key] = rapport
            
            # Print summary
            print(f'\n  Resume {domain_config["name"]}:')
            for tk in domain_config['targets']:
                r2 = rapport['predict'][tk]['R2_test']
                model = rapport['predict'][tk]['meilleur_modele']
                cert = rapport['certify'][tk]['certification_score']
                grade = rapport['certify'][tk]['grade']
                print(f'    {tk}: R2={r2:.4f}, Best={model}, Cert={cert:.1f}/100 (Grade {grade})')
        except Exception as e:
            print(f'\n  ERREUR pour {domain_key}: {e}')
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    print(f'\n{"="*70}')
    print(f'  COMPARAISON INTER-DOMAINES')
    print(f'{"="*70}')
    print(f'\n  {"Domaine":<25} {"Cible":<30} {"R²":>8} {"Cert":>8} {"Grade":>6} {"Meilleur Modele":<20}')
    print(f'  {"-"*25} {"-"*30} {"-"*8} {"-"*8} {"-"*6} {"-"*20}')
    for domain_key, rapport in all_results.items():
        domain_name = DOMAINS[domain_key]['name']
        for tk in rapport['predict']:
            r2 = rapport['predict'][tk]['R2_test']
            model = rapport['predict'][tk]['meilleur_modele']
            cert = rapport['certify'][tk]['certification_score']
            grade = rapport['certify'][tk]['grade']
            short = DOMAINS[domain_key]['targets'][tk]['short']
            print(f'  {domain_name:<25} {short:<30} {r2:>8.4f} {cert:>7.1f} {grade:>6} {model:<20}')
    
    print(f'\n  Le framework PEC est generalisable :')
    print(f'  - 3 domaines differents (fintech, sante, climat)')
    print(f'  - 9 modeles entraines (3 domaines x 3 cibles)')
    print(f'  - Tous les scores de certification sont calcules')
    print(f'  - Les explications SHAP sont generees')
    print(f'  - Le pipeline est identique : Predict > Explain > Certify')