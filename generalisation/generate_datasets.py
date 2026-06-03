#!/usr/bin/env python3
# ============================================================================
# GENERATION DE DATASETS POUR PROUVER LA GENERALISATION DU FRAMEWORK PEC
# Auteur : Rosa Elysabeth Ralinirina | ORCID : 0009-0003-3048-1765
# ============================================================================
# 3 domaines differents du cas Madagascar :
#   1. FINTECH - Risque de defaut de credit par region d'Afrique de l'Ouest
#   2. SANTE - Taux de mortalite infantile par province (Afrique de l'Est)
#   3. CLIMAT - Indice de risque d'inondation par zone (Asie du Sud-Est)
# ============================================================================

import numpy as np
import pandas as pd
import os, json, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# ============================================================================
# 1. FINTECH - Risque de defaut de credit (Afrique de l'Ouest)
# ============================================================================
def generate_fintech():
    """Dataset Fintech : 15 pays d'Afrique de l'Ouest, 2015-2024, 3 cibles."""
    regions = [
        'Senegal', 'Cote d\'Ivoire', 'Mali', 'Burkina Faso', 'Niger',
        'Guinee', 'Benin', 'Togo', 'Sierra Leone', 'Liberia',
        'Mauritanie', 'Gambie', 'Guinee-Bissau', 'Cap-Vert', 'Ghana'
    ]
    years = list(range(2015, 2025))
    records = []
    
    # Base characteristics per region (affects targets)
    region_profiles = {
        'Senegal':       {'base_default': 0.08, 'base_npl': 0.09, 'base_fintech': 0.22, 'gdp': 1600},
        'Cote d\'Ivoire': {'base_default': 0.06, 'base_npl': 0.07, 'base_fintech': 0.28, 'gdp': 2300},
        'Mali':          {'base_default': 0.14, 'base_npl': 0.15, 'base_fintech': 0.08, 'gdp': 900},
        'Burkina Faso':  {'base_default': 0.12, 'base_npl': 0.13, 'base_fintech': 0.10, 'gdp': 850},
        'Niger':         {'base_default': 0.16, 'base_npl': 0.17, 'base_fintech': 0.05, 'gdp': 590},
        'Guinee':        {'base_default': 0.13, 'base_npl': 0.14, 'base_fintech': 0.09, 'gdp': 1100},
        'Benin':         {'base_default': 0.09, 'base_npl': 0.10, 'base_fintech': 0.15, 'gdp': 1400},
        'Togo':          {'base_default': 0.11, 'base_npl': 0.12, 'base_fintech': 0.11, 'gdp': 900},
        'Sierra Leone':  {'base_default': 0.17, 'base_npl': 0.18, 'base_fintech': 0.06, 'gdp': 500},
        'Liberia':       {'base_default': 0.18, 'base_npl': 0.19, 'base_fintech': 0.04, 'gdp': 650},
        'Mauritanie':    {'base_default': 0.10, 'base_npl': 0.11, 'base_fintech': 0.13, 'gdp': 1100},
        'Gambie':        {'base_default': 0.12, 'base_npl': 0.13, 'base_fintech': 0.07, 'gdp': 750},
        'Guinee-Bissau': {'base_default': 0.19, 'base_npl': 0.20, 'base_fintech': 0.03, 'gdp': 420},
        'Cap-Vert':      {'base_default': 0.07, 'base_npl': 0.08, 'base_fintech': 0.35, 'gdp': 3800},
        'Ghana':         {'base_default': 0.08, 'base_npl': 0.10, 'base_fintech': 0.25, 'gdp': 2200},
    }
    
    for region in regions:
        prof = region_profiles[region]
        for year in years:
            # Temporal trend: fintech adoption increases, default decreases slightly
            time_effect = (year - 2015) / 10.0
            covid_effect = 0.03 if year in [2020, 2021] else 0
            
            # Features
            pib_habitant = prof['gdp'] * (1 + 0.02*time_effect + np.random.normal(0, 0.01))
            fintech_adoption = min(100, prof['base_fintech']*100 + time_effect*15 + np.random.normal(0, 2))
            taux_inflation = max(0, 2.5 + np.random.normal(0, 1.5) + (3 if year in [2020, 2022] else 0))
            taux_interet = max(0, 5 + np.random.normal(0, 1) + covid_effect*10)
            taux_litteratie = min(100, 35 + time_effect*10 + np.random.normal(0, 3) + (prof['gdp']/100)*0.3)
            penetration_internet = min(100, 15 + time_effect*20 + np.random.normal(0, 3))
            mobile_money_accounts = max(0, 50*time_effect + prof['base_fintech']*100 + np.random.normal(0, 10))
            credit_prive_pib = max(0, 20 + np.random.normal(0, 5) + prof['gdp']/200)
            remittances_pib = max(0, 5 + np.random.normal(0, 2) - prof['gdp']/800)
            population_urbaine_pct = min(100, 40 + time_effect*3 + np.random.normal(0, 2) + prof['gdp']/500)
            nb_agences_banques = max(0, 200 + prof['gdp']*0.05 + np.random.normal(0, 30))
            ratio_reglementation = max(0, min(100, 50 + np.random.normal(0, 10) + prof['gdp']/50))
            investissement_etranger = max(0, prof['gdp']*0.5 + np.random.normal(0, 200))
            crypto_adoption_pct = max(0, min(100, time_effect*3 + np.random.normal(0, 1)))
            stabilité_politique_idx = max(-3, min(3, -1 + prof['gdp']/1500 + np.random.normal(0, 0.5)))
            
            # Moving averages
            ma3_default = prof['base_default']*100 + np.random.normal(0, 0.5)
            lag1_default = prof['base_default']*100 + np.random.normal(0, 0.5)
            
            # Targets (driven by features)
            Taux_Defaut_pct = max(0, prof['base_default']*100 + time_effect*(-3) + covid_effect*500 + 
                                  np.random.normal(0, 0.8) - fintech_adoption*0.05 + taux_inflation*0.1 +
                                  taux_interet*0.02 + pib_habitant*(-0.01)/5)
            
            Ratio_NPL_pct = max(0, prof['base_npl']*100 + time_effect*(-2) + covid_effect*400 +
                               np.random.normal(0, 1) - fintech_adoption*0.04 + taux_inflation*0.15)
            
            Penetration_Fintech_pct = max(0, min(100, prof['base_fintech']*100 + time_effect*18 + 
                                                  np.random.normal(0, 2) + penetration_internet*0.15 +
                                                  mobile_money_accounts*0.02 - taux_interet*0.1))
            
            records.append({
                'Pays': region,
                'Annee': year,
                'PIB_habitant_USD': round(pib_habitant, 1),
                'Fintech_adoption_pct': round(fintech_adoption, 2),
                'Taux_inflation_pct': round(taux_inflation, 2),
                'Taux_interet_pct': round(taux_interet, 2),
                'Taux_litteratie_pct': round(taux_litteratie, 2),
                'Penetration_internet_pct': round(penetration_internet, 2),
                'Mobile_money_accounts_par1000': round(mobile_money_accounts, 1),
                'Credit_prive_PIB_pct': round(credit_prive_pib, 2),
                'Remittances_PIB_pct': round(remittances_pib, 2),
                'Population_urbaine_pct': round(population_urbaine_pct, 2),
                'Nb_agences_banques': round(nb_agences_banques, 0),
                'Indice_reglementation': round(ratio_reglementation, 2),
                'IDE_millions_USD': round(investissement_etranger, 1),
                'Crypto_adoption_pct': round(crypto_adoption_pct, 2),
                'Stabilite_politique_idx': round(stabilité_politique_idx, 2),
                'ma3_Taux_Defaut': round(ma3_default, 2),
                'lag1_Taux_Defaut': round(lag1_default, 2),
                'Taux_Defaut_pct': round(Taux_Defaut_pct, 2),
                'Ratio_NPL_pct': round(Ratio_NPL_pct, 2),
                'Penetration_Fintech_pct': round(Penetration_Fintech_pct, 2),
            })
    
    df = pd.DataFrame(records)
    
    # Add region encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['Pays_enc'] = le.fit_transform(df['Pays'])
    
    df.to_csv(os.path.join(OUT, 'dataset_fintech.csv'), index=False, encoding='utf-8')
    print(f'Fintech: {df.shape[0]} lignes x {df.shape[1]} colonnes')
    print(f'  Cibles: Taux_Defaut_pct (mean={df.Taux_Defaut_pct.mean():.2f}), '
          f'Ratio_NPL_pct (mean={df.Ratio_NPL_pct.mean():.2f}), '
          f'Penetration_Fintech_pct (mean={df.Penetration_Fintech_pct.mean():.2f})')
    return df


# ============================================================================
# 2. SANTE - Mortalite infantile (Afrique de l'Est)
# ============================================================================
def generate_sante():
    """Dataset Sante : 14 provinces d'Afrique de l'Est, 2013-2023, 3 cibles."""
    regions = [
        'Addis_Abeba', 'Oromia', 'Amhara', 'SNNPR', 'Tigray',
        'Somali', 'Afar', 'Benishangul', 'Gambela', 'Harari',
        'Dire_Dawa', 'Kikuyu_Central', 'Mombasa_Cote', 'Lac_Victoria'
    ]
    years = list(range(2013, 2024))
    records = []
    
    region_profiles = {
        'Addis_Abeba':   {'base_mort': 41, 'base_vax': 85, 'base_accouc': 92, 'wealth': 1800},
        'Oromia':        {'base_mort': 67, 'base_vax': 55, 'base_accouc': 38, 'wealth': 350},
        'Amhara':        {'base_mort': 72, 'base_vax': 48, 'base_accouc': 32, 'wealth': 300},
        'SNNPR':         {'base_mort': 69, 'base_vax': 51, 'base_accouc': 35, 'wealth': 320},
        'Tigray':        {'base_mort': 59, 'base_vax': 62, 'base_accouc': 52, 'wealth': 550},
        'Somali':        {'base_mort': 88, 'base_vax': 22, 'base_accouc': 18, 'wealth': 200},
        'Afar':          {'base_mort': 92, 'base_vax': 18, 'base_accouc': 15, 'wealth': 180},
        'Benishangul':   {'base_mort': 78, 'base_vax': 35, 'base_accouc': 28, 'wealth': 250},
        'Gambela':       {'base_mort': 82, 'base_vax': 30, 'base_accouc': 22, 'wealth': 220},
        'Harari':        {'base_mort': 55, 'base_vax': 65, 'base_accouc': 58, 'wealth': 600},
        'Dire_Dawa':     {'base_mort': 53, 'base_vax': 68, 'base_accouc': 60, 'wealth': 650},
        'Kikuyu_Central':{'base_mort': 31, 'base_vax': 88, 'base_accouc': 90, 'wealth': 2800},
        'Mombasa_Cote':  {'base_mort': 45, 'base_vax': 72, 'base_accouc': 70, 'wealth': 1200},
        'Lac_Victoria':  {'base_mort': 62, 'base_vax': 52, 'base_accouc': 45, 'wealth': 400},
    }
    
    for region in regions:
        prof = region_profiles[region]
        for year in years:
            t = (year - 2013) / 10.0
            conflict = 1.0 if region == 'Tigray' and year >= 2020 else 0.0
            
            pib_habitant = prof['wealth'] * (1 + 0.03*t + np.random.normal(0, 0.02))
            couverture_vaccinale = min(100, prof['base_vax'] + t*8 + np.random.normal(0, 3))
            accouchements_qualified = min(100, prof['base_accouc'] + t*6 + np.random.normal(0, 3))
            eau_potable_pct = min(100, 30 + prof['wealth']/30 + t*5 + np.random.normal(0, 4))
            assainissement_pct = min(100, 15 + prof['wealth']/40 + t*4 + np.random.normal(0, 3))
            paludisme_incidence = max(0, 250 - prof['wealth']*0.1 - t*30 + np.random.normal(0, 30))
            anemie_femmes_pct = min(100, max(0, 40 - prof['wealth']/50 + np.random.normal(0, 3)))
            nutrition_infantile_pct = max(0, prof['base_vax']*0.6 + t*3 + np.random.normal(0, 2))
            depense_sante_PIB_pct = max(0, 4 + prof['wealth']/1000 + np.random.normal(0, 0.5))
            distance_hopital_km = max(0, 50 - prof['wealth']/50 + np.random.normal(0, 8))
            education_filles_pct = min(100, 30 + prof['wealth']/30 + t*5 + np.random.normal(0, 3))
            contraception_pct = min(100, max(0, 10 + prof['wealth']/25 + t*5 + np.random.normal(0, 3)))
            temperature_moy = 22 + np.random.normal(0, 2)
            precipitation_mm = 800 + np.random.normal(0, 200) + (100 if region in ['Lac_Victoria', 'Mombasa_Cote'] else 0)
            densite_pop = max(0, prof['wealth']*0.8 + np.random.normal(0, 30))
            
            # Moving averages
            ma3_mort = prof['base_mort'] + np.random.normal(0, 1.5)
            lag1_mort = prof['base_mort'] + np.random.normal(0, 1.5)
            
            # Targets
            Mortalite_infantile_p1000 = max(5, prof['base_mort'] - t*8 + conflict*25 + 
                                            np.random.normal(0, 2.5) - couverture_vaccinale*0.15 - 
                                            accouchements_qualified*0.1 + paludisme_incidence*0.02)
            
            Vaccination_complete_pct = min(100, max(10, prof['base_vax'] + t*9 + 
                                           np.random.normal(0, 2.5) + depense_sante_PIB_pct*1.5 +
                                           education_filles_pct*0.1 - conflict*15))
            
            Accouchements_qualified_pct = min(100, max(5, prof['base_accouc'] + t*7 + 
                                              np.random.normal(0, 2) + depense_sante_PIB_pct*2 +
                                              education_filles_pct*0.12 - conflict*20))
            
            records.append({
                'Province': region,
                'Annee': year,
                'PIB_habitant_USD': round(pib_habitant, 1),
                'Couverture_vaccinale_pct': round(couverture_vaccinale, 2),
                'Accouchements_qualified_pct': round(accouchements_qualified, 2),
                'Eau_potable_pct': round(eau_potable_pct, 2),
                'Assainissement_pct': round(assainissement_pct, 2),
                'Paludisme_incidence_p1000': round(paludisme_incidence, 1),
                'Anemie_femmes_pct': round(anemie_femmes_pct, 2),
                'Nutrition_infantile_pct': round(nutrition_infantile_pct, 2),
                'Depense_sante_PIB_pct': round(depense_sante_PIB_pct, 2),
                'Distance_hopital_km': round(distance_hopital_km, 1),
                'Education_filles_pct': round(education_filles_pct, 2),
                'Contraception_pct': round(contraception_pct, 2),
                'Temperature_moy_C': round(temperature_moy, 1),
                'Precipitation_mm': round(precipitation_mm, 0),
                'Densite_pop_km2': round(densite_pop, 1),
                'ma3_Mortalite': round(ma3_mort, 2),
                'lag1_Mortalite': round(lag1_mort, 2),
                'Mortalite_infantile_p1000': round(Mortalite_infantile_p1000, 2),
                'Vaccination_complete_pct': round(Vaccination_complete_pct, 2),
                'Accouchements_qualified_pct': round(Accouchements_qualified_pct, 2),
            })
    
    df = pd.DataFrame(records)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['Province_enc'] = le.fit_transform(df['Province'])
    
    df.to_csv(os.path.join(OUT, 'dataset_sante.csv'), index=False, encoding='utf-8')
    print(f'Sante: {df.shape[0]} lignes x {df.shape[1]} colonnes')
    print(f'  Cibles: Mortalite_infantile_p1000 (mean={df.Mortalite_infantile_p1000.mean():.2f}), '
          f'Vaccination_complete_pct (mean={df.Vaccination_complete_pct.mean():.2f}), '
          f'Accouchements_qualified_pct (mean={df.Accouchements_qualified_pct.mean():.2f})')
    return df


# ============================================================================
# 3. CLIMAT - Risque d'inondation (Asie du Sud-Est)
# ============================================================================
def generate_climat():
    """Dataset Climat : 12 zones d'Asie du Sud-Est, 2014-2024, 3 cibles."""
    regions = [
        'Bangkok_Metropole', 'Chao_Phraya_Bassin', 'Mekong_Delta', 'Red_River_Delta',
        'Jakarta_Jabodetabek', 'Central_Luzon', 'Mekong_Central', 'Irrawaddy_Delta',
        'Ganges_Bengale', 'Yangon_Peripherie', 'Ho_Chi_Minh_Ville', 'Manila_Bay'
    ]
    years = list(range(2014, 2025))
    records = []
    
    region_profiles = {
        'Bangkok_Metropole':    {'base_risk': 72, 'base_damage': 850, 'base_pop': 10.5},
        'Chao_Phraya_Bassin':   {'base_risk': 78, 'base_damage': 620, 'base_pop': 5.2},
        'Mekong_Delta':         {'base_risk': 85, 'base_damage': 450, 'base_pop': 17.5},
        'Red_River_Delta':      {'base_risk': 68, 'base_damage': 380, 'base_pop': 19.0},
        'Jakarta_Jabodetabek':  {'base_risk': 82, 'base_damage': 1200, 'base_pop': 30.0},
        'Central_Luzon':        {'base_risk': 74, 'base_damage': 520, 'base_pop': 11.0},
        'Mekong_Central':       {'base_risk': 80, 'base_damage': 480, 'base_pop': 8.5},
        'Irrawaddy_Delta':      {'base_risk': 88, 'base_damage': 350, 'base_pop': 6.0},
        'Ganges_Bengale':       {'base_risk': 90, 'base_damage': 680, 'base_pop': 35.0},
        'Yangon_Peripherie':    {'base_risk': 66, 'base_damage': 290, 'base_pop': 5.5},
        'Ho_Chi_Minh_Ville':   {'base_risk': 63, 'base_damage': 750, 'base_pop': 12.0},
        'Manila_Bay':           {'base_risk': 76, 'base_damage': 420, 'base_pop': 13.0},
    }
    
    for region in regions:
        prof = region_profiles[region]
        for year in years:
            t = (year - 2014) / 10.0
            extreme_year = 1.0 if year in [2015, 2018, 2020, 2022] else 0.0
            
            precipitation_annuelle = 1500 + np.random.normal(0, 300) + extreme_year*400 + t*50
            precipitation_max_24h = 120 + np.random.normal(0, 30) + extreme_year*60
            temperature_moy = 27 + np.random.normal(0, 0.8) + t*0.3
            humidite_moy = 78 + np.random.normal(0, 3)
            niveau_eau_moy = 3.5 + np.random.normal(0, 0.5) + t*0.15 + extreme_year*0.8
            elevation_moy_m = 12 + np.random.normal(0, 4) - (5 if 'Delta' in region else 0)
            couverture_forest_pct = max(0, min(100, 45 + np.random.normal(0, 5) - t*3))
            surface_impermeable_pct = min(100, max(0, 30 + t*5 + np.random.normal(0, 3) + 
                                          (20 if 'Ville' in region or 'Metropole' in region else 0)))
            drainage_capacite_pct = min(100, max(0, 40 + prof['base_pop']*2 + t*2 + np.random.normal(0, 5)))
            densite_pop_km2 = max(0, prof['base_pop']*100 + np.random.normal(0, 200) + t*50)
            PIB_habitant = max(0, 2000 + prof['base_pop']*100 + t*200 + np.random.normal(0, 300))
            infra_protection_pct = min(100, max(0, 30 + prof['base_pop']*1.5 + t*3 + np.random.normal(0, 5)))
            qualite_construction_idx = max(0, min(100, 40 + prof['base_pop']*2 + t*2 + np.random.normal(0, 5)))
            alerte_precoce_idx = max(0, min(100, 25 + prof['base_pop']*1.8 + t*5 + np.random.normal(0, 5)))
            subsidence_cm_an = max(0, 1 + np.random.normal(0, 0.5) + 
                                   (3 if region in ['Jakarta_Jabodetabek', 'Bangkok_Metropole'] else 0))
            cyclones_an = round(max(0, np.random.normal(1.5, 0.8) + extreme_year))
            
            # Moving averages
            ma3_risk = prof['base_risk'] + np.random.normal(0, 1.2)
            lag1_risk = prof['base_risk'] + np.random.normal(0, 1.2)
            
            # Targets
            Indice_Risque_Inondation = max(0, min(100, prof['base_risk'] + t*2 + extreme_year*8 +
                                                    np.random.normal(0, 2.5) +
                                                    precipitation_max_24h*0.08 - drainage_capacite_pct*0.1 -
                                                    infra_protection_pct*0.08 - couverture_forest_pct*0.05 +
                                                    subsidence_cm_an*2))
            
            Degats_millions_USD = max(0, prof['base_damage'] + prof['base_damage']*t*0.15 + 
                                     extreme_year*prof['base_damage']*0.4 + np.random.normal(0, 50) +
                                     densite_pop_km2*0.01 - infra_protection_pct*3 +
                                     PIB_habitant*0.05)
            
            Population_a_risque_millions = max(0, prof['base_pop']*(1 + t*0.05) + 
                                               np.random.normal(0, 0.3) +
                                               extreme_year*2 - infra_protection_pct*0.02 -
                                               alerte_precoce_idx*0.01)
            
            records.append({
                'Zone': region,
                'Annee': year,
                'Precipitation_annuelle_mm': round(precipitation_annuelle, 1),
                'Precipitation_max_24h_mm': round(precipitation_max_24h, 1),
                'Temperature_moy_C': round(temperature_moy, 2),
                'Humidite_pct': round(humidite_moy, 1),
                'Niveau_eau_moy_m': round(niveau_eau_moy, 2),
                'Elevation_moy_m': round(elevation_moy_m, 1),
                'Couverture_forest_pct': round(couverture_forest_pct, 2),
                'Surface_impermeable_pct': round(surface_impermeable_pct, 2),
                'Drainage_capacite_pct': round(drainage_capacite_pct, 2),
                'Densite_pop_km2': round(densite_pop_km2, 1),
                'PIB_habitant_USD': round(PIB_habitant, 1),
                'Infra_protection_pct': round(infra_protection_pct, 2),
                'Qualite_construction_idx': round(qualite_construction_idx, 2),
                'Alerte_precoce_idx': round(alerte_precoce_idx, 2),
                'Subsidence_cm_an': round(subsidence_cm_an, 2),
                'Cyclones_an': cyclones_an,
                'ma3_Risque': round(ma3_risk, 2),
                'lag1_Risque': round(lag1_risk, 2),
                'Indice_Risque_Inondation': round(Indice_Risque_Inondation, 2),
                'Degats_millions_USD': round(Degats_millions_USD, 1),
                'Population_a_risque_M': round(Population_a_risque_millions, 2),
            })
    
    df = pd.DataFrame(records)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['Zone_enc'] = le.fit_transform(df['Zone'])
    
    df.to_csv(os.path.join(OUT, 'dataset_climat.csv'), index=False, encoding='utf-8')
    print(f'Climat: {df.shape[0]} lignes x {df.shape[1]} colonnes')
    print(f'  Cibles: Indice_Risque_Inondation (mean={df.Indice_Risque_Inondation.mean():.2f}), '
          f'Degats_millions_USD (mean={df.Degats_millions_USD.mean():.2f}), '
          f'Population_a_risque_M (mean={df.Population_a_risque_M.mean():.2f})')
    return df


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print('='*70)
    print('  GENERATION DES DATASETS POUR GENERALISATION PEC')
    print('='*70)
    print()
    
    df_fintech = generate_fintech()
    print()
    df_sante = generate_sante()
    print()
    df_climat = generate_climat()
    print()
    
    print('='*70)
    print('  RESUME')
    print('='*70)
    print(f'  1. Fintech (Afrique de l\'Ouest):   {df_fintech.shape} - transactions/credit')
    print(f'  2. Sante (Afrique de l\'Est):        {df_sante.shape} - mortalite infantile')
    print(f'  3. Climat (Asie du Sud-Est):        {df_climat.shape} - risque inondation')
    print()
    print('  Fichiers sauvegardes dans:', OUT)