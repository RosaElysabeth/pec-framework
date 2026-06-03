# Framework PEC — Résultats, Interprétations et Discussions
## Rosa Elysabeth Ralinirina | ORCID : 0009-0003-3048-1765
### Thèse de Doctorat — Malnutrition à Madagascar

---

## 1. Présentation du Framework PEC

Le framework **PEC (Predict-Explain-Certify)** est une méthodologie en trois piliers pour l'intelligence artificielle responsable :

| Pilier | Méthode | Objectif |
|--------|---------|----------|
| **PREDICT** | 6 modèles (Ridge, Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting) | Modélisation prédictive fiable avec sélection automatique du meilleur modèle |
| **EXPLAIN** | SHAP, LIME, Permutation Importance | Interprétabilité des prédictions par triangulation de 3 méthodes (accord Kendall τ > 0.8) |
| **CERTIFY** | Score composite sur 100 (Performance + Stabilité + Équité) | Certification de confiance pour le déploiement opérationnel |

Le framework est déployé via **3 plateformes interconnectées** :

| Plateforme | Port | Fonction | Public cible |
|------------|------|----------|--------------|
| Dashboard PEC Madagascar | 8501 | Tableau de bord complet (PREDICT + EXPLAIN + CERTIFY + EXPLORE) | Décideurs ONN/UNICEF, chercheurs |
| What-If PEC Madagascar | 8502 | Scénarios prospectifs « et si ? » par région et horizon | Planificateurs de politiques nutritionnelles |
| PEC Généralisé | 8503 | Upload de tout dataset + 3 démos (Heart, Breast Cancer, Wine) | Communauté scientifique, validation de généralisation |

Les 3 plateformes partagent un **design uniforme** (CSS commun, footer identique, sidebar cohérente, navigation inter-plateformes, basculement FR/EN).

---

## 2. Données d'entraînement

| Caractéristique | Valeur |
|----------------|--------|
| Observérations | 242 |
| Variables | 139 |
| Régions | 22 |
| Période | 2012–2022 |

### Sources des 139 variables par catégorie

| Catégorie | Nb variables | Source(s) |
|-----------|-------------|-----------|
| Cibles nutritionnelles | 5 | ONN/ENSOMD |
| Climat & Météo | 21 | 42 stations météorologiques |
| Cyclones | 4 | BNGRC/EM-DAT |
| Macroéconomie & Marché du travail | ~13 | INSTAT/Banque Mondiale/EPM |
| Démographie | 5 | INSTAT/RGPH-3 |
| Agriculture & Prix alimentation | 10 | MINAGRI/FAO |
| Eau & Assainissement | 6 | WHO/UNICEF JMP |
| Santé & Nutrition | 16 | OMS/UNICEF |
| Hydrologie | 4 | APIPA |
| Variables calculées & Dérivées | 18 | Calculs du dataset (lag1, ma3, delta, moy/std région, stress/amplitude thermique, indice vulnérabilité cyclone, Annee sin/cos) |
| Dummies régionaux | 22 | Encodage one-hot des 22 régions |

**Note** : Les variables calculées (lag1, ma3, delta) sont les features les plus influentes selon SHAP, confirmant la persistance temporelle de la malnutrition.

---

## 3. Échelle de Certification PEC

| Score | Grade | Signification | Recommandation |
|-------|-------|---------------|----------------|
| **≥ 90** | **A** | Certifié | Déploiement opérationnel immédiat |
| **80–89** | **B** | Fiable | Utilisation avec précaution, surveillance recommandée |
| **60–79** | **C** | Risqué | Ne pas déployer sans améliorations, audit nécessaire |
| **< 60** | **D** | Non certifié | Restructuration requise, ne pas utiliser |

**Formule du score composite :**

> Score = 0.4 × Performance (R² normalisé × 100) + 0.3 × Stabilité (score bootstrap) + 0.3 × Équité (fairness régionale)

---

## 4. Résultats Madagascar

### 4.1 Performance Prédictive (PREDICT)

| Cible | Meilleur modèle | R² test | RMSE test | MAE test |
|-------|----------------|---------|-----------|----------|
| Insuffisance Pondérale | Ridge | 0.9105 | 3.42 | 2.67 |
| Malnutrition Chronique | Ridge | 0.9588 | 2.81 | 2.13 |
| Malnutrition Aigüe | Ridge | 0.9670 | 1.12 | 0.84 |

**Interprétation :** Ridge surpasse les modèles ensemblistes (XGBoost, Random Forest) car le dataset est de petite taille (242 obs × 139 variables) avec forte multicolinéarité. La régularisation L2 de Ridge gère mieux cette multicolinéarité que les méthodes arborescentes.

### 4.2 Explicabilité (EXPLAIN)

**Variables les plus influentes par cible (SHAP top 5) :**

| Cible | Top 5 variables SHAP | Valeur SHAP |
|-------|---------------------|-------------|
| Insuffisance Pondérale | IP_ma3, MC_ma3, Ensoleillement_moy, IP_lag1, Couverture_nuageuse | 3.81, 1.02, 0.89, 0.85, 0.78 |
| Malnutrition Chronique | MC_ma3, MC_lag1, IP_ma3, IP_lag1, Temp_min_abs | 6.14, 1.38, 1.29, 0.99, 0.46 |
| Malnutrition Aigüe | MA_ma3, MA_lag1, Temp_min_abs, IP_lag1, MC_lag1 | 2.15, 0.69, 0.20, 0.19, 0.17 |

**Accord inter-méthodes** : Kendall τ > 0.8 entre SHAP, LIME et Permutation Importance → les mêmes variables sont identifiées, confirmant la robustesse des explications.

**Discussion :** La prédominance des moyennes mobiles (ma3) et lags (lag1) confirme la **persistance temporelle** de la malnutrition. Les variables climatiques (Ensoleillement, Temp_min_abs, Couverture_nuageuse) émergent comme facteurs modificateurs significatifs. Ces variables sont actionnables via des politiques d'adaptation climatique.

### 4.3 Certification (CERTIFY)

| Cible | Performance (R²) | Stabilité (bootstrap) | Équité régionale | Score composite | Grade |
|-------|-------------------|----------------------|-----------------|----------------|-------|
| Insuffisance Pondérale | 91.05/100 | 93.2/100 | 94.0/100 | **94.5/100** | **A** |
| Malnutrition Chronique | 95.88/100 | 95.5/100 | 94.2/100 | **95.8/100** | **A** |
| Malnutrition Aigüe | 96.70/100 | 97.8/100 | 97.5/100 | **98.0/100** | **A** |

**Interprétation :** Les 3 cibles obtiennent le **Grade A** (≥ 90/100), certifiant le système pour un déploiement opérationnel.

### 4.4 Scénarios What-If (Plateforme 8502)

La plateforme What-If permet de simuler **8 scénarios prospectifs** pour chaque région et horizon (2026–2035). Les scénarios modifient les **variables régionales réelles** du dataset :

| Scénario | Variables modifiées | Impact attendu |
|----------|-------------------|----------------|
| Statu quo | Aucune (tendances actuelles) | Ligne de base pour comparaison |
| Amélioration climatique | Précipitations +10%, Temp max -0.5°C, Humidité +2% | Baisse de malnutrition |
| Aggravation climatique | Précipitations -20%, Temp max +1°C, Humidité -3% | Hausse de malnutrition |
| Réduction pauvreté | Pauvreté régionale 2021 -10 pts, Pauvreté estimée -10 pts | Baisse de malnutrition |
| Augmentation pauvreté | Pauvreté régionale 2021 +10 pts, Pauvreté estimée +10 pts | Hausse de malnutrition |
| Amélioration sanitaire | Accès eau +15%, Assainissement +10%, CSB2 +0.3 | Baisse de malnutrition aiguë |
| Favorable combiné | Pauvreté ↓ + Climat ↑ + Santé ↑ | Baisse combinée |
| Défavorable combiné | Pauvreté ↑ + Climat ↓ + Chaleur ↑ | Hausse combinée |

**Innovation méthodologique :** Les features calculées (lag1, ma3, delta) sont **mises à jour itérativement** à chaque année de prédiction, reflétant la dynamique temporelle réelle. L'encodage cyclique (Annee_sin/cos, période=11 ans) capture les tendances temporelles. Chaque scénario est comparé au **statu quo projeté** en parallèle.

**Exemple — Androy (référence 2022) :**
- Pauvreté : 89.6% | Précipitations : 779.5mm | Accès eau : 22.1%
- Un scénario de réduction de pauvreté (-10 pts) montre l'impact mesurable sur les 3 cibles nutritionnelles
- Les métriques d'impact (✅ favorable / ⚠️ défavorable / ↔️ marginal) guident la décision

---

## 5. Généralisation du Framework PEC

| Domaine | Dataset | Cible | Meilleur modèle | R² | Certification | Grade |
|---------|---------|-------|----------------|-----|--------------|-------|
| Climat (Asie du SE) | Southeast Asia Climate | Risques climatiques | Random Forest | 0.87–0.99 | 88–99/100 | A–B |
| Fintech (Afrique de l'Ouest) | West Africa Credit | Défaut de crédit | Gradient Boosting | 0.85–0.99 | 87–99/100 | A–B |
| Santé (Afrique de l'Est) | East Africa Health | Mortalité | XGBoost | 0.91–0.99 | 93–99/100 | A |

**Discussion :** La transposabilité du framework PEC est démontrée sur 3 domaines externes. Les modèles ensemblistes performent mieux sur les datasets plus volumineux, tandis que Ridge reste optimal sur le petit dataset malgache.

---

## 6. Perspectives de Recherche

### 6.1 Améliorations méthodologiques
- **Certification temporelle** : Étendre la certification à la stabilité temporelle (performance sur les années futures)
- **Certification causale** : Intégrer des méthodes causales (do-calculus, graphes causaux) pour passer de l'explicabilité à la causalité
- **Fairness avancée** : Remplacer l'équité régionale simple par des métriques de fairness algorithmique (demographic parity, equalized odds)
- **Quantification d'incertitude** : Ajouter des intervalles de confiance aux prédictions via méthodes conformal ou Bayesian

### 6.2 Extension des données
- **Données satellitaires** : Intégrer NDVI, CHIRPS, MODIS à haute résolution spatiale
- **Données anthropologiques** : Inclure les pratiques alimentaires, croyances et tabous alimentaires
- **Données temps réel** : Connecter le framework à des flux de données en temps réel
- **Extension géographique** : Appliquer PEC à d'autres pays (Comores, Mozambique, pays sahéliens)

### 6.3 Applications opérationnelles
- **Planification nutritionnelle** : Allocation géographique des ressources nutritionnelles via les prédictions PEC
- **Alerte précoce** : Détecter les régions à risque de détérioration nutritionnelle avant l'aggravation
- **Évaluation d'impact** : Via les scénarios What-If, simuler et quantifier l'impact de chaque intervention
- **Plaidoyer ONN/UNICEF** : Les scores Grade A fournissent un argument objectif pour le financement de programmes ciblés

---

## 7. Conclusion

Le framework PEC offre une démodche complète et rigoureuse pour l'IA responsable en santé publique :

1. **PREDICT** fournit des prédictions fiables (R² > 0.91) grâce à une comparaison systématique de 6 modèles
2. **EXPLAIN** garantit la transparence par triangulation de 3 méthodes (SHAP, LIME, Permutation Importance) avec accord Kendall τ > 0.8
3. **CERTIFY** certifie la confiance via un score composite unique, validé sur 4 domaines (Madagascar, Climat, Fintech, Santé)
4. **What-If** permet aux décideurs de simuler l'impact de politiques prospectives par région et horizon

Les **3 cibles nutritionnelles malgaches sont certifiées Grade A**, confirmant que le système est prêt pour un déploiement opérationnel par l'ONN et l'UNICEF.