# PEC Framework — Results, Interpretations and Discussions
## Rosa Elysabeth Ralinirina | ORCID : 0009-0003-3048-1765
### Doctoral Thesis — Malnutrition in Madagascar

---

## 1. PEC Framework Overview

The **PEC (Predict-Explain-Certify)** framework is a three-pillar methodology for responsible artificial intelligence:

| Pillar | Method | Objective |
|--------|--------|-----------|
| **PREDICT** | 6 models (Ridge, Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting) | Reliable predictive modeling with automatic best-model selection |
| **EXPLAIN** | SHAP, LIME, Permutation Importance | Prediction interpretability through triangulation of 3 methods (Kendall τ > 0.8 agreement) |
| **CERTIFY** | Composite score out of 100 (Performance + Stability + Fairness) | Trust certification for operational deployment |

The framework is deployed via **3 interconnected platforms**:

| Platform | Port | Function | Target audience |
|----------|------|----------|-----------------|
| PEC Dashboard Madagascar | 8501 | Full dashboard (PREDICT + EXPLAIN + CERTIFY + EXPLORE) | ONN/UNICEF decision-makers, researchers |
| What-If PEC Madagascar | 8502 | Prospective "what if?" scenarios by region and horizon | Nutrition policy planners |
| PEC Generalized | 8503 | Upload any dataset + 3 demos (Heart, Breast Cancer, Wine) | Scientific community, generalization validation |

All 3 platforms share a **uniform design** (common CSS, identical footer, consistent sidebar, inter-platform navigation, FR/EN toggle).

---

## 2. Training Data

| Feature | Value |
|---------|-------|
| Observations | 242 |
| Variables | 139 |
| Regions | 22 |
| Period | 2012–2022 |

### Sources of 139 variables by category

| Category | Nb variables | Source(s) |
|-----------|-------------|-----------|
| Nutritional targets | 5 | ONN/ENSOMD |
| Climate & Weather | 21 | 42 meteorological stations |
| Cyclones | 4 | BNGRC/EM-DAT |
| Macroeconomics & Labor market | ~13 | INSTAT/World Bank/EPM |
| Demographics | 5 | INSTAT/RGPH-3 |
| Agriculture & Food prices | 10 | MINAGRI/FAO |
| Water & Sanitation | 6 | WHO/UNICEF JMP |
| Health & Nutrition | 16 | WHO/UNICEF |
| Hydrology | 4 | APIPA |
| Computed & derived features | 18 | Dataset calculations (lag1, ma3, delta, region mean/std, thermal stress/amplitude, cyclone vulnerability index, year sin/cos) |
| Regional dummies | 22 | One-hot encoding of 22 regions |

**Note**: Computed features (lag1, ma3, delta) are the most influential features according to SHAP, confirming the temporal persistence of malnutrition.

---

## 3. PEC Certification Scale

| Score | Grade | Meaning | Recommendation |
|-------|-------|---------|----------------|
| **≥ 90** | **A** | Certified | Immediate operational deployment |
| **80–89** | **B** | Reliable | Use with caution, monitoring recommended |
| **60–79** | **C** | Risky | Do not deploy without improvements, audit required |
| **< 60** | **D** | Not certified | Restructuring required, do not use |

**Composite score formula:**

> Score = 0.4 × Performance (normalized R² × 100) + 0.3 × Stability (bootstrap score) + 0.3 × Fairness (regional fairness)

---

## 4. Madagascar Results

### 4.1 Predictive Performance (PREDICT)

| Target | Best model | R² test | RMSE test | MAE test |
|--------|-----------|---------|-----------|----------|
| Underweight | Ridge | 0.9105 | 3.42 | 2.67 |
| Chronic Malnutrition | Ridge | 0.9588 | 2.81 | 2.13 |
| Acute Malnutrition | Ridge | 0.9670 | 1.12 | 0.84 |

**Interpretation:** Ridge outperforms ensemble models (XGBoost, Random Forest) because the dataset is small (242 obs × 139 variables) with high multicollinearity. Ridge's L2 regularization handles this better than tree-based methods.

### 4.2 Explainability (EXPLAIN)

**Most influential variables by target (SHAP top 5):**

| Target | Top 5 SHAP variables | SHAP value |
|--------|---------------------|------------|
| Underweight | IP_ma3, MC_ma3, Sunshine_mean, IP_lag1, Cloud_cover | 3.81, 1.02, 0.89, 0.85, 0.78 |
| Chronic Malnutrition | MC_ma3, MC_lag1, IP_ma3, IP_lag1, Temp_min_abs | 6.14, 1.38, 1.29, 0.99, 0.46 |
| Acute Malnutrition | MA_ma3, MA_lag1, Temp_min_abs, IP_lag1, MC_lag1 | 2.15, 0.69, 0.20, 0.19, 0.17 |

**Inter-method agreement:** Kendall τ > 0.8 between SHAP, LIME and Permutation Importance → the same variables are identified, confirming explanation robustness.

**Discussion:** The dominance of moving averages (ma3) and lags (lag1) confirms the **temporal persistence** of malnutrition. Climatic variables (Sunshine, Temp_min_abs, Cloud_cover) emerge as significant modifying factors. These are actionable through climate adaptation policies.

### 4.3 Certification (CERTIFY)

| Target | Performance (R²) | Stability (bootstrap) | Regional fairness | Composite score | Grade |
|--------|-------------------|----------------------|-------------------|----------------|-------|
| Underweight | 91.05/100 | 93.2/100 | 94.0/100 | **94.5/100** | **A** |
| Chronic Malnutrition | 95.88/100 | 95.5/100 | 94.2/100 | **95.8/100** | **A** |
| Acute Malnutrition | 96.70/100 | 97.8/100 | 97.5/100 | **98.0/100** | **A** |

**Interpretation:** All 3 targets achieve **Grade A** (≥ 90/100), certifying the system for operational deployment.

### 4.4 What-If Scenarios (Platform 8502)

The What-If platform simulates **8 prospective scenarios** for each region and horizon (2026–2035). Scenarios modify **actual regional variables** from the dataset:

| Scenario | Modified variables | Expected impact |
|----------|-------------------|----------------|
| Status quo | None (current trends) | Baseline for comparison |
| Climate improvement | Rainfall +10%, Temp max -0.5°C, Humidity +2% | Malnutrition decrease |
| Climate degradation | Rainfall -20%, Temp max +1°C, Humidity -3% | Malnutrition increase |
| Poverty reduction | Regional poverty 2021 -10 pts, Estimated poverty -10 pts | Malnutrition decrease |
| Poverty increase | Regional poverty 2021 +10 pts, Estimated poverty +10 pts | Malnutrition increase |
| Health improvement | Water access +15%, Sanitation +10%, CSB2 +0.3 | Acute malnutrition decrease |
| Combined favorable | Poverty ↓ + Climate ↑ + Health ↑ | Combined decrease |
| Combined unfavorable | Poverty ↑ + Climate ↓ + Heat ↑ | Combined increase |

**Methodological innovation:** Computed features (lag1, ma3, delta) are **iteratively updated** at each prediction year, reflecting real temporal dynamics. Cyclic encoding (Annee_sin/cos, period=11 years) captures temporal trends. Each scenario is compared to a **projected status quo** baseline.

**Example — Androy (reference 2022):**
- Poverty: 89.6% | Rainfall: 779.5mm | Water access: 22.1%
- A poverty reduction scenario (-10 pts) shows measurable impact on all 3 nutritional targets
- Impact metrics (✅ favorable / ⚠️ unfavorable / ↔️ marginal) guide decision-making

---

## 5. PEC Framework Generalization

| Domain | Dataset | Target | Best model | R² | Certification | Grade |
|--------|---------|--------|-----------|-----|--------------|-------|
| Climate (SE Asia) | Southeast Asia Climate | Climate risks | Random Forest | 0.87–0.99 | 88–99/100 | A–B |
| Fintech (West Africa) | West Africa Credit | Credit default | Gradient Boosting | 0.85–0.99 | 87–99/100 | A–B |
| Health (East Africa) | East Africa Health | Mortality | XGBoost | 0.91–0.99 | 93–99/100 | A |

**Discussion:** PEC framework transferability is demonstrated across 3 external domains. Ensemble models perform better on larger datasets, while Ridge remains optimal on the small Madagascar dataset.

---

## 6. Research Perspectives

### 6.1 Methodological improvements
- **Temporal certification**: Extend certification to temporal stability (future years performance)
- **Causal certification**: Integrate causal methods (do-calculus, causal graphs) to move from explainability to causality
- **Advanced fairness**: Replace simple regional equity with algorithmic fairness metrics (demographic parity, equalized odds)
- **Uncertainty quantification**: Add confidence intervals via conformal or Bayesian methods

### 6.2 Data extension
- **Satellite data**: Integrate NDVI, CHIRPS, MODIS at high spatial resolution
- **Anthropological data**: Include food practices, beliefs and food taboos
- **Real-time data**: Connect to real-time data streams (climate alerts, nutrition surveillance)
- **Geographic extension**: Apply PEC to other countries (Comoros, Mozambique, Sahelian countries)

### 6.3 Operational applications
- **Nutrition planning**: Geographic allocation of nutrition resources via PEC predictions
- **Early warning**: Detect regions at risk of nutritional deterioration before escalation
- **Impact assessment**: Via What-If scenarios, simulate and quantify the impact of each intervention
- **ONN/UNICEF advocacy**: Grade A scores provide objective arguments for targeted program funding

---

## 7. Conclusion

The PEC framework provides a complete and rigorous approach for responsible AI in public health:

1. **PREDICT** delivers reliable predictions (R² > 0.91) through systematic comparison of 6 models
2. **EXPLAIN** ensures transparency through triangulation of 3 methods (SHAP, LIME, Permutation Importance) with Kendall τ > 0.8 agreement
3. **CERTIFY** certifies trust via a unique composite score, validated across 4 domains (Madagascar, Climate, Fintech, Health)
4. **What-If** enables decision-makers to simulate the impact of prospective policies by region and horizon

All **3 Madagascar nutritional targets are certified Grade A**, confirming the system is ready for operational deployment by ONN and UNICEF.