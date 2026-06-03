# Framework PEC — Preuves de Généralisation

## Auteur : Rosa Elysabeth Ralinirina | ORCID : 0009-0003-3048-1765

Le framework PEC (Predict-Explain-Certify) a été conçu comme une méthodologie **générique** applicable à tout domaine de modélisation prédictive. Cette page démontre sa généralisation sur **3 domaines radicalement différents** du cas Madagascar (malnutrition).

---

## 1. Fintech — Afrique de l'Ouest (15 pays, 2016-2024)

| Cible | Meilleur Modèle | R² Test | Score Cert. | Grade |
|-------|----------------|---------|-------------|-------|
| Taux de Défaut (%) | CatBoost | 0.8518 | 87.4/100 | B |
| Ratio NPL (%) | CatBoost | 0.8708 | 88.9/100 | B |
| Pénétration Fintech (%) | Ridge | 0.9873 | 98.9/100 | A |

**Top Features SHAP** : Moyennes mobiles (ma3), PIB/habitant, Indice de réglementation, Mobile money

---

## 2. Santé — Afrique de l'Est (14 provinces, 2013-2023)

| Cible | Meilleur Modèle | R² Test | Score Cert. | Grade |
|-------|----------------|---------|-------------|-------|
| Mortalité Infantile (‰) | Ridge | 0.9744 | 97.7/100 | A |
| Vaccination Complète (%) | Ridge | 0.9876 | 98.9/100 | A |
| Accouchements Qualifiés (%) | Ridge | 0.9909 | 99.2/100 | A |

**Top Features SHAP** : Moyennes mobiles, Couverture vaccinale, PIB/habitant, Eau potable

---

## 3. Climat — Asie du Sud-Est (12 zones, 2014-2024)

| Cible | Meilleur Modèle | R² Test | Score Cert. | Grade |
|-------|----------------|---------|-------------|-------|
| Indice Risque Inondation | Random Forest | 0.8893 | 89.6/100 | B |
| Dégâts (M$) | Ridge | 0.9622 | 96.5/100 | A |
| Population à Risque (M) | Ridge | 0.9907 | 99.2/100 | A |

**Top Features SHAP** : Précipitations max 24h, Subsidence, Qualité construction, Alerte précoce

---

## Synthèse Comparative

| Domaine | Cibles | R² moyen | Cert. moyen | Grades |
|---------|--------|----------|-------------|--------|
| **Madagascar (Nutrition)** | 3 | 0.945 | 96.1/100 | A, A, A |
| **Fintech (Afrique Ouest)** | 3 | 0.903 | 91.7/100 | B, B, A |
| **Santé (Afrique Est)** | 3 | 0.984 | 98.6/100 | A, A, A |
| **Climat (Asie SE)** | 3 | 0.947 | 95.1/100 | B, A, A |

### Conclusion
Le framework PEC **se généralise** à travers des domaines très différents :
- **Nutrition** → **Crédit/Fintech** → **Santé publique** → **Risque climatique**
- Le même pipeline Predict → Explain → Certify s'applique partout
- Les scores de certification sont interprétables et comparables entre domaines
- Les explications SHAP identifient les variables les plus influentes dans chaque contexte