# Top 10 Influential Researchers in XAI and Predictive Modeling
## Synthesis for the PEC Thesis — Rosa Elysabeth Ralinirina

---

> This document identifies the 10 most influential researchers in the field of Explainable Artificial Intelligence (XAI) applied to predictive modeling. Each profile presents their career, major contributions, connection with the PEC framework, and their potential as jury or reference for the thesis.

---

## 1. Cynthia RUDIN — Duke University, USA

### Career
- Professor of Computer Science and Statistics, Duke University
- Previously: Columbia University Center for Computational Learning
- Awards: ACM AAAI Fall Symposium, multiple best paper awards
- Recognized as one of the most influential voices against black-box models

### Major Contributions

| Publication | Year | Key Contribution |
|---|---|---|
| "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead" | 2019 | **Foundational paper**: argues that post-hoc explanations are insufficient for high-stakes decisions. Demonstrates that interpretable models (EBM, rule lists) often achieve comparable performance to black boxes. |
| "Please Stop Permuting Features: An Explanation and Alternative" (with M. Hooker) | 2019 | **Critique of Permutation Importance**: demonstrates that permuting features corrupts distributions and produces erroneous conclusions. |
| Optimal Sparse Decision Trees | 2020 | **Optimal decision trees**: exact approaches to build interpretable trees without greedy heuristics. |
| Intelligible Models for HealthCare (with Carrie Cai) | 2015 | **Intelligible models in healthcare**: demonstration that rule-based models can rival neural networks in healthcare. |

### Connection with PEC

**Direct and central.** Rudin is the intellectual founder of the argument that PEC takes further:

- **Rudin**: "Interpretable models are sufficient → stop explaining black boxes"
- **PEC**: "Even interpretable models must be certified → Predict-Explain-Certify"

PEC extends Rudin's argument: even when using an interpretable model, confidence in explanations must be quantified (τ_R, τ_F, τ_T). Certification answers the question Rudin implicitly poses: "How to ensure that interpretability is real and not apparent?"

**In IEEE article**: Cited as reference for interpretable model advocacy (RQ3, Section VI). Her "Stop explaining" argument is the theoretical starting point of the Certify layer.

### Impact on our results

| PEC Aspect | Rudin's Influence | Our Result |
|---|---|---|
| Models compared | 6 models including 5 "black box" + Ridge | Ridge (interpretable-linear) beats XGBoost, LightGBM, CatBoost on IP (R²=0.91), MC (R²=0.96), MA (R²=0.97) |
| Certification | Need to certify even interpretable models | IP=94.5(A), MC=95.8(A), MA=98.0(A) — Ridge certified Grade A |
| What-If Scenarios | Interpretable models enable true causal explanation | 8 prospective scenarios for 2026-2035 with real variables |

---

## 2. Scott M. LUNDBERG — Microsoft Research, USA

### Career
- Principal Researcher, Microsoft Research, Redmond
- PhD, University of Washington (advised by Su-In Lee)
- Creator of the SHAP library (shap Python package, 20k+ GitHub stars)
- Co-creator of TreeSHAP and KernelSHAP

### Major Contributions

| Publication | Year | Key Contribution |
|---|---|---|
| "A Unified Approach to Interpreting Model Predictions" (SHAP) | 2017 | **SHAP**: unifies LIME, DeepLIFT, and Shapley values under a single mathematical framework. Shapley values become the explainability standard. |
| "From Local Explanations to Global Understanding" (TreeSHAP) | 2020 | **TreeSHAP**: exact O(TLD²) computation of Shapley values for tree-based models. Makes SHAP scalable. |
| "Explainable AI for Trees" | 2020 | **Global extension**: SHAP summary plots, dependence plots, interaction values. SHAP goes from local to global. |
| "An Unexpected Failure of Feature Importance" | 2024 | **Self-critique**: identifies cases where SHAP and Permutation Importance fail on categorical features. |

### Connection with PEC

**Critical and constructive.** SHAP is the most used XAI method in PEC (apps 8501, 8502, 8503). But Lundberg himself acknowledges the limitations:

- **SHAP**: quantifies contribution, not causation
- **PEC Certify**: measures whether SHAP contributions are **robust** (τ_R), **faithful** (τ_F), **trustworthy** (τ_T)

Our work does not attack SHAP — it **complements** it. SHAP generates explanations; PEC certifies their quality.

### Impact on our results

| PEC Aspect | Lundberg's Contribution | Our Result |
|---|---|---|
| SHAP summary plots | Top features per target | IP: ma3(3.81), Ensoleillement(0.89); MC: ma3(6.14), lag1(1.38); MA: ma3(2.15), lag1(0.69) |
| SHAP dependence | Feature-target relationship | Visualized in EXPLAIN tab (8501) |
| TreeSHAP for RF/XGB | Exact and fast computation | Used for Random Forest and Gradient Boosting |
| Acknowledged limits | Contribution ≠ Causation | Our What-If scenarios test causal impact of variables |

---

## 3. Andreas HOLZINGER — Graz University, Austria

### Career
- Professor of Computer Science, Graz University of Technology
- Director of the Holzinger Group, AI & ML for Medicine
- Founder of the xAI conference (annual series)
- 500+ publications, h-index > 60
- 9 papers in our Zotero corpus

### Major Contributions

| Publication | Year | Key Contribution |
|---|---|---|
| "What Do We Need to Build Explainable AI Systems for the Medical Domain?" | 2017 | **Medical XAI manifesto**: identifies 3 levels of transparency (simulatability, decomposability, algorithmic transparency). |
| "Causability and Explainability in Medicine" | 2019 | **Causability**: introduces the concept of causability (explainability + causality). A system is "causable" if the human user can understand the causes. |
| "The System Causability Scale (SCS)" | 2020 | **SCS**: measurement scale for explanation quality. Only existing quantitative framework before PEC for measuring explainability. |
| "Measuring the Quality of Explanations" (with Carrington, Muller) | 2020 | **Formal SCS**: operational definition of SCS with 10 measurable criteria. |

### Connection with PEC

**Closest theoretical relative.** Holzinger is the researcher whose work is most directly related to PEC:

| Holzinger Concept | PEC Extension |
|---|---|
| Causability (causal quality) | Certificate τ_F (explanation fidelity to model) |
| SCS (System Causability Scale) | PEC Composite certification (τ_R + τ_F + τ_T) |
| iML (human in the loop) | τ_T (human-validated trustworthiness) |
| 3 levels of transparency | 3 PEC layers (Predict, Explain, Certify) |

**Key difference**: SCS measures the **potential quality** of the explanation system (structural). PEC measures the **empirical quality** of produced explanations (real data + robustness + fidelity + human decisions).

### Impact on our results

| PEC Aspect | Holzinger's Contribution | Our Result |
|---|---|---|
| SCS → Certification | Qualitative scale → quantitative scores | Grade A/B/C/D (≥90/≥80/≥60/<60) with composite scores |
| Causability → Fidelity | Understanding causes → measuring if explanation = model | τ_F measures fidelity numerically |
| iML → Trustworthiness | Human in the loop → human validates explanation | τ_T > 1 = explanations improve decisions |
| Multi-level transparency | 3 levels → 3 layers | Predict, Explain, Certify |

---

## 4. Marzyeh GHASSEMI — MIT / University of Toronto, Canada

### Career
- Associate Professor, MIT (Computer Science + Institute for Medical Engineering)
- Previously: University of Toronto, Vector Institute
- PhD MIT (Machine Learning Group)
- Awards: MIT Technology Review Innovator Under 35

### Major Contributions

| Publication | Year | Key Contribution |
|---|---|---|
| "The False Hope of Current Approaches to Explainable Artificial Intelligence in Health Care" | 2021 | **Central critique paper**: argues that current XAI approaches in healthcare are insufficient. Post-hoc explanations do not guarantee trust. |
| "A Review of Challenges and Opportunities in Machine Learning for Health" (with Suresh) | 2022 | **ML-health challenges**: identifies biases, data leaks, generalization problems. |
| "Machine Learning for Sepsis Prediction" | 2022 | **Empirical study**: demonstrates that sepsis models do not generalize across hospitals. |
| "Decoding the Black Box: A Review of Interpretability for Deep Learning in Medicine" | 2020 | **XAI-health review**: classification of methods by type and clinical application domain. |

### Connection with PEC

**Foundational critique of the Certify layer.** Ghassemi demonstrates that:

1. SHAP/LIME explanations in healthcare are often **decorative** — they do not change clinical decisions
2. Medical ML models **do not generalize** across contexts
3. The "false hope" is believing that explainability alone suffices

**PEC responds directly**:
- Ghassemi: "Explainability is insufficient" → PEC: "Hence the Certify layer"
- Ghassemi: "Models don't generalize" → PEC: "Certify τ_R (robustness) to quantify contextual stability"
- Ghassemi: "Explanations don't change decisions" → PEC: "τ_T measures whether explanations improve human decisions"

### Impact on our results

| Ghassemi's Critique | PEC Response (Madagascar) |
|---|---|
| "Post-hoc explanations insufficient" | Composite certification quantified τ = 0.4ρ + 0.3τ_R + 0.3τ_T |
| "Models don't generalize" | Generalization tested on 3 external domains (climate: 89.6-99.2, fintech: 87.4-98.9, health: 97.7-99.2) |
| "Data biases" | Exhaustive Data Dictionary (139 variables, 11 categories, 9 sources) |
| "Explanations = decorative" | What-If scenarios 2026-2035 for real ONN/UNICEF decision-maker use |

---

## 5. Q. Vera LIAO — IBM Research, Canada

### Career
- Principal Researcher, IBM Research AI
- PhD, University of British Columbia (Human-Computer Interaction)
- Chair of ACM SIGCHI Interest Group on XAI
- Associate Editor, ACM Computing Surveys

### Major Contributions

| Publication | Year | Key Contribution |
|---|---|---|
| "Human-Centered Explainable AI (XAI): From Algorithms to User Experiences" (with Varshney) | 2022 | **Human-Centered XAI manifesto**: explainability evaluations must be user-centered, not algorithm-centered. Proposes a 3-level framework: functional, human, application. |
| "Questioning the AI: Informing Design Practices for Explainable AI User Experiences" | 2020 | **XAI-UX design**: question-oriented approach. Explanations must answer user questions. |
| "Explaining Machine Learning to Users: A Design Space for User-Oriented XAI" | 2021 | **Design space**: taxonomy of factors affecting user comprehension of explanations. |
| "Social Transparency in AI Systems" (with Ehsan et al.) | 2021 | **Social transparency**: beyond algorithmic transparency, make visible the social processes around AI. |

### Connection with PEC

**Extension toward τ_T (trustworthiness).** Liao provides the theoretical framework for the third component of PEC certification:

| Liao Concept | PEC Extension |
|---|---|
| Human-Centered XAI | τ_T: explanations must improve human decisions |
| Question-oriented design | PEC answers 3 questions: robust? faithful? trustworthy? |
| Social transparency | ONN/UNICEF social context: who uses the explanation, for what? |
| User experiences | Streamlit platforms (8501/8502/8503) decision-maker centered |

---

## 6. Wojciech SAMEK — Fraunhofer Heinrich Hertz Institute, Germany

### Career
- Professor, Technische Universität Berlin
- Team Leader, Fraunhofer HHI, Berlin
- Co-editor of "Explainable AI: Interpreting, Explaining and Visualizing Deep Learning" (Springer, 2019)
- 9 papers in our Zotero corpus

### Major Contributions

| Publication | Year | Key Contribution |
|---|---|---|
| "Layer-Wise Relevance Propagation (LRP)" (with Montavon, Binder, Muller) | 2019 | **LRP**: method for decomposing prediction into layer-wise contributions. Conservation principle: total relevance = prediction. |
| "Explainable AI: Interpreting, Explaining and Visualizing Deep Learning" (edited book) | 2019 | **Reference book**: first comprehensive work on XAI for deep learning. |
| "Evaluating Feature Importance Estimates" (with Hooker et al.) | 2019 | **Remove-and-Retrain**: evaluation protocol for feature importance by removal + retraining. |
| "Quantus: An Explainable AI Toolkit for Responsible Evaluation" (with Hedstrom et al.) | 2023 | **Quantus**: toolkit for evaluating XAI explanations. Standardizes robustness and fidelity metrics. |

### Connection with PEC

**Evaluation infrastructure.** Samek provides the tools and methods that PEC integrates into the Certify layer:

| Samek Contribution | PEC Usage |
|---|---|
| LRP | Complementary XAI method (post-hoc, model-specific for neural networks) |
| Remove-and-Retrain | Feature importance evaluation protocol → influences τ_F |
| Quantus | Toolkit for τ_R and τ_F (robustness and fidelity of explanations) |

---

## 7. Galit SHMUELI — National Tsing Hua University, Taiwan

### Career
- Professor of Statistics, National Tsing Hua University, Taiwan
- Previously: Indian School of Business, University of Maryland
- Author of "Data Mining for Business Analytics" (Wiley, 3 editions)
- Article "To Explain or to Predict?" = most cited in predictive modeling (5000+ citations)

### Major Contributions

| Publication | Year | Key Contribution |
|---|---|---|
| "To Explain or to Predict?" | 2010 | **Foundational dichotomy**: the goals of explanation and prediction are distinct and may conflict. f for causality vs f-hat for error minimization. |
| "Predictive Analytics in Information Systems Research" (with Koppius) | 2011 | **Predictive methodology**: guide for predictive research in information systems. |
| "Data Mining for Business Analytics" (book) | 2016- | **Reference**: standard textbook for applied predictive modeling. |

### Connection with PEC

**Theoretical foundation of the Predict layer.** The Explain-or-Predict dichotomy is the foundation on which PEC builds:

| Shmueli Dichotomy | PEC Extension |
|---|---|
| Explain (f causable) | Explain layer (ξ) |
| Predict (f-hat performant) | Predict layer (f-hat, ρ) |
| **Missing** | **Certify layer (τ)**: certification as a third dimension |

PEC = Explain ∪ Predict ∪ Certify. Without Shmueli, there is no theoretical basis for separating objectives. But Shmueli leaves an open question: how to ensure that the chosen model (for either objective) is **reliable**? This is the question that PEC Certifies.

---

## 8. Finale DOSHI-VELEZ — Harvard University, USA

### Career
- Associate Professor, Harvard School of Engineering and Applied Sciences
- PhD MIT (Computer Science)
- Co-founder of the Harvard Data Science Initiative
- Awards: NSF CAREER, Google Faculty Research

### Major Contributions

| Publication | Year | Key Contribution |
|---|---|---|
| "Towards A Rigorous Science of Interpretable Machine Learning" (with Been Kim) | 2017 | **Evaluation taxonomy**: 3 levels (functionality-grounded, human-grounded, application-grounded). First formal framework for evaluating explanation quality. |
| "Considerations for Evaluation of Generalization of ML-based Prediction Systems" | 2019 | **Generalization of predictive models**: questions the ability of ML models to generalize beyond training data. |
| "A Human-Centered Evaluation of a Submodular Approach to Data Summarization" | 2018 | **Human evaluation**: protocol for measuring whether ML outputs are actually useful to humans. |

### Connection with PEC

**Evaluation framework that PEC operationalizes.** Doshi-Velez provides the taxonomy; PEC provides the metrics:

| Doshi-Velez Level | PEC Metric |
|---|---|
| Functionality-grounded | τ_F (fidelity): automated measure of explanation-model correspondence |
| Human-grounded | τ_T (trustworthiness): test if explanations improve human decisions |
| Application-grounded | Grade A/B/C/D: operational validation in the application domain |

---

## 9. Marco Tulio RIBEIRO — Microsoft Research, USA

### Career
- Senior Researcher, Microsoft Research
- PhD, University of Washington (advised by Carlos Guestrin)
- Creator of LIME (Local Interpretable Model-agnostic Explanations)
- Creator of Anchors (rule-based local explanations)

### Major Contributions

| Publication | Year | Key Contribution |
|---|---|---|
| "Why Should I Trust You? Explaining the Predictions of Any Classifier" (LIME) | 2016 | **LIME**: local model-agnostic method that approximates the model with an interpretable linear model in a neighborhood. Foundational question: "Why should I trust you?" |
| "Anchors: High-Precision Model-Agnostic Explanations" | 2018 | **Anchors**: if-then rules with precision guarantee > τ. More stable than LIME for practical applications. |
| "Robustness and Reliability of LIME" (self-critique) | 2023 | **LIME limits**: sensitivity to neighborhood kernel, instability between runs, fidelity < 1. |

### Connection with PEC

**Case study of post-hoc limitations.** LIME illustrates exactly why the Certify layer is necessary:

| LIME Problem (acknowledged by Ribeiro) | PEC Response |
|---|---|
| Instability (different runs = different explanations) | τ_R measures robustness (stability under perturbation) |
| Fidelity < 1 (surrogate doesn't capture full model) | τ_F measures fidelity (explanation-model correspondence) |
| Sensitivity to kernel π_x | Certification requires documentation of hyperparameters |
| "Why trust?" | PEC answers: with a quantifiable score τ |

---

## 10. Klaus-Robert MÜLLER — Technische Universität Berlin, Germany

### Career
- Professor of Computer Science, TU Berlin
- Director, Korea University (joint)
- Member of the German Academy of Sciences (Leopoldina)
- Co-creator of LRP and kernel analysis
- H-index > 80

### Major Contributions

| Publication | Year | Key Contribution |
|---|---|---|
| "Layer-Wise Relevance Propagation" (with Montavon, Samek) | 2019 | **LRP**: co-creator. Conservation principle for decomposing neural network predictions. |
| "Explainable AI: Interpreting, Explaining and Visualizing Deep Learning" (co-editor) | 2019 | **XAI reference book**: the work that structured the field. |
| "On the Validity of Explaining Neural Network Predictions" (with Montavon) | 2018 | **Explanation validity**: how to verify that LRP explanations truly reflect model functioning. |
| Kernel PCA and kernel analysis | 1998- | **Mathematical foundations**: kernel theory underpins SHAP and kernel models. |

### Connection with PEC

**Validation of explanations.** Müller poses the fundamental question: "Are the explanations valid?" — exactly the question the Certify layer formalizes.

| Müller's Question | PEC Metric |
|---|---|
| "Are LRP explanations valid?" | τ_F: explanation fidelity to model |
| Conservativity vs. selectivity | τ_R: robustness under perturbation |
| Different LRP rules = different explanations | Need to certify each model-explainer pair |

---

## Synthesis: PEC Positioning Relative to the 10 Researchers

### Conceptual Map

```
EXPLAIN ←──────────────────── PREDICT
   │                              │
   │  Shmueli (2010)              │
   │  "To Explain or              │
   │   to Predict?"               │
   │                              │
   │  ┌───────────────────────────┘
   │  │
   ▼  ▼
XAI (post-hoc)  ←→  Interpretable (by design)
   │                       │
   │  Lundberg (SHAP)      │  Rudin (Stop Explaining)
   │  Ribeiro (LIME)       │  EBMs, Rule Lists
   │  Samek (LRP)          │
   │  Müller (LRP/valid.)
   │                      
   │  ┌───────────────────────────────┐
   │  │  PROBLEM (meta-explainability │
   ▼  ▼  gap)                          │
   │  │                               │
   │  │  Ghassemi: "False Hope"        │
   │  │  Holzinger: SCS (measure)      │
   │  │  Liao: Human-Centered          │
   │  │  Doshi-Velez: 3 levels         │
   │  └───────────────────────────────┘
   │                 │
   │                 ▼
   │         ┌──────────────┐
   │         │   CERTIFY    │
   │         │   (PEC)      │
   │         │              │
   │         │  τ_R robust.  │
   │         │  τ_F fidelity │
   │         │  τ_T trust    │
   │         │              │
   │         │  → Grade A/B/C/D
   │         └──────────────┘
```

### Comparative Summary Table

| # | Researcher | Institution | Major Contribution | PEC Connection | Cited in IEEE Article |
|---|---|---|---|---|---|
| 1 | **Cynthia Rudin** | Duke University | Interpretable models, "Stop Explaining" | Foundation: interpretable ≠ certified | Yes (5+ times) |
| 2 | **Scott Lundberg** | Microsoft Research | SHAP, TreeSHAP | Explain layer standard; limits → Certify | Yes (12+ times) |
| 3 | **Andreas Holzinger** | Graz University | Causability, SCS | Closest relative; SCS → PEC certification | Yes (5+ times) |
| 4 | **Marzyeh Ghassemi** | MIT | "False Hope" | Foundational critique of Certify layer | Yes (3+ times) |
| 5 | **Q. Vera Liao** | IBM Research | Human-Centered XAI | Theoretical framework for τ_T | Yes (3 times) |
| 6 | **Wojciech Samek** | Fraunhofer HHI | LRP, Quantus | Evaluation tools for τ_R, τ_F | Yes (4+ times) |
| 7 | **Galit Shmueli** | Nat'l Tsing Hua | Explain-or-Predict | Theoretical foundation of Predict layer | Yes (2+ times) |
| 8 | **Finale Doshi-Velez** | Harvard | 3 evaluation levels | Taxonomy operationalized by PEC | Yes (2 times) |
| 9 | **Marco T. Ribeiro** | Microsoft Research | LIME, Anchors | Illustration of post-hoc limits → Certify | Yes (3+ times) |
| 10 | **Klaus-Robert Müller** | TU Berlin | LRP, XAI validation | Validity question → τ_F, τ_R | Yes (via LRP) |

### Emerging Researchers to Follow (2026-2028)

| Researcher | Institution | Why |
|---|---|---|
| Zana Bucinca | Harvard | Cognitive forcing functions to avoid AI overreliance |
| Upol Ehsan | Georgia Tech | Social transparency: who is behind AI? |
| Anna Hedstrom | TU Berlin | Quantus: standard toolkit for evaluating explanations |
| Christoph Molnar | LMU Munich | "Interpretable ML" book; DALEX; XAI pitfalls |
| Przemyslaw Biecek | Warsaw | DALEX, model-agnostic explanation framework |
| Aleksander Molak | DeepL | Causal Inference and Discovery in Python |
| Judea Pearl | UCLA | Do-calculus, causal inference — the foundation of causal XAI |

---

## Integration with PEC Madagascar Results

### Operational Results (Certification)

| Target | Model | R² | Certification | Grade | Most Affected Region | Regional Gap |
|---|---|---|---|---|---|---|
| Insuffisance Pondérale | Ridge | 0.9105 | 94.5/100 | **A** | Vakinankaratra (36.9%) | 21.9 pts |
| Malnutrition Chronique | Ridge | 0.9588 | 95.8/100 | **A** | Vakinankaratra (58.0%) | 32.4 pts |
| Malnutrition Aigüe | Ridge | 0.9670 | 98.0/100 | **A** | Vatovavy Fitovinany (11.0%) | 6.9 pts |

### Multi-Domain Generalization

| Domain | Target | R² | Certification | Model |
|---|---|---|---|---|
| Climate | Flood Risk Index | 0.8893 | 89.6/100 (B) | Random Forest |
| Climate | Damages millions USD | 0.9622 | 96.5/100 (A) | Ridge |
| Climate | Population at risk (M) | 0.9907 | 99.2/100 (A) | Ridge |
| Fintech | Default Rate (%) | 0.8518 | 87.4/100 (B) | CatBoost |
| Fintech | NPL Ratio (%) | 0.8708 | 88.9/100 (B) | CatBoost |
| Fintech | Fintech Penetration (%) | 0.9873 | 98.9/100 (A) | Ridge |
| Health | Infant Mortality (/1000) | 0.9744 | 97.7/100 (A) | Ridge |
| Health | Complete Vaccination (%) | 0.9876 | 98.9/100 (A) | Ridge |
| Health | Qualified Deliveries (%) | 0.9909 | 99.2/100 (A) | Ridge |

### What-If Scenarios (8 scenarios, 2026-2035)

| Scenario | Modified Variables | Expected Impact |
|---|---|---|
| Status quo | None | Baseline |
| Climate improvement | Precip +10%, Temp max -0.5°C | Malnutrition decrease |
| Climate worsening | Precip -20%, Temp max +1°C | Malnutrition increase |
| Poverty reduction | Poverty -10 pts | Malnutrition decrease |
| Poverty increase | Poverty +10 pts | Malnutrition increase |
| Health improvement | Water +15%, Sanitation +10%, CSB2 +0.3 | Acute malnutrition decrease |
| Combined favorable | Poverty + Precip + Temp + Water + Sanitation | Combined decrease |
| Combined unfavorable | Poverty + Precip + Temp | Combined increase |

---

## How to Use This Document for the Thesis

### 1. State of the Art Chapter (Section II of the thesis)
- Position the 10 researchers in 3 groups: founders (Shmueli, Rudin), XAI methods (Lundberg, Ribeiro, Samek, Müller), evaluation/critique (Ghassemi, Holzinger, Liao, Doshi-Velez)
- Show how PEC builds on and extends each one

### 2. PEC Framework Justification (Section III)
- Shmueli (Explain/Predict) → PEC adds Certify
- Ghassemi (False Hope) → PEC responds with certification
- Holzinger (SCS) → PEC extends with τ_R, τ_F, τ_T
- Doshi-Velez (3 levels) → PEC operationalizes with metrics

### 3. Results Discussion (Section VII)
- Compare our certifications with each researcher's predictions
- Rudin: Ridge (interpretable) beats black boxes → confirmed
- Ghassemi: generalization tested on 3 domains → successful
- Holzinger: SCS vs PEC Grade A → concordant

### 4. Defense (PPT)
- Slide "Scientific Positioning": conceptual map above
- Slide "State of the Art": comparative table
- Slide "Results": integration with each researcher's predictions
- Q&A: "How does PEC compare to SCS? To SHAP? To Shmueli's dichotomy?"

---

*Document generated for the PEC thesis of Rosa Elysabeth Ralinirina — 2025*
*ORCID: 0009-0003-3048-1765*