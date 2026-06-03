# Résultats de Recherche — Framework PEC
# Rosa Elysabeth Ralinirina | ORCID : 0009-0003-3048-1765

## Structure des dossiers

```
resultats_recherche/
├── fr/                              # Français
│   ├── interpretation_discussion.md  (7 sections : PEC, échelle certification, résultats, scénarios What-If, généralisation, perspectives, conclusion)
│   ├── resultats_pec_madagascar.json (résultats + scénarios + features + sources)
│   ├── rapport_final.json
│   ├── comparaison_modeles.csv
│   ├── generalisation_climat/
│   ├── generalisation_fintech/
│   └── generalisation_sante/
├── en/                              # English
│   ├── interpretation_discussion.md  (7 sections: PEC, certification scale, results, What-If scenarios, generalization, perspectives, conclusion)
│   ├── pec_results_madagascar.json   (results + scenarios + features + sources)
│   ├── rapport_final.json
│   ├── comparaison_modeles.csv
│   ├── generalisation_climat/
│   ├── generalisation_fintech/
│   └── generalisation_sante/
└── README.md

PROJET_PEC/
├── CITATIONS_PEC.md                 (8 citations originales FR/EN + BibTeX)
├── DEMONSTRATION_SOUTENANCE.md      (7 phases ~25 min, scénarios test, Q&A jury)
├── shared_pec.css                   (CSS uniforme pour les 3 plateformes)
├── app.py                           (8501 Dashboard Madagascar)
├── app_inference.py                 (8502 What-If Madagascar)
└── app_decideur.py                  (8503 PEC Généralisé)
```

## Mises à jour récentes

- **Scénarios What-If** : noms de colonnes corrigés (Pauvrete_region_2021_pct, Couverture_sanitaire_CSB2_10000hab), features lag/ma3 mis à jour itérativement, statu quo projeté comme baseline, axe Y resserré
- **Sources des 139 variables** : 11 catégories avec sources affichées dans l'interface
- **Footer uniforme** : PEC Framework v4 | Predict • Explain • Certify | Rosa Elysabeth Ralinirina (avec icônes Font Awesome)
- **Certification grades** : A≥90, B≥80, C≥60, D<60 — uniforme sur les 3 plateformes
