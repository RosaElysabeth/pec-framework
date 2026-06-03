#!/bin/bash
# ============================================================
#  PEC Framework — Initialisation Git + push GitHub
# ============================================================

echo "============================================"
echo "  PEC Framework — Déploiement Streamlit Cloud"
echo "============================================"
echo ""

cd "$(dirname "$0")"

# 1. Initialiser Git
echo "[1/5] Initialisation Git..."
git init
git branch -M main

# 2. Ajouter tous les fichiers du projet
echo "[2/5] Ajout des fichiers..."
git add .

# 3. Vérifier ce qui sera poussé
echo ""
echo "[3/5] Fichiers à pousser :"
git status --short
echo ""

# 4. Premier commit
echo "[4/5] Commit initial..."
git commit -m "PEC Framework v4 — 4 plateformes prêtes pour déploiement"

echo ""
echo "============================================"
echo "  ✅ Repository initialisé !"
echo "============================================"
echo ""
echo "  PROCHAINE ÉTAPE :"
echo "  1. Créer un repo sur https://github.com/new"
echo "     Nom : pec-framework"
echo "     Public ✓  (nécessaire pour Streamlit Cloud gratuit)"
echo "     NE PAS ajouter de README"
echo ""
echo "  2. Copier les 2 commandes Git proposées par GitHub :"
echo "     git remote add origin https://github.com/VOTRE-USER/pec-framework.git"
echo "     git push -u origin main"
echo ""
echo "============================================"