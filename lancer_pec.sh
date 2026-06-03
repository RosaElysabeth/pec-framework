#!/bin/bash
# ============================================================
# PEC Framework - Lancement des 3 plateformes Streamlit
# ============================================================

echo "============================================"
echo "   PEC Framework - Predict-Explain-Certify"
echo "   Lancement des 3 plateformes Streamlit"
echo "============================================"
echo ""
echo " Port 8501 : Audit complet (app.py)"
echo " Port 8502 : Aide à la décision (app_decideur.py)"
echo " Port 8503 : Prédiction et What-If (app_inference.py)"
echo ""
echo " Appuyez sur Ctrl+C pour arrêter toutes les plateformes"
echo "============================================"
echo ""

cd "$(dirname "$0")"

# Détection de Python
PYTHON=""
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERREUR : Python non trouvé. Installez Python 3.10+ et relancez."
    exit 1
fi

echo "Python trouvé : $PYTHON"
echo ""

# Vérification que streamlit est installé
if ! $PYTHON -m streamlit version &> /dev/null; then
    echo "ERREUR : streamlit n'est pas installé. Installation en cours..."
    $PYTHON -m pip install streamlit
fi

# Lancement des 3 plateformes
$PYTHON -m streamlit run app.py --server.port 8501 --server.headless true &
PID1=$!
sleep 3

$PYTHON -m streamlit run app_decideur.py --server.port 8502 --server.headless true &
PID2=$!
sleep 3

$PYTHON -m streamlit run app_inference.py --server.port 8503 --server.headless true &
PID3=$!
sleep 3

$PYTHON -m streamlit run app_generalisation.py --server.port 8504 --server.headless true &
PID4=$!

echo ""
echo "============================================"
echo "  Les 4 plateformes sont lancées !"
echo "============================================"
echo ""
echo "  http://localhost:8501  -  Audit complet"
echo "  http://localhost:8502  -  Aide à la décision"
echo "  http://localhost:8503  -  Prédiction et What-If"
echo "  http://localhost:8504  -  Généralisation multi-domaines"
echo ""

# Ouverture du navigateur
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8501
elif command -v open &> /dev/null; then
    open http://localhost:8501
fi

echo "PIDs : $PID1 (8501) | $PID2 (8502) | $PID3 (8503) | $PID4 (8504)"
echo "Pour arrêter : kill $PID1 $PID2 $PID3 $PID4"

# Attendre Ctrl+C
trap "echo 'Arrêt en cours...'; kill $PID1 $PID2 $PID3 $PID4 2>/dev/null; exit 0" INT TERM
wait