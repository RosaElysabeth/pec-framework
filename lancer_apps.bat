@echo off
echo ============================================
echo  PEC Framework - Lancement des 4 plateformes
echo  Predict - Explain - Certify - Generalise
echo ============================================
echo.
cd /d "C:\DATA ROSA\RECHERCHE\08-PROJET-PEC"

echo [1/4] PEC Madagascar (8501)...
start "" python -m streamlit run app.py --server.port 8501 --server.headless true
timeout /t 3 /nobreak >nul

echo [2/4] Decideur (8502)...
start "" python -m streamlit run app_decideur.py --server.port 8502 --server.headless true
timeout /t 3 /nobreak >nul

echo [3/4] What-If (8503)...
start "" python -m streamlit run app_inference.py --server.port 8503 --server.headless true
timeout /t 3 /nobreak >nul

echo [4/4] Generalisation (8504)...
start "" python -m streamlit run app_generalisation.py --server.port 8504 --server.headless true
timeout /t 3 /nobreak >nul

echo.
echo ============================================
echo  4 plateformes lancees !
echo ============================================
echo   8501 : PEC Madagascar
echo   8502 : Decideur
echo   8503 : What-If
echo   8504 : Generalisation
echo.

start http://localhost:8501

echo Pour arreter : fermez les fenetres ou Ctrl+C
pause