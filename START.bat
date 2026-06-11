@echo off
title BuySignal - Customer Intent Detection
color 0A
cd /d "%~dp0"

echo.
echo  ================================================
echo    BuySignal - Customer Buying Intent Detector
echo  ================================================
echo.
echo  [1/2] Installing required libraries...
pip install flask joblib nltk scikit-learn pandas numpy datasets matplotlib --quiet
echo        Done.
echo.
echo  [2/2] Starting web server...
echo.
echo  ================================================
echo    Open your browser and go to:
echo    http://localhost:5000
echo.
echo    If model is not trained yet, the website
echo    will show a TRAIN button. Click it once.
echo  ================================================
echo.
python app.py
pause
