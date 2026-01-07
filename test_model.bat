@echo off
echo ==================================================
echo MODEL TEST - NER Pipeline
echo ==================================================
echo.
echo Bu script eğitilmiş modelinizi test edecek ve
echo doğruluğunu gösterecek.
echo.
pause

REM Create results folder if not exists
if not exist "results" mkdir results

echo.
echo ==================================================
echo Test Başlıyor...
echo ==================================================
echo.
echo ⚙️  Model: ./ner_model
echo 📥 Test data: ./data/test.json
echo 💾 Sonuç: ./results/predictions.json
echo.

python pipeline_FIXED.py --model_load_path "./ner_model" --input_file "./data/test.json" --output_file "./results/predictions.json"

if %errorlevel% neq 0 (
    echo.
    echo ❌ ERROR: Test başarısız!
    pause
    exit /b 1
)

echo.
echo ==================================================
echo ✅ TEST TAMAMLANDI!
echo ==================================================
echo.
echo 📁 Detaylı sonuçlar: results\predictions.json
echo.
echo Bu dosyayı açarak:
echo - Her cümle için tahminleri
echo - Entity tiplerini
echo - Güven skorlarını görebilirsiniz
echo.
pause