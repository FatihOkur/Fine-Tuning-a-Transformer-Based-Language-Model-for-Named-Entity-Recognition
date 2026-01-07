@echo off
echo ==================================================
echo NER TRAINING - IMPROVED VERSION
echo Class Weights + 10 Epochs + Higher Learning Rate
echo ==================================================
echo.

REM Check if train_with_weights.py exists
if not exist "train_with_weights.py" (
    echo ERROR: train_with_weights.py bulunamadi!
    echo Lutfen once train_with_weights.py dosyasini indirin.
    pause
    exit /b 1
)

REM Check if data exists
if not exist "data\train.json" (
    echo ERROR: data\train.json bulunamadi!
    echo Lutfen once prepare_datasets.py calistirin.
    pause
    exit /b 1
)

echo Eski model siliniyor...
if exist "ner_model" rmdir /s /q ner_model
if exist "results" rmdir /s /q results
echo Temizlendi!

echo.
echo ==================================================
echo TRAINING BASLIYOR
echo ==================================================
echo.
echo Ayarlar:
echo   - Epochs: 10
echo   - Learning rate: 5e-5
echo   - Class weights: ENABLED
echo   - Batch size: 8
echo.
echo Tahmini sure: 90-150 dakika (CPU)
echo.
echo Kontrol edin:
echo   - Loss her epoch'ta dusuyorsa basarili
echo   - Loss sabit kaliyorsa sorun var
echo.
pause

python train_with_weights.py --model_save_path "./ner_model" --dataset_path "./data/train.json" --num_train_epochs 10 --use_class_weights --learning_rate 5e-5

if %errorlevel% neq 0 (
    echo.
    echo ==================================================
    echo ERROR: Training basarisiz!
    echo ==================================================
    echo.
    echo Hata mesajini kontrol edin.
    echo.
    echo Olasi sorunlar:
    echo 1. scikit-learn kurulu degil
    echo    Cozum: pip install scikit-learn
    echo.
    echo 2. train_with_weights.py yanlis yerde
    echo    Cozum: Dosyayi proje klasorune koyun
    echo.
    pause
    exit /b 1
)

echo.
echo ==================================================
echo TRAINING TAMAMLANDI!
echo ==================================================
echo.
echo Sonraki adimlar:
echo 1. test_model.bat calistirin
echo 2. python view_results.py calistirin
echo.
pause