@echo off
echo ======================================================
echo NER MODEL TRAINING AND TESTING PIPELINE
echo ======================================================
echo.

REM Check if required files exist
if not exist "train_with_validation.py" (
    echo ERROR: train_with_validation.py not found!
    pause
    exit /b 1
)

if not exist "pipeline.py" (
    echo ERROR: pipeline.py not found!
    pause
    exit /b 1
)

if not exist "data\train.json" (
    echo ERROR: Training data not found ^(data\train.json^)!
    echo Please run prepare_datasets_clean.py first
    pause
    exit /b 1
)

REM Clean previous results
echo Cleaning previous results...
if exist "ner_model" rmdir /s /q ner_model
if exist "results" rmdir /s /q results
echo Cleaned!
echo.

REM Training phase
echo ======================================================
echo PHASE 1: MODEL TRAINING
echo ======================================================
echo.
echo Settings:
echo   - Model: distilbert-base-uncased
echo   - Epochs: 10
echo   - Learning rate: 5e-5
echo   - Validation split: 10%%
echo   - Class weights: ENABLED
echo.
echo Estimated time: 30-60 minutes ^(CPU^)
echo.
pause

python train_with_validation.py --model_save_path "./ner_model" --dataset_path "./data/train.json" --test_dataset_path "./data/test.json" --num_train_epochs 3 --learning_rate 2e-5 --batch_size 8 --validation_split 0.1 --use_class_weights

if %errorlevel% neq 0 (
    echo.
    echo ======================================================
    echo ERROR: Training failed!
    echo ======================================================
    pause
    exit /b 1
)

echo.
echo ======================================================
echo PHASE 2: MODEL TESTING
echo ======================================================
echo.
echo Running pipeline on test data...
echo.

REM Create results directory
if not exist "results" mkdir results

python pipeline.py --model_load_path "./ner_model" --input_file "./data/test.json" --output_file "./results/predictions.json"

if %errorlevel% neq 0 (
    echo.
    echo ======================================================
    echo ERROR: Testing failed!
    echo ======================================================
    pause
    exit /b 1
)

echo.
echo ======================================================
echo ALL PHASES COMPLETED SUCCESSFULLY!
echo ======================================================
echo.
echo Results:
echo    - Trained model: .\ner_model\
echo    - Training metrics: .\ner_model\training_metrics.json
echo    - Test predictions: .\results\predictions.json
echo.
echo To view training metrics:
echo    type ner_model\training_metrics.json
echo.
echo To view predictions:
echo    type results\predictions.json
echo.
pause