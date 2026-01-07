To run the code do the followings:
1) install requirements by -> pip install -r "requirements.txt"
2) run -> prepare_datasets_clean.py
3) run -> .\run.bat or .\run.sh
4) run -> view_training_metrics.py  to see the evaluation metrics
OR manually
1)python train_with_validation.py \
    --model_save_path "./ner_model" \
    --dataset_path "./data/train.json" \
    --test_dataset_path "./data/test.json" \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --batch_size 8 \
    --validation_split 0.1 \
    --use_class_weights
2)python pipeline.py \
    --model_load_path "./ner_model" \
    --input_file "./data/test.json" \
    --output_file "./results/predictions.json"
3)python view_training_metrics.py

Resulting predictions of test data: results/predictions.json
Training metrics: ner_model/training_metrics.json