import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate

# --- 1. Data Parsing Functions ---

def parse_conll_file(file_path):
    """
    Parses Dataset1.txt (CoNLL format).
    Returns a list of sentences, where each sentence is a dictionary:
    {'tokens': ['word1', ...], 'ner_tags': ['tag1', ...]}
    """
    data = []
    
    # Read the file using pandas for handling tab separation cleanly
    try:
        df = pd.read_csv(file_path, sep='\t', quoting=3, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    # Clean column names just in case
    df.columns = df.columns.str.strip()
    
    # Group by Sentence #
    grouped = df.groupby("Sentence #")
    
    for _, group in grouped:
        tokens = group["Word"].astype(str).tolist()
        tags = group["Tag"].astype(str).tolist()
        data.append({"tokens": tokens, "ner_tags": tags})
        
    return data

def parse_json_file(file_path):
    """
    Parses Dataset2.json.
    Returns a list of sentences in the same format as above.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            
        annotations = content.get("annotations", [])
        
        for item in annotations:
            tokens = []
            tags = []
            if "entities" in item:
                for entity in item["entities"]:
                    tokens.append(entity["word"])
                    tags.append(entity["label"])
            
            if tokens:
                data.append({"tokens": tokens, "ner_tags": tags})
                
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    return data

# --- 2. Tokenization & Alignment ---

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True
    )
    
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # Map string tags to IDs
        label_ids = [label2id[label] for label in labels]
        
        # Align
        previous_word_idx = None
        label_ids_aligned = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids_aligned.append(-100)
            elif word_idx != previous_word_idx:
                label_ids_aligned.append(label_ids[word_idx])
            else:
                label_ids_aligned.append(-100)
            previous_word_idx = word_idx
            
        new_labels.append(label_ids_aligned)

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

# --- 3. Metrics ---

def compute_metrics(p, label_list, metric):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# --- 4. Main Execution Flow ---

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for NER")
    parser.add_argument("--model_save_path", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the folder containing Dataset1.txt and Dataset2.json")
    parser.add_argument("--num_train_epoch", type=int, default=3, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # 1. Load Data
    print("Loading datasets...")
    dataset1_path = os.path.join(args.dataset_path, "Dataset1.txt")
    dataset2_path = os.path.join(args.dataset_path, "Dataset2.json")
    
    data1 = parse_conll_file(dataset1_path)
    data2 = parse_json_file(dataset2_path)
    
    if not data1 and not data2:
        raise ValueError("No data loaded! Check file paths.")

    print(f"Loaded {len(data1)} sentences from Dataset1")
    print(f"Loaded {len(data2)} sentences from Dataset2")

    # 2. Split Dataset1 (85% Train, 15% Test)
    train_data1, test_data1 = train_test_split(data1, train_size=0.85, random_state=42)
    
    # Save the test split for pipeline.py usage
    test_file_path = os.path.join(args.dataset_path, "test_data_split.json")
    with open(test_file_path, 'w') as f:
        json.dump(test_data1, f)
    print(f"Saved 15% test split ({len(test_data1)} samples) to {test_file_path}")

    # 3. Combine Training Data
    train_data_raw = train_data1 + data2
    print(f"Total training samples: {len(train_data_raw)}")

    # 4. Create Label Mappings
    all_tags = set()
    for item in train_data_raw:
        all_tags.update(item["ner_tags"])
    
    unique_tags = sorted(list(all_tags))
    label2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2label = {i: tag for i, tag in enumerate(unique_tags)}
    
    print(f"Labels found: {unique_tags}")

    # 5. Initialize Hugging Face Dataset
    hf_dataset = Dataset.from_list(train_data_raw)
    
    # 6. Initialize Tokenizer & Model
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, 
        num_labels=len(unique_tags),
        id2label=id2label,
        label2id=label2id
    )

    # 7. Tokenize Dataset
    tokenized_dataset = hf_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True
    )
    
    # Split a small validation set from the training data for evaluation during training
    dataset_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # 8. Setup Trainer
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    metric = evaluate.load("seqeval")
    
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # Changed to eval_strategy to match newer transformers versions
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=args.num_train_epoch,
        weight_decay=0.01,
        save_strategy="no", 
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, unique_tags, metric),
    )

    # 9. Train
    print("Starting training...")
    trainer.train()

    # 10. Save Model
    print(f"Saving model to {args.model_save_path}...")
    trainer.save_model(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)
    
    # Save label map for pipeline usage
    # Note: Redundant 'import json' removed here
    with open(os.path.join(args.model_save_path, "config.json"), "r") as f:
        config = json.load(f)
    config["id2label"] = id2label
    config["label2id"] = label2id
    with open(os.path.join(args.model_save_path, "config.json"), "w") as f:
        json.dump(config, f)

    print("Training complete and model saved.")

if __name__ == "__main__":
    main()