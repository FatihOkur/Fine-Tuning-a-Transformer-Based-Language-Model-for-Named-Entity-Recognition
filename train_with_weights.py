import argparse
import json
import os
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate
import torch
from torch import nn

# CPU için thread sayısını optimize et
torch.set_num_threads(8)

def load_data(dataset_path):
    """Load training data from JSON file"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_label_list(data):
    """Extract unique labels from dataset"""
    labels = set()
    for item in data:
        labels.update(item['ner_tags'])
    return sorted(list(labels))

def compute_class_weights(data, label_to_id):
    """Compute class weights for imbalanced dataset"""
    from sklearn.utils.class_weight import compute_class_weight
    
    # Collect all labels
    all_labels = []
    for item in data:
        for tag in item['ner_tags']:
            all_labels.append(label_to_id[tag])
    
    # Compute weights
    unique_labels = np.unique(all_labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=all_labels
    )
    
    # Create weight tensor (fill with 1.0 for missing classes)
    weight_dict = {int(label): float(weight) for label, weight in zip(unique_labels, class_weights)}
    weights = torch.ones(len(label_to_id))
    for label_id, weight in weight_dict.items():
        weights[label_id] = weight
    
    return weights

def tokenize_and_align_labels(examples, tokenizer, label_to_id, label_all_tokens=True):
    """Tokenize and align labels with tokens"""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False,
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(label_to_id[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p, label_list):
    """Compute metrics for evaluation"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Calculate accuracy (ignoring -100 labels)
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum()

    return {
        "accuracy": float(accuracy)
    }

# Custom Trainer with class weights and custom logging
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, num_epochs=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.epoch_train_losses = []
        self.epoch_train_accs = []
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute weighted loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs):
        """Override log to capture and format epoch metrics"""
        # Check if this is an epoch-level log
        if "loss" in logs and "epoch" in logs:
            epoch_num = int(logs["epoch"])
            
            # Only process each epoch once
            if epoch_num > self.current_epoch:
                self.current_epoch = epoch_num
                
                # Get train metrics
                train_loss = logs.get("loss", 0)
                
                # Evaluate on validation set if available
                if self.eval_dataset is not None:
                    # Compute train accuracy
                    train_results = self.evaluate(self.train_dataset, metric_key_prefix="train")
                    train_acc = train_results.get("train_accuracy", 0)
                    
                    # Compute validation metrics
                    eval_results = self.evaluate(self.eval_dataset, metric_key_prefix="eval")
                    val_loss = eval_results.get("eval_loss", 0)
                    val_acc = eval_results.get("eval_accuracy", 0)
                    
                    # Store for final summary
                    self.epoch_train_losses.append(train_loss)
                    self.epoch_train_accs.append(train_acc)
                    
                    # Print formatted output
                    print(f"\nEpoch {epoch_num}/{self.num_epochs}")
                    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
                    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
                else:
                    # No validation set
                    print(f"\nEpoch {epoch_num}/{self.num_epochs}")
                    print(f"Train Loss: {train_loss:.4f}")
        
        # Call parent log
        super().log(logs)

def main():
    parser = argparse.ArgumentParser(description="Train NER model with class weights")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val_dataset_path", type=str, default=None, help="Path to validation dataset")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--model_checkpoint", type=str, default="distilbert-base-uncased", help="Model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalanced data")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NER MODEL TRAINING - WITH CLASS WEIGHTS")
    print("=" * 70)
    
    # Load data
    print(f"\n📥 Loading training data from: {args.dataset_path}")
    train_data = load_data(args.dataset_path)
    print(f"✅ Loaded {len(train_data)} training examples")
    
    # Load validation data if provided
    val_data = None
    if args.val_dataset_path:
        print(f"📥 Loading validation data from: {args.val_dataset_path}")
        val_data = load_data(args.val_dataset_path)
        print(f"✅ Loaded {len(val_data)} validation examples")
    else:
        # Split training data for validation (10%)
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
        print(f"✅ Split data: {len(train_data)} train, {len(val_data)} validation")
    
    # Get label list
    label_list = get_label_list(train_data + val_data)
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    
    print(f"\n🏷️  Label list ({len(label_list)} labels):")
    print(label_list)
    
    # Compute class weights if enabled
    class_weights = None
    if args.use_class_weights:
        print(f"\n⚖️  Computing class weights...")
        class_weights = compute_class_weights(train_data, label_to_id)
    
    # Convert to Dataset
    train_dataset = Dataset.from_dict({
        "tokens": [item["tokens"] for item in train_data],
        "ner_tags": [item["ner_tags"] for item in train_data]
    })
    
    val_dataset = Dataset.from_dict({
        "tokens": [item["tokens"] for item in val_data],
        "ner_tags": [item["ner_tags"] for item in val_data]
    })
    
    # Load tokenizer and model
    print(f"\n📂 Loading tokenizer and model: {args.model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_checkpoint,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id
    )
    
    # Tokenize datasets
    print("\n🔧 Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    
    tokenized_val = val_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation"
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=1,
        fp16=False,
        dataloader_num_workers=0,
        no_cuda=True,
        report_to="none",
        load_best_model_at_end=False
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Create trainer
    if args.use_class_weights and class_weights is not None:
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=lambda p: compute_metrics(p, label_list),
            class_weights=class_weights.to(model.device),
            num_epochs=args.num_train_epochs
        )
    else:
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=lambda p: compute_metrics(p, label_list),
            num_epochs=args.num_train_epochs
        )
    
    # Train
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING...")
    print("=" * 70 + "\n")
    
    trainer.train()
    
    # Final evaluation on both train and validation
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION")
    print("=" * 70)
    
    # Evaluate on train set
    train_results = trainer.evaluate(tokenized_train, metric_key_prefix="train")
    train_accuracy = train_results.get("train_accuracy", 0)
    
    # Evaluate on validation set
    val_results = trainer.evaluate(tokenized_val, metric_key_prefix="eval")
    val_accuracy = val_results.get("eval_accuracy", 0)
    
    print(f"\n✅ Final Train Accuracy: {train_accuracy:.4f}")
    print(f"✅ Final Validation Accuracy: {val_accuracy:.4f}")
    
    # Save model
    print(f"\n💾 Saving model to: {args.model_save_path}")
    trainer.save_model(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)
    
    # Save label mapping
    label_map_path = os.path.join(args.model_save_path, "label_map.json")
    with open(label_map_path, 'w') as f:
        json.dump({"label_list": label_list, "label_to_id": label_to_id}, f, indent=2)
    
    # Save final metrics to JSON
    metrics_path = os.path.join(args.model_save_path, "training_metrics.json")
    metrics_data = {
        "final_train_accuracy": float(train_accuracy),
        "final_val_accuracy": float(val_accuracy),
        "num_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "use_class_weights": args.use_class_weights
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"💾 Saved training metrics to: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Train Accuracy: {train_accuracy:.4f}")
    print(f"   Val Accuracy: {val_accuracy:.4f}")
    print(f"   Model: {args.model_save_path}")
    print(f"   Metrics: {metrics_path}")

if __name__ == "__main__":
    main()