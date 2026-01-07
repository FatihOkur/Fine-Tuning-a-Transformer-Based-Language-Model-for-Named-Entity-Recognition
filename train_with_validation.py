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
    DataCollatorForTokenClassification,
    TrainerCallback
)
import evaluate
import torch
from torch import nn
from sklearn.model_selection import train_test_split

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

def compute_metrics(p, label_list, metric):
    """Compute metrics for evaluation"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

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

# Custom Trainer with class weights
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute weighted loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Custom callback for printing progress with proper evaluation
class ProgressCallback(TrainerCallback):
    def __init__(self, num_epochs, trainer=None):
        self.num_epochs = num_epochs
        self.epoch_metrics = []
        self.trainer = trainer
        self.train_dataset = None
        self.val_dataset = None
    
    def set_datasets(self, train_dataset, val_dataset):
        """Set train and validation datasets after initialization"""
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        epoch = int(state.epoch)
        
        print(f"\n📊 Evaluating epoch {epoch}/{self.num_epochs}...", end=" ", flush=True)
        
        # Calculate training metrics on a subset (for speed)
        train_loss = 0
        train_accuracy = 0
        if self.trainer and self.train_dataset:
            try:
                # Use 20% of training data for evaluation (for speed)
                subset_size = min(len(self.train_dataset), max(100, len(self.train_dataset) // 5))
                train_subset = self.train_dataset.select(range(subset_size))
                
                # Temporarily disable logging to avoid clutter
                original_logging = args.logging_steps
                args.logging_steps = 100000
                
                # Evaluate on training subset
                train_metrics = self.trainer.evaluate(train_subset, metric_key_prefix="train")
                train_loss = train_metrics.get('train_loss', 0)
                train_accuracy = train_metrics.get('train_accuracy', 0)
                
                # Restore logging
                args.logging_steps = original_logging
            except Exception as e:
                print(f"\n⚠️  Warning: Could not evaluate training set: {e}")
                train_loss = 0
                train_accuracy = 0
        
        # Calculate validation metrics on full validation set
        val_loss = 0
        val_accuracy = 0
        if self.trainer and self.val_dataset:
            try:
                # Temporarily disable logging
                original_logging = args.logging_steps
                args.logging_steps = 100000
                
                # Evaluate on full validation set
                val_metrics = self.trainer.evaluate(self.val_dataset, metric_key_prefix="val")
                val_loss = val_metrics.get('val_loss', 0)
                val_accuracy = val_metrics.get('val_accuracy', 0)
                
                # Restore logging
                args.logging_steps = original_logging
            except Exception as e:
                print(f"\n⚠️  Warning: Could not evaluate validation set: {e}")
                val_loss = 0
                val_accuracy = 0
        
        print("Done!")
        
        # Store metrics
        self.epoch_metrics.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_accuracy),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy)
        })
        
        # Print in the requested format
        print(f"Epoch {epoch}/{self.num_epochs} "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train NER model with validation set")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--test_dataset_path", type=str, default=None, help="Path to test dataset (optional)")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--model_checkpoint", type=str, default="distilbert-base-uncased", help="Model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalanced data")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NER MODEL TRAINING - WITH VALIDATION SET")
    print("=" * 70)
    print(f"\n⚙️  CPU thread count: {torch.get_num_threads()}")
    print(f"⚙️  Batch size: {args.batch_size}")
    print(f"⚙️  Epochs: {args.num_train_epochs}")
    print(f"⚙️  Learning rate: {args.learning_rate}")
    print(f"⚙️  Validation split: {args.validation_split:.1%}")
    print(f"⚙️  Class weights: {'ENABLED' if args.use_class_weights else 'DISABLED'}")
    
    # Load data
    print(f"\n📥 Loading training data from: {args.dataset_path}")
    all_train_data = load_data(args.dataset_path)
    print(f"✅ Loaded {len(all_train_data)} training examples")
    
    # Split into train and validation
    print(f"\n✂️  Splitting into train ({1-args.validation_split:.0%}) and validation ({args.validation_split:.0%})...")
    train_data, val_data = train_test_split(
        all_train_data, 
        test_size=args.validation_split, 
        random_state=42
    )
    print(f"✅ Train: {len(train_data)} examples")
    print(f"✅ Validation: {len(val_data)} examples")
    
    # Get label list
    label_list = get_label_list(all_train_data)
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    
    print(f"\n🏷️  Label list ({len(label_list)} labels):")
    print(label_list)
    
    # Analyze label distribution
    label_counts = {}
    total_tokens = 0
    for item in train_data:
        for tag in item['ner_tags']:
            label_counts[tag] = label_counts.get(tag, 0) + 1
            total_tokens += 1
    
    print(f"\n📊 Training label distribution:")
    for label in label_list:
        count = label_counts.get(label, 0)
        percentage = (count / total_tokens) * 100
        marker = "📍" if label != 'O' else "  "
        print(f"   {marker} {label:10s}: {count:6d} ({percentage:5.1f}%)")
    
    # Compute class weights if enabled
    class_weights = None
    if args.use_class_weights:
        print(f"\n⚖️  Computing class weights for imbalanced data...")
        class_weights = compute_class_weights(train_data, label_to_id)
        print(f"   Class weights computed:")
        for i, label in enumerate(label_list[:5]):  # Show first 5
            print(f"      {label:10s}: weight={class_weights[i]:.3f}")
        if len(label_list) > 5:
            print(f"      ... and {len(label_list) - 5} more")
    
    # Convert to Datasets
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
    
    # Setup training arguments WITHOUT automatic evaluation
    # We'll do evaluation manually in the callback for better control
    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        eval_strategy="no",  # Disable automatic evaluation
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        fp16=False,
        dataloader_num_workers=0,
        no_cuda=True,
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Metric
    metric = evaluate.load("seqeval")
    
    # Create callback for progress printing
    progress_callback = ProgressCallback(args.num_train_epochs)
    
    # Create trainer (with or without class weights)
    if args.use_class_weights and class_weights is not None:
        print("\n🏋️  Using WeightedTrainer with class weights")
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=lambda p: compute_metrics(p, label_list, metric),
            class_weights=class_weights.to(model.device),
            callbacks=[progress_callback]
        )
    else:
        print("\n🏋️  Using standard Trainer (no class weights)")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=lambda p: compute_metrics(p, label_list, metric),
            callbacks=[progress_callback]
        )
    
    # Set datasets in callback after trainer is created
    progress_callback.trainer = trainer
    progress_callback.set_datasets(tokenized_train, tokenized_val)
    
    # Train
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING WITH VALIDATION...")
    print("=" * 70)
    print(f"⏱️  Estimated time: {args.num_train_epochs * 10}-{args.num_train_epochs * 20} minutes (CPU)")
    print("=" * 70 + "\n")
    
    train_result = trainer.train()
    
    # Get final validation metrics
    print("\n" + "=" * 70)
    print("📊 EVALUATING ON VALIDATION SET...")
    print("=" * 70)
    val_metrics = trainer.evaluate()
    
    print(f"\n✅ Final Validation Metrics:")
    print(f"   Loss:      {val_metrics['eval_loss']:.4f}")
    print(f"   Accuracy:  {val_metrics['eval_accuracy']:.4f}")
    print(f"   Precision: {val_metrics['eval_precision']:.4f}")
    print(f"   Recall:    {val_metrics['eval_recall']:.4f}")
    print(f"   F1 Score:  {val_metrics['eval_f1']:.4f}")
    
    # Evaluate on test set if provided
    test_metrics = None
    if args.test_dataset_path and os.path.exists(args.test_dataset_path):
        print("\n" + "=" * 70)
        print("📊 EVALUATING ON TEST SET...")
        print("=" * 70)
        
        test_data = load_data(args.test_dataset_path)
        test_dataset = Dataset.from_dict({
            "tokens": [item["tokens"] for item in test_data],
            "ner_tags": [item["ner_tags"] for item in test_data]
        })
        
        tokenized_test = test_dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id),
            batched=True,
            remove_columns=test_dataset.column_names,
            desc="Tokenizing test"
        )
        
        test_metrics = trainer.evaluate(tokenized_test)
        
        print(f"\n✅ Test Set Metrics:")
        print(f"   Loss:      {test_metrics['eval_loss']:.4f}")
        print(f"   Accuracy:  {test_metrics['eval_accuracy']:.4f}")
        print(f"   Precision: {test_metrics['eval_precision']:.4f}")
        print(f"   Recall:    {test_metrics['eval_recall']:.4f}")
        print(f"   F1 Score:  {test_metrics['eval_f1']:.4f}")
    
    # Save final model
    print(f"\n💾 Saving model to: {args.model_save_path}")
    trainer.save_model(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)
    
    # Save label mapping
    label_map_path = os.path.join(args.model_save_path, "label_map.json")
    with open(label_map_path, 'w') as f:
        json.dump({"label_list": label_list, "label_to_id": label_to_id}, f, indent=2)
    
    # Prepare metrics dictionary
    metrics_dict = {
        "training_info": {
            "model": args.model_checkpoint,
            "num_epochs": args.num_train_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "validation_split": args.validation_split,
            "use_class_weights": args.use_class_weights,
            "num_train_examples": len(train_data),
            "num_val_examples": len(val_data)
        },
        "epoch_metrics": progress_callback.epoch_metrics,
        "final_validation_metrics": {
            "loss": float(val_metrics['eval_loss']),
            "accuracy": float(val_metrics['eval_accuracy']),
            "precision": float(val_metrics['eval_precision']),
            "recall": float(val_metrics['eval_recall']),
            "f1": float(val_metrics['eval_f1'])
        }
    }
    
    # Add test metrics if available
    if test_metrics:
        metrics_dict["num_test_examples"] = len(test_data)
        metrics_dict["test_metrics"] = {
            "loss": float(test_metrics['eval_loss']),
            "accuracy": float(test_metrics['eval_accuracy']),
            "precision": float(test_metrics['eval_precision']),
            "recall": float(test_metrics['eval_recall']),
            "f1": float(test_metrics['eval_f1'])
        }
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.model_save_path, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📁 Model saved to: {args.model_save_path}")
    print(f"📊 Metrics saved to: {metrics_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("📈 TRAINING SUMMARY")
    print("=" * 70)
    
    print("\n🔄 Per-Epoch Progress:")
    for epoch_metric in progress_callback.epoch_metrics:
        print(f"   Epoch {epoch_metric['epoch']}/{args.num_train_epochs}: "
              f"Train Loss={epoch_metric['train_loss']:.4f}, "
              f"Train Acc={epoch_metric['train_accuracy']:.4f}, "
              f"Val Loss={epoch_metric['val_loss']:.4f}, "
              f"Val Acc={epoch_metric['val_accuracy']:.4f}")
    
    print(f"\n🎯 Final Performance:")
    print(f"   Validation Accuracy: {val_metrics['eval_accuracy']:.4f}")
    print(f"   Validation F1 Score: {val_metrics['eval_f1']:.4f}")
    
    if test_metrics:
        print(f"   Test Accuracy:       {test_metrics['eval_accuracy']:.4f}")
        print(f"   Test F1 Score:       {test_metrics['eval_f1']:.4f}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()