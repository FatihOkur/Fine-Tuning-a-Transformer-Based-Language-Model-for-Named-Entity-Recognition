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

def main():
    parser = argparse.ArgumentParser(description="Train NER model with class weights")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs (default: 10)")
    parser.add_argument("--model_checkpoint", type=str, default="distilbert-base-uncased", help="Model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights for imbalanced data")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NER MODEL TRAINING - WITH CLASS WEIGHTS")
    print("=" * 70)
    print(f"\n⚙️  CPU thread count: {torch.get_num_threads()}")
    print(f"⚙️  Batch size: {args.batch_size}")
    print(f"⚙️  Epochs: {args.num_train_epochs}")
    print(f"⚙️  Learning rate: {args.learning_rate}")
    print(f"⚙️  Class weights: {'ENABLED' if args.use_class_weights else 'DISABLED'}")
    
    # Load data
    print(f"\n📥 Loading training data from: {args.dataset_path}")
    train_data = load_data(args.dataset_path)
    print(f"✅ Loaded {len(train_data)} training examples")
    
    # Get label list
    label_list = get_label_list(train_data)
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
    
    print(f"\n📊 Label distribution:")
    for label in label_list:
        count = label_counts.get(label, 0)
        percentage = (count / total_tokens) * 100
        marker = "📍" if label != 'O' else "  "
        print(f"   {marker} {label:10s}: {count:6d} ({percentage:5.1f}%)")
    
    # Check if data is too small
    entity_count = sum(count for label, count in label_counts.items() if label != 'O')
    if entity_count < 100:
        print(f"\n⚠️  WARNING: Very few entities ({entity_count})!")
        print(f"   Model might not learn well with this little data.")
        print(f"   Recommendation: Add more training data or increase epochs to {args.num_train_epochs * 2}")
    
    # Compute class weights if enabled
    class_weights = None
    if args.use_class_weights:
        print(f"\n⚖️  Computing class weights for imbalanced data...")
        class_weights = compute_class_weights(train_data, label_to_id)
        print(f"   Class weights computed:")
        for i, label in enumerate(label_list):
            print(f"      {label:10s} (ID={i:2d}): weight={class_weights[i]:.3f}")
    
    # Convert to Dataset
    dataset = Dataset.from_dict({
        "tokens": [item["tokens"] for item in train_data],
        "ner_tags": [item["ner_tags"] for item in train_data]
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
    
    # Tokenize dataset
    print("\n🔧 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        eval_strategy="no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=20,
        save_total_limit=1,
        fp16=False,
        dataloader_num_workers=0,
        no_cuda=True,
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Metric
    metric = evaluate.load("seqeval")
    
    # Create trainer (with or without class weights)
    if args.use_class_weights and class_weights is not None:
        print("\n🏋️  Using WeightedTrainer with class weights")
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=lambda p: compute_metrics(p, label_list, metric),
            class_weights=class_weights.to(model.device)
        )
    else:
        print("\n🏋️  Using standard Trainer (no class weights)")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=lambda p: compute_metrics(p, label_list, metric)
        )
    
    # Train
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING...")
    print("=" * 70)
    print(f"⏱️  Estimated time: {args.num_train_epochs * 15}-{args.num_train_epochs * 30} minutes (CPU)")
    print(f"💡 Watch for:")
    print(f"   - Loss should decrease each epoch")
    print(f"   - If loss stays constant, increase epochs or learning rate")
    print("=" * 70 + "\n")
    
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving model to: {args.model_save_path}")
    trainer.save_model(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)
    
    # Save label mapping
    label_map_path = os.path.join(args.model_save_path, "label_map.json")
    with open(label_map_path, 'w') as f:
        json.dump({"label_list": label_list, "label_to_id": label_to_id}, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📁 Model saved to: {args.model_save_path}")
    print(f"\n📊 Model size: ~{sum(os.path.getsize(os.path.join(args.model_save_path, f)) for f in os.listdir(args.model_save_path) if os.path.isfile(os.path.join(args.model_save_path, f))) / (1024*1024):.1f} MB")
    print(f"\n🎯 Next steps:")
    print(f"   1. Test the model: test_model.bat")
    print(f"   2. View results: python view_results.py")

if __name__ == "__main__":
    main()