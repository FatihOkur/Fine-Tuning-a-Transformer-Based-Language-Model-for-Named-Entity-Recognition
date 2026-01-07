import json
import os
import sys

def view_metrics(metrics_path="./ner_model/training_metrics.json"):
    """View and format training metrics from JSON file"""
    
    if not os.path.exists(metrics_path):
        print(f" ERROR: Metrics file not found: {metrics_path}")
        print("\nPlease train the model first using:")
        print("  python train_with_validation.py --model_save_path ./ner_model --dataset_path ./data/train.json --num_train_epochs 3")
        return
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    print("\n" + "=" * 80)
    print(" TRAINING METRICS SUMMARY")
    print("=" * 80)
    
    # Training info
    if 'training_info' in metrics:
        info = metrics['training_info']
        print("\n Training Configuration:")
        print(f"   Model:              {info.get('model', 'N/A')}")
        print(f"   Number of Epochs:   {info.get('num_epochs', 'N/A')}")
        print(f"   Batch Size:         {info.get('batch_size', 'N/A')}")
        print(f"   Learning Rate:      {info.get('learning_rate', 'N/A')}")
        print(f"   Validation Split:   {info.get('validation_split', 0)*100:.0f}%")
        print(f"   Class Weights:      {'Yes' if info.get('use_class_weights', False) else 'No'}")
        print(f"   Train Examples:     {info.get('num_train_examples', 'N/A')}")
        print(f"   Validation Examples: {info.get('num_val_examples', 'N/A')}")
    
    # Epoch-by-epoch progress
    if 'epoch_metrics' in metrics:
        print("\n Training Progress (Per Epoch):")
        print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
        print("-" * 60)
        
        for epoch_data in metrics['epoch_metrics']:
            epoch = epoch_data.get('epoch', 'N/A')
            train_loss = epoch_data.get('train_loss', 0)
            train_acc = epoch_data.get('train_accuracy', 0)
            val_loss = epoch_data.get('val_loss', 0)
            val_acc = epoch_data.get('val_accuracy', 0)
            
            print(f"{epoch}/{metrics['training_info']['num_epochs']:<6} "
                  f"{train_loss:<12.4f} {train_acc:<12.4f} "
                  f"{val_loss:<12.4f} {val_acc:<12.4f}")
    
    # Final validation metrics
    if 'final_validation_metrics' in metrics:
        val_metrics = metrics['final_validation_metrics']
        print("\n Final Validation Metrics:")
        print(f"   Loss:      {val_metrics.get('loss', 0):.4f}")
        print(f"   Accuracy:  {val_metrics.get('accuracy', 0):.4f}")
        print(f"   Precision: {val_metrics.get('precision', 0):.4f}")
        print(f"   Recall:    {val_metrics.get('recall', 0):.4f}")
        print(f"   F1 Score:  {val_metrics.get('f1', 0):.4f}")
    
    # Test metrics if available
    if 'test_metrics' in metrics:
        test_metrics = metrics['test_metrics']
        print("\n Test Set Metrics:")
        print(f"   Test Examples: {metrics.get('num_test_examples', 'N/A')}")
        print(f"   Loss:      {test_metrics.get('loss', 0):.4f}")
        print(f"   Accuracy:  {test_metrics.get('accuracy', 0):.4f}")
        print(f"   Precision: {test_metrics.get('precision', 0):.4f}")
        print(f"   Recall:    {test_metrics.get('recall', 0):.4f}")
        print(f"   F1 Score:  {test_metrics.get('f1', 0):.4f}")
    
    print("\n" + "=" * 80)
    
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        metrics_path = sys.argv[1]
    else:
        metrics_path = "./ner_model/training_metrics.json"
    
    view_metrics(metrics_path)