#!/bin/bash

################################################################################
# NER Model Training and Testing Pipeline
# COMP 451 - Project #3
# 
# This script automates the complete workflow:
# 1. Data preparation
# 2. Model training with validation
# 3. Testing and evaluation
################################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_NAME="distilbert-base-uncased"
MODEL_SAVE_PATH="./ner_model"
DATA_DIR="./data"
RESULTS_DIR="./results"
NUM_EPOCHS=10
LEARNING_RATE=5e-5
BATCH_SIZE=8
VALIDATION_SPLIT=0.1

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_file() {
    if [ ! -f "$1" ]; then
        print_error "Required file not found: $1"
        exit 1
    fi
}

check_python() {
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_info "Python version: $PYTHON_VERSION"
}

################################################################################
# Main Pipeline
################################################################################

main() {
    print_header "NER MODEL TRAINING AND TESTING PIPELINE"
    
    echo "Configuration:"
    echo "  Model: $MODEL_NAME"
    echo "  Epochs: $NUM_EPOCHS"
    echo "  Learning Rate: $LEARNING_RATE"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Validation Split: $VALIDATION_SPLIT"
    echo ""
    
    # Check Python installation
    check_python
    
    # Check required files
    print_info "Checking required files..."
    check_file "train_with_validation.py"
    check_file "pipeline.py"
    check_file "prepare_datasets_clean.py"
    check_file "Dataset1.txt"
    check_file "Dataset2.json"
    print_success "All required files found"
    
    # Clean previous results
    print_header "STEP 0: CLEANING PREVIOUS RESULTS"
    if [ -d "$MODEL_SAVE_PATH" ]; then
        print_warning "Removing previous model: $MODEL_SAVE_PATH"
        rm -rf "$MODEL_SAVE_PATH"
    fi
    if [ -d "$RESULTS_DIR" ]; then
        print_warning "Removing previous results: $RESULTS_DIR"
        rm -rf "$RESULTS_DIR"
    fi
    if [ -d "$DATA_DIR" ]; then
        print_warning "Removing previous data: $DATA_DIR"
        rm -rf "$DATA_DIR"
    fi
    print_success "Cleanup complete"
    
    # Step 1: Prepare datasets
    print_header "STEP 1: PREPARING DATASETS"
    print_info "Splitting Dataset1 (85% train, 15% test)..."
    print_info "Adding Dataset2 to training data..."
    
    python prepare_datasets_clean.py
    
    if [ $? -ne 0 ]; then
        print_error "Data preparation failed!"
        exit 1
    fi
    
    check_file "$DATA_DIR/train.json"
    check_file "$DATA_DIR/test.json"
    print_success "Datasets prepared successfully"
    
    # Step 2: Train model
    print_header "STEP 2: TRAINING MODEL"
    print_info "This may take 100-200 minutes on CPU (~10-20 min/epoch)"
    print_info "Training with validation monitoring..."
    echo ""
    
    python train_with_validation.py \
        --model_save_path "$MODEL_SAVE_PATH" \
        --dataset_path "$DATA_DIR/train.json" \
        --test_dataset_path "$DATA_DIR/test.json" \
        --num_train_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --validation_split $VALIDATION_SPLIT \
        --model_checkpoint "$MODEL_NAME" \
        --use_class_weights
    
    if [ $? -ne 0 ]; then
        print_error "Training failed!"
        exit 1
    fi
    
    check_file "$MODEL_SAVE_PATH/config.json"
    check_file "$MODEL_SAVE_PATH/training_metrics.json"
    print_success "Model trained successfully"
    
    # Step 3: Run inference
    print_header "STEP 3: RUNNING INFERENCE ON TEST DATA"
    mkdir -p "$RESULTS_DIR"
    
    python pipeline.py \
        --model_load_path "$MODEL_SAVE_PATH" \
        --input_file "$DATA_DIR/test.json" \
        --output_file "$RESULTS_DIR/predictions.json"
    
    if [ $? -ne 0 ]; then
        print_error "Inference failed!"
        exit 1
    fi
    
    check_file "$RESULTS_DIR/predictions.json"
    print_success "Inference completed successfully"
    
    # Final summary
    print_header "✅ PIPELINE COMPLETED SUCCESSFULLY"
    
    echo "📁 Generated Files:"
    echo ""
    echo "  Data:"
    echo "    - $DATA_DIR/train.json           (Training data)"
    echo "    - $DATA_DIR/test.json            (Test data)"
    echo ""
    echo "  Model:"
    echo "    - $MODEL_SAVE_PATH/config.json            (Model configuration)"
    echo "    - $MODEL_SAVE_PATH/model.safetensors      (Model weights)"
    echo "    - $MODEL_SAVE_PATH/training_metrics.json  (Training metrics)"
    echo "    - $MODEL_SAVE_PATH/label_map.json         (Label mappings)"
    echo ""
    echo "  Results:"
    echo "    - $RESULTS_DIR/predictions.json  (Test predictions)"
    echo ""
    
    # Display metrics if jq is available
    if command -v jq &> /dev/null; then
        print_header "📊 MODEL PERFORMANCE SUMMARY"
        
        echo "Training Configuration:"
        jq -r '.training_info | 
            "  Model: \(.model)\n  Epochs: \(.num_epochs)\n  Learning Rate: \(.learning_rate)\n  Batch Size: \(.batch_size)\n  Train Examples: \(.num_train_examples)\n  Val Examples: \(.num_val_examples)"' \
            "$MODEL_SAVE_PATH/training_metrics.json"
        
        echo ""
        echo "Final Validation Metrics:"
        jq -r '.final_validation_metrics | 
            "  Accuracy:  \(.accuracy | tonumber * 100 | round / 100)%\n  Precision: \(.precision | tonumber * 100 | round / 100)%\n  Recall:    \(.recall | tonumber * 100 | round / 100)%\n  F1 Score:  \(.f1 | tonumber * 100 | round / 100)%"' \
            "$MODEL_SAVE_PATH/training_metrics.json"
        
        if jq -e '.test_metrics' "$MODEL_SAVE_PATH/training_metrics.json" > /dev/null 2>&1; then
            echo ""
            echo "Test Set Metrics:"
            jq -r '.test_metrics | 
                "  Accuracy:  \(.accuracy | tonumber * 100 | round / 100)%\n  Precision: \(.precision | tonumber * 100 | round / 100)%\n  Recall:    \(.recall | tonumber * 100 | round / 100)%\n  F1 Score:  \(.f1 | tonumber * 100 | round / 100)%"' \
                "$MODEL_SAVE_PATH/training_metrics.json"
        fi
        
        echo ""
        echo "Test Predictions Summary:"
        jq -r '.metrics | 
            "  Total Examples:    \(.total_examples)\n  Total Tokens:      \(.total_tokens)\n  Entities Found:    \(.total_entities_found)\n  Avg per Sentence:  \(.avg_entities_per_sentence | tonumber * 100 | round / 100)"' \
            "$RESULTS_DIR/predictions.json"
    else
        print_warning "Install 'jq' to see formatted metrics"
        echo ""
        echo "View metrics manually:"
        echo "  cat $MODEL_SAVE_PATH/training_metrics.json"
        echo "  cat $RESULTS_DIR/predictions.json"
    fi
    
    echo ""
    print_header "📝 NEXT STEPS"
    echo "1. View training metrics:"
    echo "   cat $MODEL_SAVE_PATH/training_metrics.json"
    echo ""
    echo "2. View test predictions:"
    echo "   cat $RESULTS_DIR/predictions.json"
    echo ""
    echo "3. Use the model for new predictions:"
    echo "   python pipeline.py \\"
    echo "       --model_load_path $MODEL_SAVE_PATH \\"
    echo "       --input_file <your_data.json> \\"
    echo "       --output_file <your_output.json>"
    echo ""
    echo ""
}

################################################################################
# Script Entry Point
################################################################################

# Parse command line arguments
SKIP_CLEANUP=false
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-cleanup)
            SKIP_CLEANUP=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            NUM_EPOCHS=3
            print_warning "Quick mode: Training for only 3 epochs"
            shift
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-cleanup    Don't remove previous model and results"
            echo "  --quick           Quick training (3 epochs instead of 10)"
            echo "  --epochs N        Set number of training epochs to N"
            echo "  --help            Show this help message"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main pipeline
main

exit 0