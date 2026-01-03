import argparse
import json
import os
from transformers import pipeline

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Run NER inference pipeline")
    parser.add_argument("--model_load_path", type=str, required=True, help="Path to the saved model directory")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the test data file (JSON)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the prediction results")
    
    args = parser.parse_args()

    # 2. Validation
    if not os.path.exists(args.model_load_path):
        raise FileNotFoundError(f"Model path not found: {args.model_load_path}")
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    print(f"Loading model from {args.model_load_path}...")
    
    # 3. Initialize Pipeline
    # aggregation_strategy="simple" merges sub-tokens (e.g. "San" + "Francisco") into one entity "San Francisco"
    nlp = pipeline(
        "token-classification", 
        model=args.model_load_path, 
        tokenizer=args.model_load_path, 
        aggregation_strategy="simple" 
    )

    # 4. Load Test Data
    print(f"Reading test data from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 5. Run Inference
    results = []
    print(f"Running inference on {len(test_data)} sentences...")
    
    for i, item in enumerate(test_data):
        # Reconstruct the sentence from tokens (standard inference usually takes raw text)
        sentence = " ".join(item["tokens"])
        
        # Run the pipeline
        # We assume the model creates standard NER tags
        predictions = nlp(sentence)
        
        # Convert non-serializable types (like numpy floats) to standard python types
        clean_predictions = []
        for pred in predictions:
            clean_pred = {
                "entity_group": pred["entity_group"],
                "score": float(pred["score"]),
                "word": pred["word"],
                "start": pred["start"],
                "end": pred["end"]
            }
            clean_predictions.append(clean_pred)

        results.append({
            "sentence_id": i,
            "text": sentence,
            "ground_truth_tags": item["ner_tags"], # The original tags for comparison
            "predicted_entities": clean_predictions
        })

    # 6. Save Output
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()