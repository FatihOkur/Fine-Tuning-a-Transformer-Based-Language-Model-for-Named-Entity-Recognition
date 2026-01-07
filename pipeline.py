import argparse
import json
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import numpy as np
from collections import Counter


torch.set_num_threads(4)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

def convert_to_native(obj):
    """Convert any object to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

def load_test_data(input_file):
    """Load test data from JSON file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def run_ner_pipeline(model_path, input_file, output_file):
    """Run NER pipeline on test data"""
    print("=" * 60)
    print("NER PIPELINE - MODEL TESTİ")
    print("=" * 60)
    
    # Load model and tokenizer
    print(f"\n Loading the model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    print(f" Model loaded!")
    print(f" Model type: {model.config.model_type}")
    print(f" Label count: {model.config.num_labels}")
    
    # Create NER pipeline
    print("\n Creating the NER pipeline")
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1
    )
    print(" Pipeline is ready")
    
    # Load test data
    print(f"\n Loading the test data: {input_file}")
    test_data = load_test_data(input_file)
    print(f" {len(test_data)} test example is uploaded")
    
    # Process
    print("\n Processing the test data")
    print("-" * 60)
    
    results = []
    for i, item in enumerate(test_data):
        sentence = " ".join(item["tokens"])
        predictions = ner_pipeline(sentence)
        
        result = {
            "sentence_id": int(i + 1),
            "tokens": list(item["tokens"]),
            "true_labels": list(item["ner_tags"]),
            "predicted_entities": [],
            "sentence": str(sentence)
        }
        
        for pred in predictions:
            entity = {
                "entity_group": str(pred["entity_group"]),
                "score": float(pred["score"]),
                "word": str(pred["word"]),
                "start": int(pred["start"]),
                "end": int(pred["end"])
            }
            result["predicted_entities"].append(entity)
        
        results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"Processing: {i + 1}/{len(test_data)} sample")
    
    print("\n" + "-" * 60)
    print(" Process complete")
    
    # Calculate metrics
    print("\n Metrics are calculated")
    total_entities = sum(len(r["predicted_entities"]) for r in results)
    
    metrics = {
        "total_examples": int(len(results)),
        "total_tokens": int(sum(len(r["tokens"]) for r in results)),
        "total_entities_found": int(total_entities),
        "avg_entities_per_sentence": float(total_entities / len(results))
    }
    
    # Save results
    print(f"\n Saving the results {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    output_data = {
        "metrics": metrics,
        "predictions": results
    }
    
    output_data = convert_to_native(output_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print("Results are saved.")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n Total number of samples: {metrics['total_examples']}")
    print(f" Total number of tokens: {metrics['total_tokens']}")
    print(f" Total number of entities found: {metrics['total_entities_found']}")
    print(f" Number of average entities per sentence: {metrics['avg_entities_per_sentence']:.2f}")
    
    # Entity types
    entity_types = {}
    for result in results:
        for entity in result["predicted_entities"]:
            entity_type = entity["entity_group"]
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    if entity_types:
        print(f"\n  Types of entities found:")
        for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / metrics['total_entities_found']) * 100
            print(f"   {entity_type:10s}: {count:5d} ({percentage:5.1f}%)")
    
    # Confidence scores
    all_scores = [entity['score'] for result in results for entity in result['predicted_entities']]
    if all_scores:
        print(f"\n Confidence Scores:")
        print(f"   Avarage: {np.mean(all_scores):.2%}")
        print(f"   The lowest: {np.min(all_scores):.2%}")
        print(f"   The highest: {np.max(all_scores):.2%}")
    
    print("\n" + "=" * 60)
    print(" TEST COMPLETED!")
    print("=" * 60)
    print(f"\n Detailed results: {output_file}")
    
    # Sample predictions
    print(f"\n First 3 predictions:")
    print("-" * 60)
    for i, result in enumerate(results[:3]):
        print(f"\n{i+1}. Sentence:")
        print(f"   \"{result['sentence'][:100]}...\"")
        if result['predicted_entities']:
            print(f"   Found Entities:")
            for entity in result['predicted_entities'][:5]:
                print(f"   • {entity['word']:20s} → {entity['entity_group']:8s} ({entity['score']:.1%})")
        else:
            print(f"   (No Entity Found)")
    
    print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Run NER pipeline on test data")
    parser.add_argument("--model_load_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    
    args = parser.parse_args()
    run_ner_pipeline(args.model_load_path, args.input_file, args.output_file)

if __name__ == "__main__":
    main()