import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

def load_dataset1_clean(filepath):
    """
    Load Dataset1 (tab-separated format)
    Only keep: geo, org, per, tim (remove gpe, art)
    """
    print(f"Reading {filepath}...")
    df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
    
    print(f"Total rows: {len(df)}")
    
    # Group by sentence
    sentences = []
    current_sentence = {'tokens': [], 'ner_tags': []}
    current_sentence_id = None
    
    removed_labels = {'B-gpe', 'I-gpe', 'B-art', 'I-art'}
    removed_count = 0
    
    for idx, row in df.iterrows():
        sentence_id = row['Sentence #']
        
        # Only start new sentence when ID CHANGES
        if pd.notna(sentence_id) and sentence_id != current_sentence_id:
            # Save previous sentence if it exists
            if current_sentence_id is not None and current_sentence['tokens']:
                sentences.append(current_sentence)
            
            # Start new sentence
            current_sentence = {'tokens': [], 'ner_tags': []}
            current_sentence_id = sentence_id
        
        # Clean label: Convert gpe, art to O
        original_tag = row['Tag']
        if original_tag in removed_labels:
            cleaned_tag = 'O'
            removed_count += 1
        else:
            cleaned_tag = original_tag
        
        # Add token to current sentence
        current_sentence['tokens'].append(row['Word'])
        current_sentence['ner_tags'].append(cleaned_tag)
    
    # Add last sentence
    if current_sentence['tokens']:
        sentences.append(current_sentence)
    
    print(f"\n✅ Loaded {len(sentences)} sentences from Dataset1")
    print(f"🧹 Cleaned {removed_count} labels (gpe, art → O)")
    
    # Print statistics
    if sentences:
        total_tokens = sum(len(s['tokens']) for s in sentences)
        avg_tokens = total_tokens / len(sentences)
        print(f"   Total tokens: {total_tokens}")
        print(f"   Average tokens per sentence: {avg_tokens:.1f}")
        
        # Count labels AFTER cleaning
        label_counts = {}
        for sent in sentences:
            for tag in sent['ner_tags']:
                label_counts[tag] = label_counts.get(tag, 0) + 1
        
        print(f"\n   Label distribution (AFTER cleaning):")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            marker = "📍" if label != 'O' else "  "
            print(f"      {marker} {label:10s}: {count:5d}")
    
    return sentences

def load_dataset2_clean(filepath):
    """
    Load Dataset2 (JSON format)
    Only keep: geo, org, per, tim (remove cur)
    """
    print(f"\nReading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = []
    removed_labels = {'B-cur', 'I-cur'}
    removed_count = 0
    
    for annotation in data['annotations']:
        tokens = [entity['word'] for entity in annotation['entities']]
        
        # Clean labels: Convert cur to O
        cleaned_tags = []
        for entity in annotation['entities']:
            if entity['label'] in removed_labels:
                cleaned_tags.append('O')
                removed_count += 1
            else:
                cleaned_tags.append(entity['label'])
        
        sentences.append({'tokens': tokens, 'ner_tags': cleaned_tags})
    
    print(f"✅ Loaded {len(sentences)} sentences from Dataset2")
    print(f"🧹 Cleaned {removed_count} labels (cur → O)")
    
    # Print statistics
    if sentences:
        total_tokens = sum(len(s['tokens']) for s in sentences)
        avg_tokens = total_tokens / len(sentences)
        print(f"   Total tokens: {total_tokens}")
        print(f"   Average tokens per sentence: {avg_tokens:.1f}")
        
        # Count labels AFTER cleaning
        label_counts = {}
        for sent in sentences:
            for tag in sent['ner_tags']:
                label_counts[tag] = label_counts.get(tag, 0) + 1
        
        print(f"\n   Label distribution (AFTER cleaning):")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            marker = "📍" if label != 'O' else "  "
            print(f"      {marker} {label:10s}: {count:5d}")
    
    return sentences

def prepare_datasets_clean(dataset1_path, dataset2_path, output_dir='./data', test_size=0.15):
    """Prepare train and test datasets with ONLY common labels"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("DATASET PREPARATION - CLEAN VERSION")
    print("Only using common labels: geo, org, per, tim")
    print("=" * 80)
    
    # Load datasets
    print("\n📥 STEP 1: Loading Dataset1 (cleaning gpe, art)...")
    dataset1 = load_dataset1_clean(dataset1_path)
    
    print("\n📥 STEP 2: Loading Dataset2 (cleaning cur)...")
    dataset2 = load_dataset2_clean(dataset2_path)
    
    # Verify dataset1 is not empty
    if not dataset1:
        raise ValueError("❌ Dataset1 is empty!")
    
    # Check if sentences have multiple tokens
    avg_tokens = sum(len(s['tokens']) for s in dataset1) / len(dataset1)
    print(f"\n📊 Dataset1 verification:")
    print(f"   Sentences: {len(dataset1)}")
    print(f"   Average tokens per sentence: {avg_tokens:.1f}")
    
    if avg_tokens < 5:
        print("\n⚠️  WARNING: Sentences are very short!")
    else:
        print(f"   ✅ Sentence length looks good!")
    
    # Split Dataset1 into train (85%) and test (15%)
    print(f"\n✂️  STEP 3: Splitting Dataset1 (85% train, 15% test)...")
    train_data, test_data = train_test_split(dataset1, test_size=test_size, random_state=42)
    
    # Add Dataset2 to training data
    print(f"\n➕ STEP 4: Adding Dataset2 to training data...")
    original_train_size = len(train_data)
    train_data.extend(dataset2)
    print(f"   Added {len(dataset2)} sentences from Dataset2")
    
    # Verify all labels are consistent
    print(f"\n🔍 STEP 5: Verifying label consistency...")
    all_train_labels = set()
    all_test_labels = set()
    
    for sent in train_data:
        all_train_labels.update(sent['ner_tags'])
    for sent in test_data:
        all_test_labels.update(sent['ner_tags'])
    
    print(f"\n   Train labels: {sorted(all_train_labels)}")
    print(f"   Test labels: {sorted(all_test_labels)}")
    
    # Extract entity types (without B-/I- prefix)
    train_entity_types = set()
    test_entity_types = set()
    
    for label in all_train_labels:
        if label != 'O' and '-' in label:
            train_entity_types.add(label.split('-')[1])
    
    for label in all_test_labels:
        if label != 'O' and '-' in label:
            test_entity_types.add(label.split('-')[1])
    
    print(f"\n   Train entity types: {sorted(train_entity_types)}")
    print(f"   Test entity types: {sorted(test_entity_types)}")
    
    if train_entity_types == test_entity_types:
        print(f"\n   ✅✅✅ PERFECT! Train and test have IDENTICAL entity types!")
    else:
        print(f"\n   ⚠️  WARNING: Entity types differ!")
        print(f"   Only in train: {train_entity_types - test_entity_types}")
        print(f"   Only in test: {test_entity_types - train_entity_types}")
    
    print(f"\n📊 FINAL DATASET SIZES:")
    print("=" * 80)
    print(f"Train data:")
    print(f"   From Dataset1: {original_train_size} sentences")
    print(f"   From Dataset2: {len(dataset2)} sentences")
    print(f"   TOTAL:         {len(train_data)} sentences")
    print(f"\nTest data:")
    print(f"   From Dataset1: {len(test_data)} sentences")
    
    # Calculate token counts
    train_tokens = sum(len(s['tokens']) for s in train_data)
    test_tokens = sum(len(s['tokens']) for s in test_data)
    
    # Calculate entity counts
    train_entities = sum(sum(1 for tag in s['ner_tags'] if tag != 'O') for s in train_data)
    test_entities = sum(sum(1 for tag in s['ner_tags'] if tag != 'O') for s in test_data)
    
    print(f"\n📊 TOKEN & ENTITY COUNTS:")
    print(f"   Train: {train_tokens:,} tokens, {train_entities} entities ({train_entities/train_tokens*100:.1f}%)")
    print(f"   Test:  {test_tokens:,} tokens, {test_entities} entities ({test_entities/test_tokens*100:.1f}%)")
    
    # Save datasets
    train_path = os.path.join(output_dir, 'train_clean.json')
    test_path = os.path.join(output_dir, 'test_clean.json')
    
    print(f"\n💾 STEP 6: Saving cleaned datasets...")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Saved: {train_path}")
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Saved: {test_path}")
    
    print("\n" + "=" * 80)
    print("✅ CLEAN DATASET PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\n🎯 Key improvements:")
    print(f"   • IDENTICAL labels in train and test")
    print(f"   • Removed problematic labels: gpe, art, cur")
    print(f"   • Clean evaluation: {sorted(train_entity_types)}")
    print(f"\nNext steps:")
    print(f"1. Delete old model: rmdir /s /q ner_model")
    print(f"2. Train with clean data:")
    print(f"   python train_with_weights.py --model_save_path ./ner_model --dataset_path ./data/train_clean.json --num_train_epochs 10 --use_class_weights")
    print(f"3. Test with clean data:")
    print(f"   python pipeline.py --model_load_path ./ner_model --input_file ./data/test_clean.json --output_file ./results/predictions_clean.json")
    
    return train_path, test_path

if __name__ == "__main__":
    # Prepare clean datasets
    prepare_datasets_clean(
        dataset1_path='./Dataset1.txt',
        dataset2_path='./Dataset2.json',
        output_dir='./data'
    )