"""
Sonuçları güzel bir şekilde görüntüle
"""
import json
import os

def view_results(results_file="./results/predictions.json"):
    """View test results in a readable format"""
    
    if not os.path.exists(results_file):
        print("❌ Sonuç dosyası bulunamadı!")
        print(f"   Aradığım: {results_file}")
        print("\n💡 Önce 'test_model.bat' çalıştırın!")
        return
    
    print("=" * 70)
    print("📊 MODEL TEST SONUÇLARI")
    print("=" * 70)
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Metrics
    if "metrics" in data:
        metrics = data["metrics"]
        print("\n📈 GENEL İSTATİSTİKLER:")
        print("-" * 70)
        print(f"✅ Test edilen cümle sayısı: {metrics['total_examples']:,}")
        print(f"✅ Toplam token sayısı: {metrics['total_tokens']:,}")
        print(f"✅ Bulunan entity sayısı: {metrics['total_entities_found']:,}")
        print(f"✅ Cümle başına ortalama: {metrics['avg_entities_per_sentence']:.2f} entity")
        
        results_list = data["predictions"]
    else:
        results_list = data
        print(f"\n✅ Toplam {len(results_list)} test örneği")
    
    # Entity type counts
    entity_counts = {}
    all_scores = []
    
    for result in results_list:
        for entity in result.get("predicted_entities", []):
            entity_type = entity["entity_group"]
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            all_scores.append(entity["score"])
    
    if entity_counts:
        print("\n🏷️  ENTITY TİPLERİ DAĞILIMI:")
        print("-" * 70)
        total = sum(entity_counts.values())
        for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            print(f"{entity_type:12s}: {count:5d} ({percentage:5.1f}%) {bar}")
    
    if all_scores:
        import numpy as np
        print("\n🎯 GÜVEN SKORLARI:")
        print("-" * 70)
        print(f"Ortalama: {np.mean(all_scores):.2%}")
        print(f"Minimum:  {np.min(all_scores):.2%}")
        print(f"Maximum:  {np.max(all_scores):.2%}")
        
        # Score distribution
        high_conf = sum(1 for s in all_scores if s > 0.9)
        med_conf = sum(1 for s in all_scores if 0.7 <= s <= 0.9)
        low_conf = sum(1 for s in all_scores if s < 0.7)
        
        print("\nGüven Dağılımı:")
        print(f"  Yüksek (>90%): {high_conf:5d} ({high_conf/len(all_scores)*100:5.1f}%)")
        print(f"  Orta (70-90%): {med_conf:5d} ({med_conf/len(all_scores)*100:5.1f}%)")
        print(f"  Düşük (<70%):  {low_conf:5d} ({low_conf/len(all_scores)*100:5.1f}%)")
    
    # Show examples
    print("\n" + "=" * 70)
    print("📝 ÖRNEK TAHMİNLER (İlk 5)")
    print("=" * 70)
    
    for i, result in enumerate(results_list[:5]):
        print(f"\n{i+1}. Cümle:")
        sentence = result.get("sentence", " ".join(result.get("tokens", [])))
        print(f"   \"{sentence[:80]}{'...' if len(sentence) > 80 else ''}\"")
        
        entities = result.get("predicted_entities", [])
        if entities:
            print(f"\n   Bulunan {len(entities)} entity:")
            for entity in entities[:5]:  # İlk 5 entity
                word = entity.get("word", "?")
                entity_type = entity.get("entity_group", "?")
                score = entity.get("score", 0)
                
                # Güven seviyesi emoji
                if score > 0.9:
                    conf_emoji = "🟢"
                elif score > 0.7:
                    conf_emoji = "🟡"
                else:
                    conf_emoji = "🔴"
                
                print(f"   {conf_emoji} {word:20s} → {entity_type:10s} ({score:.1%})")
            
            if len(entities) > 5:
                print(f"   ... ve {len(entities)-5} entity daha")
        else:
            print("   (Entity bulunamadı)")
    
    # Accuracy hints
    print("\n" + "=" * 70)
    print("💡 MODEL DEĞERLENDİRMESİ")
    print("=" * 70)
    
    if all_scores:
        avg_score = np.mean(all_scores)
        
        if avg_score > 0.9:
            print("\n✅ MÜKEMMEL! Model çok yüksek güvenle tahmin yapıyor.")
            print("   Ortalama güven >90% - Model iyi eğitilmiş.")
        elif avg_score > 0.8:
            print("\n✅ İYİ! Model güvenilir tahminler yapıyor.")
            print("   Ortalama güven >80% - Kabul edilebilir performans.")
        elif avg_score > 0.7:
            print("\n⚠️  ORTA. Model genel olarak iyi ama bazı tahminlerde kararsız.")
            print("   Ortalama güven >70% - Daha fazla eğitim gerekebilir.")
        else:
            print("\n❌ DÜŞÜK. Model çok kararsız tahminler yapıyor.")
            print("   Ortalama güven <70% - Model daha fazla eğitilmeli.")
        
        print(f"\n📊 Sonuç Özeti:")
        print(f"   • {len(results_list):,} cümle test edildi")
        print(f"   • {len(all_scores):,} entity bulundu")
        print(f"   • Ortalama güven: {avg_score:.1%}")
    
    print("\n" + "=" * 70)
    print(f"📁 Detaylı sonuçlar: {results_file}")
    print("=" * 70)

if __name__ == "__main__":
    print("\n")
    view_results()
    print("\n")
    input("Çıkmak için Enter'a basın...")