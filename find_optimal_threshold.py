import argparse
from evaluate_metrics import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="Buscar el threshold óptimo para maximizar F1-Score")
    parser.add_argument("--results-dir", type=str, required=True, help="Carpeta con los JSONs de resultados")
    parser.add_argument("--k", type=int, default=5, help="Valor de K usado en los resultados")
    args = parser.parse_args()

    print(f"\nBuscando threshold óptimo para K={args.k} en: {args.results_dir}")
    print("-" * 65)
    print(f"{'Threshold':<10} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10}")
    print("-" * 65)

    best_f1 = 0.0
    best_thresh = 0.0
    best_metrics = {}

    # Probamos thresholds de 0.5 a 10.0 en pasos de 0.5
    thresholds = [x * 0.5 for x in range(1, 21)]
    
    for thresh in thresholds:
        # Ejecutamos evaluación en modo silencioso
        metrics = run_evaluation(args.results_dir, k=args.k, score_threshold=thresh, quiet=True)
        
        r = metrics["recall"]
        p = metrics["precision"]
        
        # Calculamos F1 Score (Media armónica entre Recall y Precision)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        
        print(f"{thresh:<10.1f} | {r:<10.4f} | {p:<10.4f} | {f1:<10.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = metrics

    print("-" * 65)
    print(f"🏆 MEJOR THRESHOLD: {best_thresh} (F1: {best_f1:.4f})")
    print(f"   Recall: {best_metrics.get('recall', 0):.4f} | Precision: {best_metrics.get('precision', 0):.4f}")

if __name__ == "__main__":
    main()