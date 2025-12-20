import argparse
import json
import os
import statistics
from src.datasets.LongMemEvalDataset import LongMemEvalDataset

def main():
    parser = argparse.ArgumentParser(description="Evaluar métricas de Retrieval (Recall/Precision)")
    parser.add_argument("--results-dir", type=str, required=True, help="Carpeta con los JSONs generados por main.py")
    parser.add_argument("--dataset-set", type=str, default="longmemeval", help="El set usado (longmemeval, investigathon_evaluation)")
    parser.add_argument("--k", type=int, default=5, help="El corte K para las métricas")
    parser.add_argument("--output-file", type=str, help="Ruta del archivo para guardar el reporte (opcional)")
    
    args = parser.parse_args()

    # 1. Cargar Ground Truth (Oracle)
    print(f"Cargando Ground Truth (Oracle) para '{args.dataset_set}'...")
    try:
        # Instanciamos el dataset en modo 'oracle'. 
        # En este modo, la lista 'sessions' de cada instancia contiene SOLO las sesiones relevantes.
        oracle_dataset = LongMemEvalDataset(type="oracle", set=args.dataset_set)
    except ValueError as e:
        print(f"Error cargando dataset: {e}")
        print("Asegúrate de usar un set que tenga oracle disponible (ej. longmemeval, investigathon_evaluation).")
        return

    # Crear mapa: question_id -> {ids: set, type: str}
    ground_truth_map = {}
    for instance in oracle_dataset:
        # Convertimos a string para asegurar coincidencia con el JSON
        relevant_ids = {str(s.session_id) for s in instance.sessions}
        ground_truth_map[instance.question_id] = {
            "ids": relevant_ids,
            "type": getattr(instance, "question_type", "unknown")
        }

    print(f"Ground Truth cargado: {len(ground_truth_map)} preguntas indexadas.")

    # 2. Leer Resultados Generados
    if not os.path.exists(args.results_dir):
        print(f"No se encuentra el directorio de resultados: {args.results_dir}")
        return

    result_files = [f for f in os.listdir(args.results_dir) if f.endswith(".json")]
    print(f"Evaluando {len(result_files)} archivos de resultados en '{args.results_dir}'...")

    metrics_by_type = {}  # { "type_name": {"recalls": [], "precisions": []} }
    missing_in_oracle = 0

    for filename in result_files:
        filepath = os.path.join(args.results_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        
        q_id = result_data.get("question_id")
        
        # Recuperamos la lista de sesiones que trajo nuestro algoritmo (BM25)
        retrieved_ids = [str(x) for x in result_data.get("retrieved_sessions", [])]
        
        # Buscamos las relevantes en el mapa
        if q_id not in ground_truth_map:
            # Esto puede pasar si el dataset oracle tiene menos preguntas o IDs distintos
            missing_in_oracle += 1
            continue
            
        gt_data = ground_truth_map[q_id]
        relevant_ids = gt_data["ids"]
        q_type = gt_data["type"]
        
        # Cortamos en K (por si el archivo tiene más resultados guardados)
        current_k_retrieved = retrieved_ids[:args.k]
        
        # 3. Calcular Métricas
        # Intersección: Cuántas de las recuperadas están en las relevantes
        hits = len(set(current_k_retrieved).intersection(relevant_ids))
        
        # Recall: Hits / Total Relevantes
        recall = hits / len(relevant_ids) if len(relevant_ids) > 0 else 0.0
        
        # Precision: Hits / K
        precision = hits / args.k if args.k > 0 else 0.0
        
        if q_type not in metrics_by_type:
            metrics_by_type[q_type] = {"recalls": [], "precisions": []}
        
        metrics_by_type[q_type]["recalls"].append(recall)
        metrics_by_type[q_type]["precisions"].append(precision)

    # 4. Reporte Final
    lines = []
    lines.append("\n" + "="*60)
    lines.append(f"RESULTADOS DE EVALUACIÓN POR TIPO (K={args.k})")
    lines.append("="*60)
    lines.append(f"{'TIPO':<30} | {'RECALL':<10} | {'PRECISION':<10} | {'COUNT':<5}")
    lines.append("-" * 60)

    all_recalls = []
    all_precisions = []

    for q_type, data in sorted(metrics_by_type.items()):
        avg_r = statistics.mean(data["recalls"])
        avg_p = statistics.mean(data["precisions"])
        count = len(data["recalls"])
        
        all_recalls.extend(data["recalls"])
        all_precisions.extend(data["precisions"])
        
        lines.append(f"{q_type:<30} | {avg_r:.4f}     | {avg_p:.4f}     | {count:<5}")

    if all_recalls:
        global_recall = statistics.mean(all_recalls)
        global_precision = statistics.mean(all_precisions)
        lines.append("-" * 60)
        lines.append(f"{'GLOBAL':<30} | {global_recall:.4f}     | {global_precision:.4f}     | {len(all_recalls):<5}")
    lines.append("="*60 + "\n")

    report_text = "\n".join(lines)
    print(report_text)

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Reporte guardado en: {args.output_file}")

if __name__ == "__main__":
    main()