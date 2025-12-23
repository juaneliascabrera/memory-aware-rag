import argparse
import json
import os
import statistics
from src.datasets.LongMemEvalDataset import LongMemEvalDataset

def run_evaluation(results_dir, dataset_set="longmemeval", k=5, output_file=None, score_threshold=None, quiet=False):
    # 1. Cargar Ground Truth (Oracle)
    if not quiet:
        print(f"Cargando Ground Truth (Oracle) para '{dataset_set}'...")
    try:
        # Instanciamos el dataset en modo 'oracle'. 
        # En este modo, la lista 'sessions' de cada instancia contiene SOLO las sesiones relevantes.
        oracle_dataset = LongMemEvalDataset(type="oracle", set=dataset_set)
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

    if not quiet:
        print(f"Ground Truth cargado: {len(ground_truth_map)} preguntas indexadas.")

    # 2. Leer Resultados Generados
    if not os.path.exists(results_dir):
        print(f"No se encuentra el directorio de resultados: {results_dir}")
        return

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    if not quiet:
        print(f"Evaluando {len(result_files)} archivos de resultados en '{results_dir}'...")

    metrics_by_type = {}  # { "type_name": {"recalls": [], "precisions": []} }
    missing_in_oracle = 0

    for filename in result_files:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        
        q_id = result_data.get("question_id")
        
        # Recuperamos la lista de sesiones que trajo nuestro algoritmo (BM25)
        retrieved_ids = [str(x) for x in result_data.get("retrieved_sessions", [])]
        retrieved_scores = result_data.get("retrieved_scores", [])

        # --- LÓGICA DE THRESHOLD DINÁMICO (POST-PROCESAMIENTO) ---
        if score_threshold is not None and retrieved_scores:
            filtered_ids = []
            best_score = retrieved_scores[0] # Asumimos que están ordenados descendente
            
            for i, sess_id in enumerate(retrieved_ids):
                # Verificamos que exista score correspondiente (por seguridad)
                if i < len(retrieved_scores):
                    if (best_score - retrieved_scores[i]) > score_threshold:
                        break # Cortamos la lista aquí
                filtered_ids.append(sess_id)
            retrieved_ids = filtered_ids
        
        # Buscamos las relevantes en el mapa
        if q_id not in ground_truth_map:
            # Esto puede pasar si el dataset oracle tiene menos preguntas o IDs distintos
            missing_in_oracle += 1
            continue
            
        gt_data = ground_truth_map[q_id]
        relevant_ids = gt_data["ids"]
        q_type = gt_data["type"]
        
        # Cortamos en K (por si el archivo tiene más resultados guardados)
        current_k_retrieved = retrieved_ids[:k]
        
        # 3. Calcular Métricas
        # Intersección: Cuántas de las recuperadas están en las relevantes
        hits = len(set(current_k_retrieved).intersection(relevant_ids))
        
        # Recall: Hits / Total Relevantes
        recall = hits / len(relevant_ids) if len(relevant_ids) > 0 else 0.0
        
        # Precision: Hits / Cantidad Recuperada Real (ajustado para Threshold Dinámico)
        num_retrieved = len(current_k_retrieved)
        precision = hits / num_retrieved if num_retrieved > 0 else 0.0
        
        # MRR: Solo calculamos si hay exactamente 1 sesión relevante (Single Session)
        rr = None
        if len(relevant_ids) == 1:
            rr = 0.0
            for i, doc_id in enumerate(current_k_retrieved):
                if doc_id in relevant_ids:
                    rr = 1.0 / (i + 1)
                    break
        
        if q_type not in metrics_by_type:
            metrics_by_type[q_type] = {"recalls": [], "precisions": [], "rrs": []}
        
        metrics_by_type[q_type]["recalls"].append(recall)
        metrics_by_type[q_type]["precisions"].append(precision)
        if rr is not None:
            metrics_by_type[q_type]["rrs"].append(rr)

    # 4. Reporte Final
    lines = []
    lines.append("\n" + "="*75)
    lines.append(f"RESULTADOS DE EVALUACIÓN POR TIPO (K={k})")
    lines.append("="*75)
    lines.append(f"{'TIPO':<30} | {'RECALL':<10} | {'PRECISION':<10} | {'MRR':<10} | {'COUNT':<5}")
    lines.append("-" * 75)

    all_recalls = []
    all_precisions = []
    all_rrs = []

    global_recall = 0.0
    global_precision = 0.0
    global_mrr = 0.0

    for q_type, data in sorted(metrics_by_type.items()):
        avg_r = statistics.mean(data["recalls"])
        avg_p = statistics.mean(data["precisions"])
        
        if data["rrs"]:
            avg_mrr = statistics.mean(data["rrs"])
            mrr_str = f"{avg_mrr:.4f}"
            all_rrs.extend(data["rrs"])
        else:
            mrr_str = "N/A"
            
        count = len(data["recalls"])
        
        all_recalls.extend(data["recalls"])
        all_precisions.extend(data["precisions"])
        
        lines.append(f"{q_type:<30} | {avg_r:.4f}     | {avg_p:.4f}     | {mrr_str:<10} | {count:<5}")

    if all_recalls:
        global_recall = statistics.mean(all_recalls)
        global_precision = statistics.mean(all_precisions)
        
        if all_rrs:
            global_mrr = statistics.mean(all_rrs)
            global_mrr_str = f"{global_mrr:.4f}"
        else:
            global_mrr_str = "N/A"
            
        lines.append("-" * 75)
        lines.append(f"{'GLOBAL':<30} | {global_recall:.4f}     | {global_precision:.4f}     | {global_mrr_str:<10} | {len(all_recalls):<5}")
    lines.append("="*75 + "\n")

    report_text = "\n".join(lines)
    
    if not quiet:
        print(report_text)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        if not quiet:
            print(f"Reporte guardado en: {output_file}")
            
    return {
        "recall": global_recall,
        "precision": global_precision,
        "mrr": global_mrr
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluar métricas de Retrieval (Recall/Precision)")
    parser.add_argument("--results-dir", type=str, required=True, help="Carpeta con los JSONs generados por main.py")
    parser.add_argument("--dataset-set", type=str, default="longmemeval", help="El set usado (longmemeval, investigathon_evaluation)")
    parser.add_argument("--k", type=int, default=5, help="El corte K para las métricas")
    parser.add_argument("--output-file", type=str, help="Ruta del archivo para guardar el reporte (opcional)")
    parser.add_argument("--score-threshold", type=float, default=None, help="Aplicar threshold dinámico a los resultados existentes")
    
    args = parser.parse_args()
    run_evaluation(args.results_dir, args.dataset_set, args.k, args.output_file, args.score_threshold)

if __name__ == "__main__":
    main()