import argparse
import subprocess
import os
import sys
from evaluate_metrics import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="Correr experimento BM25 completo (Retrieval + Eval)")
    parser.add_argument("--k", type=int, required=True, help="Valor de K para retrieval y evaluación")
    parser.add_argument("--experiment-name", type=str, required=True, help="Nombre del experimento (se usará para la carpeta de salida)")
    parser.add_argument("--dataset", type=str, default="longmemeval", help="Dataset a utilizar")
    parser.add_argument("--granularity", type=str, default="session", choices=["session", "message"], help="Granularidad: session o message")
    parser.add_argument("--use-reranker", action="store_true", help="Activar re-ranking")
    parser.add_argument("--top-n", type=int, default=100, help="Candidatos para re-ranking")
    
    args = parser.parse_args()
    
    # Definir rutas
    # Los resultados se guardarán en experiments/<experiment_name>
    base_output_dir = "experiments"
    experiment_dir = os.path.join(base_output_dir, args.experiment_name)
    
    questions_dir = os.path.join(experiment_dir, "questions")
    report_dir = os.path.join(experiment_dir, "report")
    report_file = os.path.join(report_dir, f"report_k{args.k}.txt")
    
    print(f"=== Iniciando Experimento: {args.experiment_name} (K={args.k}) ===")
    
    # 1. Ejecutar Retrieval (main.py)
    # Asumimos que main.py acepta --output-dir, --k y --dataset. 
    # Ajusta estos argumentos según tu implementación real de main.py.
    cmd = [
        sys.executable, "main.py",
        "--output-dir", questions_dir,
        "--k", str(args.k),
        "--granularity", args.granularity,
        "--top-n", str(args.top_n),
        # Si tu main.py usa otro flag para el dataset, cámbialo aquí
        # "--dataset", args.dataset 
    ]
    
    if args.use_reranker:
        cmd.append("--use-reranker")
    
    print(f"Ejecutando: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar main.py: {e}")
        return

    # 2. Ejecutar Evaluación
    print("\n=== Ejecutando Evaluación ===")
    run_evaluation(
        results_dir=questions_dir,
        dataset_set=args.dataset,
        k=args.k,
        output_file=report_file
    )
    
    print(f"\nExperimento completado. Reporte guardado en: {report_file}")

if __name__ == "__main__":
    main()