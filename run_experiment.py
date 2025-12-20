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
        # Si tu main.py usa otro flag para el dataset, cámbialo aquí
        # "--dataset", args.dataset 
    ]
    
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