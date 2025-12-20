import argparse
import json
import os
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from src.datasets.LongMemEvalDataset import LongMemEvalDataset

def simple_tokenize(text):
    """Tokenización más robusta usando regex para separar palabras y eliminar puntuación."""
    # \w+ encuentra palabras alfanuméricas, eliminando comas, puntos, etc.
    return re.findall(r'\w+', text.lower())

def main():
    parser = argparse.ArgumentParser(description="BM25 Retrieval Baseline para LongMemEval")
    parser.add_argument("--output-dir", type=str, default="experiments/retrieval_results/bm25_top4", help="Directorio donde guardar los JSONs")
    parser.add_argument("--num-samples", type=int, default=500, help="Número de preguntas a procesar")
    parser.add_argument("--k", type=int, default=4, help="Top-K sesiones a recuperar")
    parser.add_argument("--dataset-type", type=str, default="short", choices=["oracle", "short"])
    parser.add_argument("--dataset-set", type=str, default="longmemeval")
    
    args = parser.parse_args()

    # 3. Cargar Dataset de Preguntas
    print(f"Cargando dataset {args.dataset_set}...")
    dataset = LongMemEvalDataset(args.dataset_type, args.dataset_set)

    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Loop de Retrieval
    print(f"Procesando {args.num_samples} preguntas...")
    
    count = 0
    # Iteramos sobre el dataset. Asumimos que dataset es iterable.
    for instance in tqdm(dataset):
        if count >= args.num_samples:
            break
            
        output_file = os.path.join(args.output_dir, f"{instance.question_id}.json")
        
        # --- PREPARACIÓN DEL CORPUS LOCAL (Por pregunta) ---
        # Asumimos que 'instance' tiene un atributo 'sessions' con el historial asociado
        candidate_sessions = getattr(instance, 'sessions', [])
        
        corpus_ids = []
        tokenized_corpus = []
        
        for session in candidate_sessions:
            # Manejo si session es dict
            if isinstance(session, dict):
                s_id = session.get('session_id')
                text_content = session.get('text', "")
                if not text_content and 'turns' in session:
                    text_content = " ".join([t['content'] for t in session['turns']])
            else:
                # Manejo si session es objeto
                s_id = getattr(session, 'session_id', str(session))
                text_content = getattr(session, 'text', "")
                
                # Soporte para atributo 'messages' (Clase Session)
                if not text_content and hasattr(session, 'messages'):
                    msgs = session.messages
                    if isinstance(msgs, list) and msgs:
                        if isinstance(msgs[0], dict):
                            text_content = " ".join([m.get('content', '') for m in msgs])
                        elif isinstance(msgs[0], str):
                            text_content = " ".join(msgs)
            
            tokenized_corpus.append(simple_tokenize(text_content))
            corpus_ids.append(str(s_id))

        if not tokenized_corpus:
            print(f"Skipping {instance.question_id}: Empty corpus")
            continue

        # Construir índice BM25 solo para esta pregunta
        bm25 = BM25Okapi(tokenized_corpus)

        query = instance.question
        tokenized_query = simple_tokenize(query)
        
        # Obtener scores de BM25
        scores = bm25.get_scores(tokenized_query)
        
        # Obtener los índices de los top-k scores
        actual_k = min(args.k, len(scores))
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_k]
        
        retrieved_sessions = [corpus_ids[i] for i in top_n_indices]
        retrieved_scores = [float(scores[i]) for i in top_n_indices]

        # 5. Guardar resultado
        result = {
            "question_id": instance.question_id,
            "question": instance.question,
            "question_type": instance.question_type,
            "retrieved_sessions": retrieved_sessions,
            "retrieved_scores": retrieved_scores,
            "method": "bm25",
            "k": args.k
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            
        count += 1

    print(f"Completado. Resultados guardados en: {args.output_dir}")

if __name__ == "__main__":
    main()
