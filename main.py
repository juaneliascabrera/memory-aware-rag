import argparse
import json
import os
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from src.datasets.LongMemEvalDataset import LongMemEvalDataset
from sentence_transformers import CrossEncoder

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
    parser.add_argument("--granularity", type=str, default="session", choices=["session", "message"], help="Nivel de granularidad: 'session' concatena todo, 'message' indexa por turno")
    parser.add_argument("--dataset-set", type=str, default="longmemeval")
    parser.add_argument("--use-reranker", action="store_true", help="Activar re-ranking con Cross-Encoder")
    parser.add_argument("--reranker-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Modelo de Cross-Encoder a usar")
    parser.add_argument("--top-n", type=int, default=100, help="Número de candidatos iniciales para re-rankear")
    
    args = parser.parse_args()

    # 3. Cargar Dataset de Preguntas
    print(f"Cargando dataset {args.dataset_set}...")
    dataset = LongMemEvalDataset(args.dataset_type, args.dataset_set)

    # Inicializar Cross-Encoder si se solicita
    reranker = None
    if args.use_reranker:
        print(f"Cargando Cross-Encoder: {args.reranker_model}...")
        reranker = CrossEncoder(args.reranker_model)
        if args.granularity == "session":
            print("ADVERTENCIA: Usando Cross-Encoder con granularidad 'session'.")
            print("  Las sesiones largas (>512 tokens) serán truncadas, lo que puede afectar el rendimiento.")

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
        corpus_texts = []
        
        for session in candidate_sessions:
            chunks = []
            # Manejo si session es dict
            if isinstance(session, dict):
                s_id = session.get('session_id')
                if session.get('text'):
                    chunks = [session['text']]
                elif 'turns' in session:
                    chunks = [t['content'] for t in session['turns']]
            else:
                # Manejo si session es objeto
                s_id = getattr(session, 'session_id', str(session))
                if hasattr(session, 'text') and session.text:
                    chunks = [session.text]
                elif hasattr(session, 'messages'):
                    msgs = session.messages
                    if isinstance(msgs, list) and msgs:
                        if isinstance(msgs[0], dict):
                            chunks = [m.get('content', '') for m in msgs]
                        elif isinstance(msgs[0], str):
                            chunks = msgs
            
            for chunk in (chunks if args.granularity == 'message' else [" ".join(chunks)]):
                tokenized_corpus.append(simple_tokenize(chunk))
                corpus_ids.append(str(s_id))
                corpus_texts.append(chunk)

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
        if args.use_reranker:
            # 1. Fase de Candidatos: Traemos Top-N con BM25
            search_limit = min(len(scores), args.top_n)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:search_limit]
            
            # 2. Fase de Re-ranking: Cross-Encoder
            # Preparamos pares [Query, Documento]
            candidate_pairs = [[query, corpus_texts[i]] for i in top_indices]
            ce_scores = reranker.predict(candidate_pairs)
            
            # Ordenamos los candidatos basados en el score del Cross-Encoder
            # ranked_candidates es una lista de tuplas (indice_original, score_ce)
            ranked_candidates = sorted(zip(top_indices, ce_scores), key=lambda x: x[1], reverse=True)
        else:
            # Lógica original sin re-ranking
            search_limit = len(scores) if args.granularity == 'message' else args.k
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:search_limit]
            ranked_candidates = [(i, scores[i]) for i in top_indices]
        
        retrieved_sessions = []
        retrieved_scores = []
        seen_sessions = set()

        for i, score in ranked_candidates:
            sess_id = corpus_ids[i]
            if sess_id not in seen_sessions:
                seen_sessions.add(sess_id)
                retrieved_sessions.append(sess_id)
                retrieved_scores.append(float(score))
                if len(retrieved_sessions) >= args.k:
                    break

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
