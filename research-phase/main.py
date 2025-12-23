import argparse
import json
import os
import re
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from src.datasets.LongMemEvalDataset import LongMemEvalDataset
from sentence_transformers import CrossEncoder, SentenceTransformer

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
    parser.add_argument("--score-threshold", type=float, default=None, help="Diferencia máxima de score permitida respecto al mejor resultado para seguir agregando sesiones.")
    parser.add_argument("--use-embeddings", action="store_true", help="Activar búsqueda híbrida con embeddings")
    parser.add_argument("--embeddings-dir", type=str, default="data/embeddings", help="Directorio con corpus_embeddings.pt y corpus_mapping.json")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Modelo para codificar la query")
    
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

    # Inicializar Embeddings si se solicita
    global_embeddings = None
    session_to_indices = {}
    embedding_model = None

    if args.use_embeddings:
        print(f"Cargando embeddings desde {args.embeddings_dir}...")
        emb_path = os.path.join(args.embeddings_dir, "corpus_embeddings.pt")
        map_path = os.path.join(args.embeddings_dir, "corpus_mapping.json")
        
        if os.path.exists(emb_path) and os.path.exists(map_path):
            global_embeddings = torch.load(emb_path, map_location="cpu")
            with open(map_path, "r") as f:
                global_mapping = json.load(f)
            
            # Construir índice inverso: session_id -> lista de índices globales
            for idx, s_id in enumerate(global_mapping):
                if s_id not in session_to_indices:
                    session_to_indices[s_id] = []
                session_to_indices[s_id].append(idx)
            
            print(f"Cargando modelo de embeddings para query: {args.embedding_model}...")
            embedding_model = SentenceTransformer(args.embedding_model)
        else:
            print(f"ERROR: No se encontraron archivos en {args.embeddings_dir}. Ejecuta build_vector_index.py primero.")
            return

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
        session_id_to_text = {}
        local_embedding_indices = [] # Para mapear mensaje local -> vector global
        
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
            
            # Guardamos el texto completo de la sesión para referencia futura (Prompt)
            session_id_to_text[str(s_id)] = " ".join(chunks)

            # Recuperamos los índices globales de embeddings para esta sesión
            # Asumimos que el orden de los mensajes en 'chunks' coincide con el orden indexado en build_vector_index.py
            # Esto es cierto si ambos iteran la lista de mensajes en orden.
            g_indices = session_to_indices.get(str(s_id), [])
            g_idx_counter = 0

            for chunk in (chunks if args.granularity == 'message' else [" ".join(chunks)]):
                # Importante: build_vector_index salta mensajes vacíos (content.strip()). Debemos hacer lo mismo para mantener sincronía.
                if not chunk.strip():
                    continue
                    
                tokenized_corpus.append(simple_tokenize(chunk))
                corpus_ids.append(str(s_id))
                corpus_texts.append(chunk)
                
                if args.use_embeddings and g_idx_counter < len(g_indices):
                    local_embedding_indices.append(g_indices[g_idx_counter])
                    g_idx_counter += 1

        if not tokenized_corpus:
            print(f"Skipping {instance.question_id}: Empty corpus")
            continue

        # Construir índice BM25 solo para esta pregunta
        bm25 = BM25Okapi(tokenized_corpus)

        query = instance.question
        tokenized_query = simple_tokenize(query)
        
        # Obtener scores de BM25
        scores = bm25.get_scores(tokenized_query)
        final_scores = scores # Por defecto solo BM25
        
        # --- BÚSQUEDA HÍBRIDA (BM25 + VECTORES) ---
        if args.use_embeddings and local_embedding_indices:
            # 1. Calcular Scores Vectoriales
            # Codificamos la query
            q_emb = embedding_model.encode(query, convert_to_tensor=True)
            
            # Extraemos los vectores relevantes del tensor global
            # local_embedding_indices tiene el índice global de cada mensaje en corpus_texts
            if len(local_embedding_indices) == len(corpus_texts):
                relevant_embeddings = global_embeddings[local_embedding_indices]
                
                # Asegurar que ambos tensores estén en el mismo dispositivo (para evitar error GPU vs CPU)
                if q_emb.device != relevant_embeddings.device:
                    relevant_embeddings = relevant_embeddings.to(q_emb.device)

                # Similitud Coseno
                vector_scores = torch.nn.functional.cosine_similarity(q_emb, relevant_embeddings, dim=-1).cpu().numpy()
                
                # 2. Fusión RRF (Reciprocal Rank Fusion)
                # RRF score = 1 / (k + rank_bm25) + 1 / (k + rank_vector)
                k_rrf = 60
                
                # Obtenemos los rankings (índices ordenados de mayor score a menor)
                bm25_ranks = np.argsort(scores)[::-1]
                vector_ranks = np.argsort(vector_scores)[::-1]
                
                # Mapeamos índice_documento -> ranking (0-based)
                bm25_rank_map = {idx: r for r, idx in enumerate(bm25_ranks)}
                vector_rank_map = {idx: r for r, idx in enumerate(vector_ranks)}
                
                rrf_scores = []
                for i in range(len(corpus_texts)):
                    r_bm25 = bm25_rank_map.get(i, 99999)
                    r_vec = vector_rank_map.get(i, 99999)
                    score = (1.0 / (k_rrf + r_bm25 + 1)) + (1.0 / (k_rrf + r_vec + 1))
                    rrf_scores.append(score)
                
                final_scores = np.array(rrf_scores)
            else:
                # Si hay desajuste de índices (raro), fallamos a BM25
                print(f"Warning: Mismatch indices for {instance.question_id}. Fallback to BM25.")
        
        # Obtener los índices de los top-k scores
        if args.use_reranker:
            # 1. Fase de Candidatos: Traemos Top-N con Hybrid (o BM25)
            search_limit = min(len(final_scores), args.top_n)
            top_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)[:search_limit]
            
            # 2. Fase de Re-ranking: Cross-Encoder
            # Preparamos pares [Query, Documento]
            candidate_pairs = [[query, corpus_texts[i]] for i in top_indices]
            ce_scores = reranker.predict(candidate_pairs)
            
            # Ordenamos los candidatos basados en el score del Cross-Encoder
            # ranked_candidates es una lista de tuplas (indice_original, score_ce)
            ranked_candidates = sorted(zip(top_indices, ce_scores), key=lambda x: x[1], reverse=True)
        else:
            # Si no se usa re-ranker, no hay candidatos intermedios
            top_indices = [] 
            ranked_candidates = []
            bm25_candidates = None

            # Lógica original sin re-ranking
            search_limit = len(final_scores) if args.granularity == 'message' else args.k
            top_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)[:search_limit]
            ranked_candidates = [(i, final_scores[i]) for i in top_indices]
        
        retrieved_sessions = []
        retrieved_scores = []
        seen_sessions = set()
        best_score = None

        for i, score in ranked_candidates:
            sess_id = corpus_ids[i]
            if sess_id not in seen_sessions:
                # Lógica de Threshold Dinámico
                if args.score_threshold is not None:
                    if best_score is None:
                        best_score = score
                    elif (best_score - score) > args.score_threshold:
                        # Si la diferencia con el mejor es muy grande, cortamos
                        break

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
            "retrieved_session_texts": [session_id_to_text.get(sid, "") for sid in retrieved_sessions],
            "parameters": {
                "k": args.k,
                "granularity": args.granularity,
                "use_reranker": args.use_reranker,
                "reranker_model": args.reranker_model if args.use_reranker else None,
                "top_n": args.top_n if args.use_reranker else None,
            },
            "method": ("hybrid" if args.use_embeddings else "bm25") + ("+rerank" if args.use_reranker else ""),
        }

        if args.use_reranker:
            # Guardamos los candidatos que BM25 le pasó al re-ranker para análisis
            result["candidate_ids"] = [corpus_ids[i] for i in top_indices]
            result["candidate_scores"] = [float(final_scores[i]) for i in top_indices]
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            
        count += 1

    print(f"Completado. Resultados guardados en: {args.output_dir}")

if __name__ == "__main__":
    main()
