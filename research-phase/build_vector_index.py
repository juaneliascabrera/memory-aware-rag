import argparse
import json
import os
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.datasets.LongMemEvalDataset import LongMemEvalDataset

def main():
    parser = argparse.ArgumentParser(description="Pre-calcular embeddings para LongMemEval")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Modelo de HF")
    parser.add_argument("--output-dir", type=str, default="data/embeddings", help="Dónde guardar los vectores")
    parser.add_argument("--dataset-set", type=str, default="longmemeval")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño del lote para inferencia")
    args = parser.parse_args()

    # 1. Cargar Modelo
    print(f"Cargando modelo de embeddings: {args.model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model_name, device=device)

    # 2. Cargar Dataset (Corpus)
    # Usamos 'short' porque contiene el historial completo (candidatos)
    print(f"Cargando corpus del dataset '{args.dataset_set}' (short)...")
    dataset = LongMemEvalDataset("short", args.dataset_set)

    # 3. Extraer Textos Únicos
    # Necesitamos mapear: ID Único -> Texto
    # Dado que usamos granularidad 'message' en el experimento exitoso, indexaremos mensajes.
    # Para ahorrar espacio, si un mensaje se repite exactamente en varias sesiones (raro pero posible), lo guardamos una vez.
    
    unique_texts = []
    text_to_id_map = [] # Lista paralela para saber a qué pertenece cada vector
    
    print("Extrayendo mensajes del corpus...")
    # Usamos un set para evitar procesar la misma sesión múltiples veces si aparece en varias instancias
    processed_session_ids = set()

    for instance in tqdm(dataset):
        candidate_sessions = getattr(instance, 'sessions', [])
        
        for session in candidate_sessions:
            # Obtener ID de sesión y Timestamp
            session_timestamp = ""
            if isinstance(session, dict):
                s_id = session.get('session_id')
                session_timestamp = session.get('timestamp', '') or session.get('date', '')
                msgs = session.get('turns', []) if 'turns' in session else ([session['text']] if 'text' in session else [])
            else:
                s_id = getattr(session, 'session_id', str(session))
                session_timestamp = getattr(session, 'timestamp', '') or getattr(session, 'date', '')
                if hasattr(session, 'messages'):
                    msgs = session.messages
                elif hasattr(session, 'text'):
                    msgs = [session.text]
                else:
                    msgs = []

            s_id_str = str(s_id)
            if s_id_str in processed_session_ids:
                continue
            
            processed_session_ids.add(s_id_str)

            # Procesar mensajes de la sesión
            # Guardamos: (Texto del mensaje, ID de sesión al que pertenece)
            # Nota: Si quisieras granularidad mensaje pura con ID de mensaje, necesitarías un ID único por mensaje.
            # Aquí asociaremos el vector al Session ID. Si una sesión tiene 10 mensajes, tendrá 10 vectores apuntando al mismo Session ID.
            
            if isinstance(msgs, list):
                for m in msgs:
                    content = ""
                    role = ""
                    
                    if isinstance(m, dict):
                        content = m.get('content', '')
                        role = m.get('role', '')
                    elif isinstance(m, str):
                        content = m
                    else:
                        # Intento genérico para objetos
                        content = getattr(m, 'content', '')
                        role = getattr(m, 'role', '')
                    
                    if content.strip():
                        # Construimos el texto enriquecido: "[Timestamp] Role: Content"
                        parts = []
                        if session_timestamp:
                            parts.append(f"[{session_timestamp}]")
                        if role:
                            parts.append(f"{role}:")
                        parts.append(content)
                        
                        enriched_text = " ".join(parts)
                        
                        unique_texts.append(enriched_text)
                        text_to_id_map.append(s_id_str)

    print(f"Total de mensajes a indexar: {len(unique_texts)}")

    # 4. Generar Embeddings
    print("Generando embeddings (esto puede tardar)...")
    embeddings = model.encode(
        unique_texts, 
        batch_size=args.batch_size, 
        show_progress_bar=True, 
        convert_to_tensor=True,
        normalize_embeddings=True # Importante para usar producto punto como similitud coseno
    )

    # 5. Guardar en Disco
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Guardamos los tensores
    embeddings_path = os.path.join(args.output_dir, "corpus_embeddings.pt")
    torch.save(embeddings, embeddings_path)
    
    # Guardamos el mapeo (índice -> session_id) y los textos (opcional, para debug)
    mapping_path = os.path.join(args.output_dir, "corpus_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(text_to_id_map, f)
        
    print(f"✅ Indexación completada.")
    print(f"Embeddings guardados en: {embeddings_path}")
    print(f"Mapping guardado en: {mapping_path}")

if __name__ == "__main__":
    main()
