import argparse
import json
import os
import torch
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generar respuestas usando Qwen basado en retrieval results")
    parser.add_argument("--results-dir", type=str, required=True, help="Carpeta con los JSONs de main.py")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="ID de HuggingFace o ruta a archivo .gguf")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--score-threshold", type=float, default=None, help="Filtrar contexto usando threshold dinámico (si no se aplicó antes)")
    args = parser.parse_args()

    use_gguf = args.model_name.endswith(".gguf")
    model = None
    tokenizer = None
    llm = None

    if use_gguf:
        print(f"🧠 Cargando modelo GGUF (llama.cpp): {args.model_name}...")
        try:
            from llama_cpp import Llama
            llm = Llama(
                model_path=args.model_name,
                n_ctx=8192, # Ajustar según capacidad de tu hardware
                n_gpu_layers=-1, 
                verbose=False
            )
        except ImportError:
            print("Error: Necesitas instalar llama-cpp-python para usar modelos .gguf")
            return
    else:
        print(f"🤗 Cargando modelo Hugging Face (transformers): {args.model_name}...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, 
                device_map="auto", 
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).eval()
        except Exception as e:
            print(f"Error cargando el modelo: {e}")
            return

    files = [f for f in os.listdir(args.results_dir) if f.endswith(".json")]
    print(f"Generando respuestas para {len(files)} preguntas en {args.results_dir}...")

    for filename in tqdm(files):
        filepath = os.path.join(args.results_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Si ya tiene respuesta, saltar
        if "generated_answer" in data:
            continue

        # Recuperar textos de sesión (ya filtrados por main.py si se usó threshold allí)
        session_texts = data.get("retrieved_session_texts", [])
        
        # Construir Prompt
        context_text = "\n\n--- Session Separator ---\n\n".join(session_texts)
        question = data["question"]
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question accurately."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nUser Question: {question}"}
        ]
        
        # Generación
        if use_gguf:
            output = llm.create_chat_completion(messages=messages, max_tokens=args.max_new_tokens, temperature=0.7)
            response = output['choices'][0]['message']['content']
        else:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        data["generated_answer"] = response
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()