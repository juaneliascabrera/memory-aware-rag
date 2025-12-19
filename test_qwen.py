from llama_cpp import Llama

def test_model():
    print("🧠 Cargando Qwen 2.5 3B...")
    llm = Llama(
        model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf",
        n_ctx=8192,
        n_gpu_layers=0, # 0 = CPU. Si lograste instalar con GPU, pon -1
        verbose=False
    )

    prompt = "Hola Qwen, responde brevemente: ¿Qué es la recuperación semántica (RAG)?"
    
    # Formato de chat para Qwen
    messages = [
        {"role": "system", "content": "Eres un asistente técnico experto."},
        {"role": "user", "content": prompt}
    ]

    print("💬 Generando respuesta...")
    output = llm.create_chat_completion(messages=messages, temperature=0.7)
    
    print("\nRespuesta del Modelo:")
    print("="*40)
    print(output['choices'][0]['message']['content'])
    print("="*40)

if __name__ == "__main__":
    test_model()