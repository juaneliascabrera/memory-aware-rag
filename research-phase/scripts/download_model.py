from huggingface_hub import hf_hub_download
import os

def download_qwen():
    model_name = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    # Usamos la cuantización q4_k_m (aprox 2GB, balance ideal)
    filename = "qwen2.5-3b-instruct-q4_k_m.gguf" 
    
    print(f"Descargando {filename}...")
    model_path = hf_hub_download(
        repo_id=model_name,
        filename=filename,
        local_dir="models",
        local_dir_use_symlinks=False
    )
    print(f"✅ Modelo descargado en: {model_path}")

if __name__ == "__main__":
    download_qwen()