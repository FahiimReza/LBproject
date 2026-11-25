from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # or "Qwen/Qwen2.5-7B-Instruct"
TARGET_DIR = "checkpoints/local_llm"

snapshot_download(repo_id=MODEL_ID, local_dir=TARGET_DIR)
print(f"Downloaded {MODEL_ID} to {TARGET_DIR}")