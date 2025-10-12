from huggingface_hub import snapshot_download

# Download the entire repo to a local folder
local_dir = snapshot_download(repo_id="arnir0/Tiny-LLM")

print("Model downloaded to:", local_dir)
