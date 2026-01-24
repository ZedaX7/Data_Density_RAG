from huggingface_hub import snapshot_download

local_dir = "./pre_trained/Llama-3.3-70B-Instruct"  # or any full path you want
snapshot_download(
    # repo_id="meta-llama/Meta-Llama-3-8B-Instruct", 
    # repo_id="meta-llama/Llama-3.1-8B-Instruct", 
    # repo_id="meta-llama/Llama-3.3-70B-Instruct",
    # repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    # repo_id="RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",
    # repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # download actual files, not symlinks
    resume_download=True  # resume if interrupted
)
