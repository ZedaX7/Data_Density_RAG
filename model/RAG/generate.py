from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from accelerate import infer_auto_device_map
import torch
import gc
import os

# Clear all GPU memory
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
gc.collect()

# Set environment variables for multi-GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB")


# Available model configurations
MODEL_CONFIGS = {
    "llama3-8b": "/mnt/c/Users/kovac/Desktop/pre_trained/Llama-3-8B-Instruct",
    "llama31-8b": "/mnt/c/Users/kovac/Desktop/pre_trained/Llama-3.1-8B-Instruct",
    "llama32-3b": "/mnt/c/Users/kovac/Desktop/pre_trained/Llama-3.2-3B-Instruct",
    "llama33-70b": "/mnt/c/Users/kovac/Desktop/pre_trained/Llama-3.3-70B-Instruct",
    "llama4-17b": "/mnt/c/Users/kovac/Desktop/pre_trained/Llama-4-Scout-17B-16E-Instruct",
}

# Global variables for model and tokenizer (will be initialized when needed)
model = None
tokenizer = None
current_model_id = None

def initialize_model(model_key="llama33-70b"):
    """Initialize the model and tokenizer based on the model key."""
    global model, tokenizer, current_model_id

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model key: {model_key}. Available options: {list(MODEL_CONFIGS.keys())}")

    model_id = MODEL_CONFIGS[model_key]

    # Only reinitialize if different model is requested
    if current_model_id == model_id and model is not None:
        print(f"[INFO] Model {model_key} already loaded.")
        return

    # Clear existing model if any
    if model is not None:
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[INFO] Loading model: {model_key} from {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Quantization config using bitsandbytes
    # 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # 8-bit
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     bnb_8bit_compute_dtype=torch.float16,
    #     llm_int8_enable_fp32_cpu_offload=True # allows CPU fallback
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        max_memory={
            0: "24GiB",   # 4090
            1: "24GiB",   # 3090 Ti
            # 2: "24GiB",   # 3090
            # 3: "24GiB",   # 3090
        },
    )

    current_model_id = model_id
    print(f"[INFO] Model {model_key} loaded successfully.")

# Print device mapping to see how the model is distributed
# print("\nModel device mapping:")
# for name, param in model.named_parameters():
#     if hasattr(param, 'device'):
#         print(f"{name}: {param.device}")
#     if "layers.0" in name:  # Just show first few layers to avoid spam
#         break
    
    
def generate_answer(query, context_docs, model_key="llama33-70b"):
    # Ensure model is initialized
    if model is None:
        initialize_model(model_key)

    context = "\n".join(f"- {doc}" for doc in context_docs)
    prompt = (
        f"You are an expert battery pack designer.\n"
        f"You will be asked to provide battery pack design solutions for a user. \n"
        f"There is NO correct answer. Please provide the best feasible design solution.\n"
        f"Please answer with concise and complete sentences and with a polite and professional tone. \n"
        f"Include and Explicitly show cell_locations information, in the format of \"cell_locations: \". \n"
        f"Explicitly show series_count, and parallel_count information, in the format of \"series_count: \" and \"parallel_count: \. \n"
        f"Explicitly show voltage and capacity, in the format of \"design_voltage: \" and \"design_capacity: \". \n"
        f"Explicitly show width, depth, and height information, in the format of \"design_width: \", \"design_depth: \", and \"design_height: \". \n"
        f"Prioritize battery voltage and capacity, when facing conflicts in design requirement.\n"
        f"Query: {query}\n"
        f"Relevant past designs:\n{context}\n"
        f"Answer:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    outputs = model.generate(**inputs, 
                             max_new_tokens=1024, 
                             temperature=0.7, 
                             top_p=0.95)
    
    # Clean up
    del inputs
    torch.cuda.empty_cache()

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
