from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from accelerate import infer_auto_device_map
import torch
import gc
import os
import outlines
from schemas import BatteryDesign

# Clear all GPU memory
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
gc.collect()

# Set environment variables for multi-GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
outlines_model = None
structured_generator = None

def initialize_model(model_key="llama33-70b"):
    """Initialize the model and tokenizer based on the model key."""
    global model, tokenizer, current_model_id, outlines_model, structured_generator

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
        del outlines_model
        del structured_generator
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
            # 1: "24GiB",   # 3090 Ti
            # 2: "24GiB",   # 3090
            # 3: "24GiB",   # 3090
        },
    )

    current_model_id = model_id
    print(f"[INFO] Model {model_key} loaded successfully.")

    # Initialize outlines model wrapper for structured generation
    print(f"[INFO] Initializing structured generation with Outlines...")
    outlines_model = outlines.models.Transformers(model, tokenizer)
    structured_generator = outlines.generate.json(outlines_model, BatteryDesign)
    print(f"[INFO] Structured generation initialized.")
    
    
def generate_answer(query, context_docs, model_key="llama33-70b") -> BatteryDesign:
    """
    Generate a structured battery pack design using Outlines constrained generation.

    Args:
        query: User's design requirements
        context_docs: Retrieved relevant designs for context
        model_key: Which model to use

    Returns:
        BatteryDesign: Structured Pydantic object with design details
    """
    # Ensure model is initialized
    if model is None or structured_generator is None:
        initialize_model(model_key)

    context = "\n".join(f"- {doc}" for doc in context_docs)

    # Prompt optimized for JSON output
    prompt = f"""You are an expert battery pack designer. Design a battery pack based on the requirements.

        Requirements: {query}

        Reference designs from database:
        {context}

        Based on these requirements and references, provide a battery pack design as JSON with these fields:
        - series_count: number of cells in series (determines voltage)
        - parallel_count: number of cells in parallel (determines capacity)
        - design_voltage: total pack voltage in V (series_count × 3.7V)
        - design_capacity: total capacity in Ah (parallel_count × 2.5Ah)
        - design_width: pack width in mm
        - design_depth: pack depth in mm
        - design_height: pack height in mm
        - cell_locations: list of [x, y, z] coordinates for each cell
        - explanation: brief explanation of design choices

        Cell specifications (18650): 3.7V nominal, 2.5Ah capacity, 18mm diameter, 65mm length.
        Ensure dimensions fit all cells with 2mm spacing between cells and 5mm safety margin.

        JSON output:"""

    # Use structured generator for guaranteed valid JSON
    result = structured_generator(prompt)

    torch.cuda.empty_cache()
    return result


def generate_answer_legacy(query, context_docs, model_key="llama33-70b"):
    """
    Legacy text-based generation (kept for comparison/fallback).
    Returns raw text output that requires regex parsing.
    """
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
