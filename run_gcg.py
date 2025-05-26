# Some useful libraries, feel free to import any others you need.
import os
import torch
import time
import json
import jailbreakbench as jbb

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from gcg.algorithm import GCGConfig, run

from config import NUM_SAMPLES, MODEL_NAME

# 1. Load the model and tokenizer

model_id = MODEL_NAME
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ...
tokenizer = ...


# 2. Load the harmful queries and target responses (this part is done for you)

queries = jbb.read_dataset().goals[:NUM_SAMPLES]
targets = jbb.read_dataset().targets[:NUM_SAMPLES]


# 3. Run GCG (this part is done for you)

config = GCGConfig(
    num_steps=250,           # You can try to adjust these to lower the runtime but be sure it doesn't hinder the attack success.
    search_width=512,
    topk=256,
    verbosity="WARNING",     # Set to "INFO" for more detailed output
    use_prefix_cache=False
)

suffixes = []
losses = []
start_time = time.time()
for query, target in tqdm(zip(queries, targets), total=len(queries), desc="Running GCG", unit="query"):
    result = run(model, tokenizer, query, target, config)
    suffixes.append(result.best_string)
    losses.append(result.best_loss)

print(f"Time taken: {time.time() - start_time} seconds")
print("Average loss: ", sum(losses) / len(losses))


# 4. Save the adversarial suffixes

os.makedirs("data", exist_ok=True)
with open("data/suffixes.json", "w", encoding="utf-8") as f:
    json.dump(suffixes, f, indent=4, ensure_ascii=False)

print("Suffixes saved to data/suffixes.json")