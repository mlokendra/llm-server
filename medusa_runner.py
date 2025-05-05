from llama_cpp import Llama
import torch
import random


model_path="/home/codespace/.cache/huggingface/hub/models--TheBloke--vicuna-7B-v1.5-GGUF/snapshots/8b4a138d6ba32660c42b5df6dad7ad5c23b80c8c/vicuna-7b-v1.5.Q2_K.gguf"
# Initialize the quantized Vicuna model
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    n_batch=8,
    use_mlock=True
)

# Simulated Medusa speculative decoder
def medusa_generate(prompts, max_tokens=100, medusa_top_k=3):
    results = []
    
    for prompt in prompts:
        # Initial prompt embedding
        output = llm(prompt, max_tokens=1, stop=["\n"], echo=True)
        prompt_tokens = output["tokens"]
        generated = prompt

        for _ in range(max_tokens):
            # Step 1: Medusa head proposes k tokens
            candidates = medusa_propose(prompt_tokens, k=medusa_top_k)

            # Step 2: Validate via original model
            for candidate in candidates:
                test_output = llm(generated + candidate, max_tokens=1, stop=["\n"])
                if test_output["choices"][0]["text"].startswith(candidate):
                    generated += candidate
                    break
            else:
                break  # None of the Medusa proposals were valid

        results.append(generated[len(prompt):])  # Strip original prompt

    return results


def medusa_propose(tokens, k=3):
    """Mock function to simulate speculative decoding proposals"""
    # This is a stub — you'd use Medusa logits or a head for real proposal.
    # Here, we simulate with random strings.
    dummy_tokens = [" the", " of", " and", " is", " to", " cat", " dog", " moon"]
    return random.sample(dummy_tokens, k)
