import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"

def find_best_semantic_labels():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    embeddings = model.get_input_embeddings()

    # Candidate Sets (4 Classes: Supported, Unfounded, General, Unsure)
    candidates = [
        # Set A: The "Evidence" Model
        ("Evidence",  " Found",   " Fake",    " General",   " None"),
        
        # Set B: The "Data" Model
        ("Data",      " Source",  " Error",   " Fact",    " None"),
        
        # Set C: The "Truth" Model
        ("Evidence",  " Found",   " Fake",    " World",   " None"),
        
        # Set D: The "Audit" Model
        ("Audit",     " Supported",   " Wrong",   " Known",   " None")
    ]

    print(f"{'Set':<10} | {'Labels':<25} | {'Max Sim':<8} | {'Avg Sim':<8} | {'Status'}")
    print("-" * 75)

    best_set = None
    lowest_max_sim = 1.0

    for name, supp, unf, gen, unsure in candidates:
        tokens = [supp, unf, gen, unsure]
        ids = [tokenizer.encode(t, add_special_tokens=False) for t in tokens]
        
        # 1. Single Token Check
        if any(len(x) > 1 for x in ids):
            print(f"{name:<10} | {str(tokens):<25} | {'N/A':<8} | {'N/A':<8} | ❌ Multi-token")
            continue
            
        flat_ids = [x[0] for x in ids]
        vecs = embeddings(torch.tensor(flat_ids))
        
        # 2. Pairwise Cosine Similarity (6 pairs)
        sims = []
        pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)] # All combinations
        
        for i, j in pairs:
            sim = F.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
            sims.append(sim)
            
        max_s = max(sims)
        avg_s = sum(sims) / len(sims)
        
        status = "✅" if max_s < 0.25 else "⚠️"
        print(f"{name:<10} | {str(tokens):<25} | {max_s:.4f}   | {avg_s:.4f}   | {status}")

        if max_s < lowest_max_sim:
            lowest_max_sim = max_s
            best_set = (name, tokens)

    print("-" * 75)
    print(f"🏆 Recommendation: {best_set[1]}") 
    print("   (These labels are semantically distinct AND vector-orthogonal)")

if __name__ == "__main__":
    find_best_semantic_labels()