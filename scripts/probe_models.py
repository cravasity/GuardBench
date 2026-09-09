import os, json, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))

CANDIDATES = [
 # --- Meta Llama Guard line ---
 "meta-llama/LlamaGuard-7b","meta-llama/Meta-Llama-Guard-2-8B","meta-llama/Llama-Guard-3-8B",
 "meta-llama/Llama-Guard-3-1B","meta-llama/Llama-Guard-4-12B","meta-llama/Llama-Guard-3-11B-Vision",
 "meta-llama/Llama-Prompt-Guard-2-86M","meta-llama/Llama-Prompt-Guard-2-22M",
 # --- Google ---
 "google/shieldgemma-2b","google/shieldgemma-9b","google/shieldgemma-27b","google/shieldgemma-2",
 # --- AI2 ---
 "allenai/wildguard",
 # --- Qwen ---
 "Qwen/Qwen3Guard-Gen-0.6B","Qwen/Qwen3Guard-Gen-4B","Qwen/Qwen3Guard-Gen-8B",
 "Qwen/Qwen3Guard-Stream-0.6B","Qwen/Qwen3Guard-Stream-4B","Qwen/Qwen3Guard-Stream-8B",
 # --- IBM ---
 "ibm-granite/granite-guardian-3.0-2b","ibm-granite/granite-guardian-3.0-8b",
 "ibm-granite/granite-guardian-3.1-2b","ibm-granite/granite-guardian-3.2-5b",
 "ibm-granite/granite-guardian-3.3-8b","ibm-granite/granite-guardian-4.1-8b",
 "ibm-granite/granite-guardian-hap-125m",
 # --- NVIDIA ---
 "nvidia/llama-3.1-nemoguard-8b-content-safety","nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3",
 "nvidia/Nemotron-3.5-Content-Safety","nvidia/NemoGuard-JailbreakDetect",
 "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
 # --- OpenAI ---
 "openai/gpt-oss-safeguard-20b","openai/gpt-oss-safeguard-120b",
 # --- Mistral ---
 "mistralai/Shieldstral-1.0-3B",
 # --- Judges / classifiers ---
 "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b","OpenSafetyLab/MD-Judge-v0.1",
 "cais/HarmBench-Llama-2-13b-cls","cais/HarmBench-Mistral-7b-val-cls",
 # --- Reasoning / dynamic guards ---
 "yueliu1999/GuardReasoner-1B","yueliu1999/GuardReasoner-3B","yueliu1999/GuardReasoner-8B",
 "tomg-group-umd/DynaGuard-1.7B","tomg-group-umd/DynaGuard-4B","tomg-group-umd/DynaGuard-8B",
 # --- Multilingual ---
 "ToxicityPrompts/PolyGuard-Qwen","ToxicityPrompts/PolyGuard-Ministral","ToxicityPrompts/PolyGuard-Qwen-Smol",
 "saillab/x-guard",
 # --- Small / distilled ---
 "DuoGuard/DuoGuard-0.5B","DuoGuard/DuoGuard-1B-Llama-3.2-transfer","DuoGuard/DuoGuard-1.5B-transfer",
 "hbseong/HarmAug-Guard","unitary/toxic-bert","unitary/unbiased-toxic-roberta",
 "protectai/deberta-v3-base-prompt-injection-v2",
 "ServiceNow-AI/AprielGuard",
 # --- Others in survey ---
 "PKU-Alignment/beaver-dam-7b","thu-coai/ShieldLM-7B-internlm2",
 "LibrAI/longformer-harmful-ro","LibrAI/longformer-action-ro",
 "SamsungSDS-Research/SGuard-ContentFilter-2B-v1","kakaocorp/kanana-safeguard-8b",
 "Salesforce/BingoGuard-8B","Salesforce/BingoGuard-Llama-3.2-1B",
]

out=[]
for m in CANDIDATES:
    rec={"repo":m}
    try:
        i=api.model_info(m, files_metadata=False)
        rec.update(dict(exists=True, gated=str(i.gated), created_at=str(i.created_at),
                        last_modified=str(i.last_modified), downloads=i.downloads,
                        likes=i.likes, tags=[t for t in (i.tags or []) if not t.startswith(("dataset:","arxiv:","region:","base_model"))][:12],
                        arxiv=[t for t in (i.tags or []) if t.startswith("arxiv:")],
                        license=(i.card_data.get("license") if i.card_data else None),
                        pipeline=i.pipeline_tag))
        # test actual file access
        try:
            api.hf_hub_download(m,"config.json"); rec["downloadable"]=True
        except Exception as e:
            try:
                fs=[s.rfilename for s in i.siblings or []]
                rec["downloadable"]= "PROBE_FAIL:"+type(e).__name__
            except Exception: rec["downloadable"]=False
    except Exception as e:
        rec.update(dict(exists=False, error=f"{type(e).__name__}: {str(e)[:100]}"))
    out.append(rec)
    st = "OK " if rec.get("exists") else "MISS"
    print(st, m.ljust(60), "gated="+str(rec.get("gated")).ljust(8), "dl="+str(rec.get("downloadable")), "created="+str(rec.get("created_at"))[:10], "dloads="+str(rec.get("downloads")))
json.dump(out, open("/raid/MLP/hgkim/guardbench/models/model_probe.json","w"), indent=2)
print("\nWROTE model_probe.json ; exists:", sum(1 for r in out if r.get("exists")), "/", len(out))
