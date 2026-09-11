import os, json
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
CANDS = [
 ("WildGuardTest","allenai/wildguardmix"),
 ("WildGuardTest-alt","allenai/wildguardmix"),
 ("ToxicChat","lmsys/toxic-chat"),
 ("OpenAIModeration","mmathys/openai-moderation-api-evaluation"),
 ("Aegis1.0","nvidia/Aegis-AI-Content-Safety-Dataset-1.0"),
 ("Aegis2.0","nvidia/Aegis-AI-Content-Safety-Dataset-2.0"),
 ("Nemotron-SafetyGuard-v3","nvidia/Nemotron-Safety-Guard-Dataset-v3"),
 ("SimpleSafetyTests","Bertievidgen/SimpleSafetyTests"),
 ("SimpleSafetyTests-walled","walledai/SimpleSafetyTests"),
 ("HarmBench-walled","walledai/HarmBench"),
 ("HarmBench-cais","cais/HarmBench"),
 ("BeaverTails","PKU-Alignment/BeaverTails"),
 ("BeaverTails-Eval","PKU-Alignment/BeaverTails-Evaluation"),
 ("PKU-SafeRLHF","PKU-Alignment/PKU-SafeRLHF"),
 ("PKU-SafeRLHF-30K","PKU-Alignment/PKU-SafeRLHF-30K"),
 ("XSTest-Response","allenai/xstest-response"),
 ("XSTest","walledai/XSTest"),
 ("XSTest-orig","Paul/XSTest"),
 ("SORRY-Bench","sorry-bench/sorry-bench-202503"),
 ("SORRY-Bench-2406","sorry-bench/sorry-bench-202406"),
 ("OR-Bench","bench-llm/or-bench"),
 ("StrongREJECT","walledai/StrongREJECT"),
 ("AdvBench","walledai/AdvBench"),
 ("DoNotAnswer","LibrAI/do-not-answer"),
 ("SALAD-Data","OpenSafetyLab/Salad-Data"),
 ("PolyGuardPrompts","ToxicityPrompts/PolyGuardPrompts"),
 ("PolyGuardMix","ToxicityPrompts/PolyGuardMix"),
 ("ToxiGen","toxigen/toxigen-data"),
 ("RealToxicityPrompts","allenai/real-toxicity-prompts"),
 ("JBB-Behaviors","JailbreakBench/JBB-Behaviors"),
 ("WildJailbreak","allenai/wildjailbreak"),
 ("AIR-Bench","stanford-crfm/air-bench-2024"),
]
out=[]
for name,repo in CANDS:
    rec={"name":name,"repo":repo}
    try:
        i=api.dataset_info(repo)
        cfgs=None
        try:
            from datasets import get_dataset_config_names
            cfgs=get_dataset_config_names(repo, token=os.environ.get("HF_TOKEN"))
        except Exception as e: cfgs="ERR:"+type(e).__name__
        rec.update(exists=True, gated=str(i.gated), created_at=str(i.created_at)[:10],
                   downloads=i.downloads, likes=i.likes, configs=cfgs,
                   license=(i.card_data.get("license") if i.card_data else None))
    except Exception as e:
        rec.update(exists=False, error=f"{type(e).__name__}: {str(e)[:80]}")
    out.append(rec)
    print(("OK  " if rec.get("exists") else "MISS"), name.ljust(26), repo.ljust(50),
          "gated="+str(rec.get("gated")).ljust(7), "dl="+str(rec.get("downloads")).ljust(8),
          "cfgs="+str(rec.get("configs"))[:70])
json.dump(out, open("/raid/MLP/hgkim/guardbench/data/dataset_probe.json","w"), indent=2)
