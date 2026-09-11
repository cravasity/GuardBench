"""Guard model manifest: what to download and how to run each one."""

# runner: "vllm_gen" = generative guard served by vLLM (logprobs on decision token)
#         "hf_cls"   = HF sequence-classification head (softmax probs)
MODELS = [
 # ---------------- Meta Llama Guard family ----------------
 dict(key="llamaguard-1-7b",   repo="meta-llama/LlamaGuard-7b",                  runner="vllm_gen", family="Llama Guard", params="6.7B", released="2023-12-05", blocked="gated-manual (license not accepted)"),
 dict(key="llamaguard-2-8b",   repo="meta-llama/Meta-Llama-Guard-2-8B",          runner="vllm_gen", family="Llama Guard", params="8.0B", released="2024-04-17"),
 dict(key="llamaguard-3-1b",   repo="meta-llama/Llama-Guard-3-1B",               runner="vllm_gen", family="Llama Guard", params="1.5B", released="2024-09-20"),
 dict(key="llamaguard-3-8b",   repo="meta-llama/Llama-Guard-3-8B",               runner="vllm_gen", family="Llama Guard", params="8.0B", released="2024-07-22"),
 dict(key="llamaguard-4-12b",  repo="meta-llama/Llama-Guard-4-12B",              runner="vllm_gen", family="Llama Guard", params="12B",  released="2025-04-23"),
 # ---------------- Google ShieldGemma ----------------
 dict(key="shieldgemma-2b",    repo="google/shieldgemma-2b",                     runner="vllm_gen", family="ShieldGemma", params="2.6B", released="2024-07-16"),
 dict(key="shieldgemma-9b",    repo="google/shieldgemma-9b",                     runner="vllm_gen", family="ShieldGemma", params="9.2B", released="2024-07-16"),
 dict(key="shieldgemma-27b",   repo="google/shieldgemma-27b",                    runner="vllm_gen", family="ShieldGemma", params="27.2B",released="2024-07-16"),
 # ---------------- AI2 ----------------
 dict(key="wildguard-7b",      repo="allenai/wildguard",                         runner="vllm_gen", family="WildGuard",   params="7.2B", released="2024-06-15", blocked="gated-auto (needs one-click accept)"),
 # ---------------- Qwen3Guard ----------------
 dict(key="qwen3guard-0.6b",   repo="Qwen/Qwen3Guard-Gen-0.6B",                  runner="vllm_gen", family="Qwen3Guard",  params="0.6B", released="2025-09-23"),
 dict(key="qwen3guard-4b",     repo="Qwen/Qwen3Guard-Gen-4B",                    runner="vllm_gen", family="Qwen3Guard",  params="4B",   released="2025-09-23"),
 dict(key="qwen3guard-8b",     repo="Qwen/Qwen3Guard-Gen-8B",                    runner="vllm_gen", family="Qwen3Guard",  params="8B",   released="2025-09-23"),
 # ---------------- IBM Granite Guardian ----------------
 dict(key="granite-3.0-8b",    repo="ibm-granite/granite-guardian-3.0-8b",       runner="vllm_gen", family="GraniteGuardian", params="8.1B", released="2024-10-15"),
 dict(key="granite-3.3-8b",    repo="ibm-granite/granite-guardian-3.3-8b",       runner="vllm_gen", family="GraniteGuardian", params="8.2B", released="2025-06-03"),
 dict(key="granite-4.1-8b",    repo="ibm-granite/granite-guardian-4.1-8b",       runner="vllm_gen", family="GraniteGuardian", params="8.4B", released="2026-04-16"),
 # ---------------- NVIDIA ----------------
 dict(key="nemotron-guard-v3", repo="nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3", runner="vllm_gen", family="Nemotron", params="8.0B", released="2025-08-20"),
 dict(key="nemotron-3.5-cs",   repo="nvidia/Nemotron-3.5-Content-Safety",        runner="vllm_gen", family="Nemotron", params="4.3B", released="2026-05-22"),
 # ---------------- OpenAI ----------------
 dict(key="gptoss-safeguard-20b", repo="openai/gpt-oss-safeguard-20b",           runner="vllm_gen", family="gpt-oss-safeguard", params="21.5B", released="2025-09-18"),
 # ---------------- Mistral ----------------
 dict(key="shieldstral-3b",    repo="mistralai/Shieldstral-1.0-3B",              runner="vllm_gen", family="Shieldstral", params="3.85B", released="2026-07-16"),
 # ---------------- Benchmark judges ----------------
 dict(key="mdjudge-v0.2-7b",   repo="OpenSafetyLab/MD-Judge-v0_2-internlm2_7b",  runner="vllm_gen", family="MD-Judge", params="7.7B", released="2024-07-21"),
 dict(key="harmbench-cls-13b", repo="cais/HarmBench-Llama-2-13b-cls",            runner="vllm_gen", family="HarmBench-cls", params="13B", released="2024-02-03"),
 # ---------------- Reasoning / dynamic ----------------
 dict(key="guardreasoner-8b",  repo="yueliu1999/GuardReasoner-8B",               runner="vllm_gen", family="GuardReasoner", params="8B", released="2025-01-30"),
 dict(key="dynaguard-8b",      repo="tomg-group-umd/DynaGuard-8B",               runner="vllm_gen", family="DynaGuard", params="8B", released="2025-07-02"),
 # ---------------- Multilingual ----------------
 dict(key="polyguard-qwen-7b", repo="ToxicityPrompts/PolyGuard-Qwen",            runner="vllm_gen", family="PolyGuard", params="7.6B", released="2024-12-31"),
 # ---------------- Newer / 2026 ----------------
 dict(key="aprielguard-8b",    repo="ServiceNow-AI/AprielGuard",                 runner="vllm_gen", family="AprielGuard", params="7.9B", released="2025-11-21"),
 # ---------------- Non-English ----------------
 dict(key="shieldlm-7b",       repo="thu-coai/ShieldLM-7B-internlm2",            runner="vllm_gen", family="ShieldLM", params="7.7B", released="2024-02-26"),
 dict(key="sguard-2b",         repo="SamsungSDS-Research/SGuard-ContentFilter-2B-v1", runner="hf_cls", family="SGuard", params="2.5B", released="2025-11-11"),
 dict(key="kanana-safeguard-8b", repo="kakaocorp/kanana-safeguard-8b",           runner="vllm_gen", family="Kanana Safeguard", params="8B", released="2025-05-26"),
 # ---------------- Encoder / small classifiers ----------------
 dict(key="harmaug-guard",     repo="hbseong/HarmAug-Guard",                     runner="hf_cls", family="HarmAug", params="435M", released="2024-10-11"),
 dict(key="duoguard-0.5b",     repo="DuoGuard/DuoGuard-0.5B",                    runner="hf_cls", family="DuoGuard", params="0.5B", released="2025-02-07"),
 dict(key="librai-harmful",    repo="LibrAI/longformer-harmful-ro",              runner="hf_cls", family="LibrAI", params="149M", released="2023-08-25"),
 dict(key="librai-action",     repo="LibrAI/longformer-action-ro",               runner="hf_cls", family="LibrAI", params="149M", released="2023-08-24"),
 dict(key="granite-hap-125m",  repo="ibm-granite/granite-guardian-hap-125m",     runner="hf_cls", family="GraniteGuardian", params="125M", released="2024-09-05"),
 dict(key="toxic-bert",        repo="unitary/toxic-bert",                        runner="hf_cls", family="Detoxify", params="109M", released="2022-03-02"),
]
