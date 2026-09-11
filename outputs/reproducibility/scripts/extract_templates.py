"""Pull verbatim prompt templates out of the downloaded model cards."""
import re, glob, json, os
HUB="/raid/MLP/.cache/huggingface/hub"; OUT="/raid/MLP/hgkim/guardbench/templates"
def card(repo):
    p=glob.glob(f"{HUB}/models--{repo.replace('/','--')}/snapshots/*/README.md")
    return open(p[0]).read() if p else None

# ---- ShieldGemma: 4 guidelines x {prompt-only, response} variants (markdown tables) ----
sg=card("google/shieldgemma-9b")
rows=re.findall(r"^\|\s*([A-Z][A-Za-z ]+?)\s*\|\s*`(\"No [^`]+?)`\s*\|\s*$", sg, re.M)
pol={"prompt":{},"response":{}}
for name,txt in rows:
    key=name.strip().lower().replace(" ","_")
    tgt = "prompt" if "The prompt shall not" in txt else ("response" if "The chatbot shall not" in txt else None)
    if tgt: pol[tgt][key]=txt.strip()
assert len(pol["prompt"])==4 and len(pol["response"])==4, (len(pol["prompt"]),len(pol["response"]))
json.dump(pol, open(f"{OUT}/shieldgemma_policies.json","w"), indent=1)
print("shieldgemma:", list(pol["prompt"]), "| response variants:", len(pol["response"]))

# ---- Nemotron Safety Guard 8B v3: the Template(""" ... """) instruction ----
nm=card("nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3")
m=re.search(r'template\s*=\s*Template\("""(.+?)"""\)', nm, re.S)
t=m.group(1)
open(f"{OUT}/nemotron_v3.txt","w").write(t)
print("nemotron_v3: %d chars, has {{query}}=%s {{response}}=%s" % (len(t), "{{ query }}" in t or "{{query}}" in t, "{{ response }}" in t or "{{response}}" in t))
print("   tail:", repr(t[-260:]))
