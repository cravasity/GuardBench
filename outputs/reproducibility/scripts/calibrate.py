import json, time, os, collections
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

items=[json.loads(l) for l in open("/raid/MLP/hgkim/guardbench/data/eval_items.jsonl")]
by_task=collections.defaultdict(list)
for it in items: by_task[it["task"]].append(it)

M="meta-llama/Llama-Guard-3-8B"
tok=AutoTokenizer.from_pretrained(M)
N=300
sample = by_task["prompt_harm"][:N//2] + by_task["response_harm"][:N//2]
prompts=[]
for it in sample:
    msgs=[{"role":"user","content":it["prompt"]}]
    if it["task"]=="response_harm": msgs.append({"role":"assistant","content":it["response"]})
    prompts.append(tok.apply_chat_template(msgs, tokenize=False))

ntok=[len(tok(p).input_ids) for p in prompts]
print(f"prompt tokens: mean={sum(ntok)/len(ntok):.0f} max={max(ntok)} min={min(ntok)}")

t0=time.time()
llm=LLM(model=M, tensor_parallel_size=1, gpu_memory_utilization=0.85,
        max_model_len=8192, enforce_eager=False, disable_log_stats=True)
tload=time.time()-t0
sp=SamplingParams(temperature=0.0, max_tokens=12, logprobs=20)
t1=time.time(); outs=llm.generate(prompts, sp); tgen=time.time()-t1

tot_in=sum(ntok); tot_out=sum(len(o.outputs[0].token_ids) for o in outs)
print(f"\nLOAD  {tload:.1f}s")
print(f"GEN   {tgen:.1f}s for {N} items -> {N/tgen:.1f} items/s ; in={tot_in} out={tot_out} ; {tot_in/tgen:.0f} in-tok/s")
print(f"\nsample outputs: {[o.outputs[0].text.strip()[:22] for o in outs[:4]]}")
ALL=len(items)
print(f"\nEXTRAPOLATION for full suite ({ALL} items): {ALL/(N/tgen)/60:.1f} min gen + {tload/60:.1f} min load")
json.dump(dict(items_per_s=N/tgen, load_s=tload, mean_in_tok=sum(ntok)/len(ntok)),
          open("/raid/MLP/hgkim/guardbench/logs/calib.json","w"))
