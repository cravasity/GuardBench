"""vLLM guard runner -> one JSON per (model, benchmark, task), schema 2.0."""
import os, json, time, math, socket, argparse, collections, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_manifest import MODELS
import adapters as AD

D = "/raid/MLP/hgkim/guardbench"
SCHEMA_VERSION = "2.0"

def sha1(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def load_items(limit=None, benchmarks=None, tasks=None):
    it = [json.loads(l) for l in open(f"{D}/data/eval_items.jsonl")]
    if benchmarks: it = [x for x in it if x["benchmark"] in benchmarks]
    if tasks:      it = [x for x in it if x["task"] in tasks]
    if limit:
        g = collections.defaultdict(list)
        for x in it: g[(x["benchmark"], x["task"])].append(x)
        it = [x for k in g for x in g[k][:limit]]
    return it

def decision_probs(out, call):
    """Locate the decision token (optionally after an anchor regex) and read its logprobs."""
    import re
    def matches(word, vocab):
        return any(word == label[1:] if label.startswith("=") else word.startswith(label) for label in vocab)
    lps = out.logprobs or []
    toks = out.token_ids
    start_tok = 0
    if call.anchor:
        m = re.search(call.anchor, out.text, re.I)
        if m is None: return None, {}, None, None
        off = m.end(); acc = 0
        for i, tid in enumerate(toks):
            piece = ""
            if i < len(lps) and lps[i] and tid in lps[i]:
                piece = lps[i][tid].decoded_token or ""
            acc += len(piece)
            if acc > off: start_tok = i; break
        else: return None, {}, None, None
    for i in range(start_tok, len(toks)):
        if i >= len(lps) or not lps[i]: continue
        cand = lps[i]; tid = toks[i]
        txt = cand[tid].decoded_token if tid in cand else None
        if txt is None: continue
        w = txt.strip().lower().lstrip('"\'')
        if not (matches(w, call.pos) or matches(w, call.neg)):
            continue
        pm = nm = 0.0; table = {}
        for t, lp in cand.items():
            d = (lp.decoded_token or "").strip().lower().lstrip('"\'')
            # Distinct token IDs can decode to the same text (notably Llama-2).
            table[f"{lp.decoded_token} [token_id={t}]"] = lp.logprob
            p = math.exp(lp.logprob)
            if matches(d, call.pos): pm += p
            elif matches(d, call.neg): nm += p
        tot = pm + nm
        return (pm/tot if tot > 0 else None), table, i, txt
    return None, {}, None, None

def _mset(rows):
    tp = sum(1 for r in rows if r["pred"] == 1 and r["label_gt"] == 1)
    fp = sum(1 for r in rows if r["pred"] == 1 and r["label_gt"] == 0)
    fn = sum(1 for r in rows if r["pred"] == 0 and r["label_gt"] == 1)
    tn = sum(1 for r in rows if r["pred"] == 0 and r["label_gt"] == 0)
    P = tp/(tp+fp) if tp+fp else 0.0
    R = tp/(tp+fn) if tp+fn else 0.0
    F = 2*P*R/(P+R) if P+R else 0.0
    sc = [(r["p_positive"], r["label_gt"]) for r in rows if r["p_positive"] is not None]
    auroc = None
    if sc and 0 < sum(y for _, y in sc) < len(sc):
        sc.sort(key=lambda x: x[0]); npos = sum(y for _, y in sc); nneg = len(sc)-npos
        rank = 0.0; i = 0
        while i < len(sc):
            j = i
            while j+1 < len(sc) and sc[j+1][0] == sc[i][0]: j += 1
            avg = (i+j)/2 + 1
            rank += sum(avg for k in range(i, j+1) if sc[k][1] == 1)
            i = j+1
        auroc = (rank - npos*(npos+1)/2) / (npos*nneg)
    return dict(n=len(rows), tp=tp, fp=fp, tn=tn, fn=fn, precision=round(P,4), recall=round(R,4),
                f1=round(F,4), accuracy=round((tp+tn)/len(rows),4) if rows else 0.0,
                auroc=round(auroc,4) if auroc is not None else None)

def metrics(rows):
    """Conservative = parse failure counted as the negative class (safety-eval convention)."""
    cons = [dict(r, pred=(r["pred"] if r["parse_ok"] else 0)) for r in rows]
    only = [r for r in rows if r["parse_ok"]]
    return dict(metrics_conservative=_mset(cons), metrics_parsed_only=_mset(only),
                parse_error_rate=round(sum(1 for r in rows if not r["parse_ok"])/len(rows),4) if rows else 0.0,
                n_parsed=len(only), n_total=len(rows))

def env_block():
    import torch, transformers, vllm
    import importlib.metadata
    return dict(torch=torch.__version__, transformers=transformers.__version__, vllm=vllm.__version__,
                python=sys.version.split()[0], cuda=torch.version.cuda,
                sentencepiece=importlib.metadata.version("sentencepiece"))

def run_model(mkey, limit=None, benchmarks=None, outdir=None, gpu_mem=0.88, tp=1, max_len=8192):
    m = next(x for x in MODELS if x["key"] == mkey)
    if m["runner"] == "hf_cls":
        from hf_cls import run_classifier
        return run_classifier(mkey, limit, benchmarks, outdir, max_len)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from huggingface_hub import snapshot_download
    m = next(x for x in MODELS if x["key"] == mkey)
    cls = AD.pick(mkey, m["family"])
    if cls is None: raise SystemExit(f"no adapter registered for {mkey}")
    snapshot = snapshot_download(m["repo"], local_files_only=True)
    rev = os.path.basename(snapshot)
    if mkey == "shieldstral-3b":
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=f"{snapshot}/tokenizer.json",
                                      bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                      chat_template=open(f"{snapshot}/chat_template.jinja").read())
    else:
        tok = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True, use_fast=mkey != "shieldlm-7b")
    ad = cls(tok, m["repo"])
    max_len = min(max_len, 2048) if mkey == "harmbench-cls-13b" else max_len
    ad.input_budget = max_len - ad.max_tokens
    if mkey.startswith("shieldgemma-"):
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

    items = [i for i in load_items(limit, benchmarks) if i["task"] in ad.supports]
    if not items: raise SystemExit(f"{mkey}: no items for supported tasks {ad.supports}")

    flat, index = [], []          # index: (item_pos, call)
    for pos, it in enumerate(items):
        for c in ad.calls(it):
            flat.append(c.prompt); index.append((pos, c))

    llm = LLM(model=snapshot, tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem,
              max_model_len=max_len, disable_log_stats=True, trust_remote_code=True,
              tokenizer_mode="slow" if mkey == "shieldlm-7b" else "auto",
              **({"limit_mm_per_prompt": {"image": 0}} if mkey == "shieldstral-3b" else {}))
    sampling_options = dict(temperature=0.0, max_tokens=ad.max_tokens, logprobs=20,
                            skip_special_tokens=getattr(ad, "skip_special_tokens", True),
                            stop=["</answer>"] if mkey == "dynaguard-8b" else None)
    generate_options = {}
    if mkey == "shieldstral-3b":
        generate_options["tokenization_kwargs"] = dict(truncation=True, max_length=max_len-ad.max_tokens,
                                                       add_special_tokens=False)
    else:
        sampling_options["truncate_prompt_tokens"] = max_len-ad.max_tokens
    sp = SamplingParams(**sampling_options)
    t0 = time.time(); outs = llm.generate(flat, sp, **generate_options); elapsed = time.time()-t0

    per_item = collections.defaultdict(list)
    for (pos, call), o in zip(index, outs):
        g = o.outputs[0]
        pr = ad.parse_call(call.call_id, g.text, items[pos]["task"])
        p, table, idx, dtok = decision_probs(g, call)
        per_item[pos].append(dict(
            call_id=call.call_id, label_native=pr.get("label_native"), pred=pr.get("pred"),
            categories=pr.get("categories") or [], p_positive=round(p,6) if p is not None else None,
            raw_output=g.text, finish_reason=g.finish_reason,
            decision=dict(token_index=idx, token=dtok, top_logprobs=table),
            prompt_sha1=sha1(call.prompt), n_prompt_tokens=len(o.prompt_token_ids),
            n_output_tokens=len(g.token_ids)))

    grouped = collections.defaultdict(list)
    for pos, it in enumerate(items):
        calls = per_item[pos]
        agg = ad.aggregate(it, calls)
        grouped[(it["benchmark"], it["task"])].append(dict(
            item_id=it["item_id"], label_gt=it["label"], label_gt_str=it["label_str"],
            pred=agg["pred"] if agg["pred"] is not None else 0, pred_raw=agg["pred"],
            pred_native=agg["pred_native"], parse_ok=agg["pred"] is not None,
            p_positive=agg["p_positive"], categories=agg["categories"],
            aggregation=agg["aggregation"], n_calls=len(calls), calls=calls))

    return write_results(m, ad, rev, grouped, elapsed, outdir, tp,
                         dict(temperature=0.0, max_tokens=ad.max_tokens, logprobs=20, greedy=True,
                              truncate_prompt_tokens=max_len-ad.max_tokens,
                              skip_special_tokens=getattr(ad, "skip_special_tokens", True)),
                         dtype=str(llm.llm_engine.model_config.dtype).replace("torch.", ""))

def write_results(m, ad, rev, grouped, elapsed, outdir, tp, sampling, dtype="bfloat16"):
    mkey = m["key"]
    outdir = outdir or f"{D}/outputs"; os.makedirs(outdir, exist_ok=True); written = []
    for (b, t), rows in grouped.items():
        mt = metrics(rows)
        doc = dict(schema_version=SCHEMA_VERSION,
            run=dict(run_id=f"{mkey}__{b}__{t}",
                     timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     host=socket.gethostname(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES","?"),
                     elapsed_s_model_total=round(elapsed,1), vllm_tensor_parallel=tp,
                     environment=env_block()),
            model=dict(key=mkey, repo=m["repo"], revision=rev, family=m["family"], params=m["params"],
                       released=m["released"], runner=m["runner"], dtype=dtype,
                       adapter=ad.name, adapter_version=ad.version,
                       label_mapping=ad.label_mapping, prompt_template=ad.template_text()),
            benchmark=b, task=t,
            positive_class="refusal" if t=="response_refusal" else "harmful",
            sampling=sampling,
            item_source=f"{D}/data/eval_items.jsonl",
            metrics=mt, n_items=len(rows), results=rows)
        p = f"{outdir}/{mkey}__{b}__{t}.json"
        with open(p + ".tmp", "w") as f:
            json.dump(doc, f, ensure_ascii=False, allow_nan=False)
        os.replace(p + ".tmp", p)
        written.append((p, len(rows), mt["metrics_conservative"]["f1"], mt["parse_error_rate"]))
    return written

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--limit", type=int)
    ap.add_argument("--benchmarks", nargs="*"); ap.add_argument("--outdir")
    ap.add_argument("--tp", type=int, default=1); ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--max-len", type=int, default=8192)
    a = ap.parse_args()
    for p,n,f1,per in run_model(a.model, a.limit, a.benchmarks, a.outdir, a.gpu_mem, a.tp, a.max_len):
        print(f"WROTE {os.path.basename(p)}  n={n}  f1={f1}  parse_err={per}")
