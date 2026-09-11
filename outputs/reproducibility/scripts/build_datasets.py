"""Build the guard-model evaluation suite into one unified item JSONL.

Reproduces the splits/filters of allenai/safety-eval -- the reference implementation
whose 9 datasets / 12 task-instances are the de-facto standard eval suite for guard
models (used by WildGuard, Llama Guard 3, Aegis 2.0, Qwen3Guard, PolyGuard, ...).

Item schema:
  item_id, benchmark, task, prompt, response, label (1=positive class), label_str, core, meta
  task in {prompt_harm, response_harm, response_refusal}
  positive class: *_harm -> "harmful";  response_refusal -> "refusal"
"""
import os, json, random, ast, gzip, lzma, io, collections
import pandas as pd
from huggingface_hub import hf_hub_download

TOK = os.environ.get("HF_TOKEN")
SE  = "/raid/MLP/hgkim/guardbench/safety-eval/evaluation/tasks/classification"
OUT = "/raid/MLP/hgkim/guardbench/data"
items = []

PROVENANCE = {}
def dl(repo, fn, rev=None):
    return hf_hub_download(repo, fn, repo_type="dataset", token=TOK, revision=rev)

def dl_any(name, candidates):
    """Try (repo, file, revision) candidates in order; record which one was used.
    Official gated sources are listed first; ungated schema-identical mirrors follow."""
    errs = []
    for repo, fn, rev in candidates:
        try:
            p = dl(repo, fn, rev)
            PROVENANCE[name] = dict(repo=repo, file=fn, revision=rev,
                                    official=(candidates[0][0] == repo))
            if candidates[0][0] != repo:
                print(f"  [!] {name}: official '{candidates[0][0]}' is gated -> using mirror '{repo}'")
            return p
        except Exception as e:
            errs.append(f"{repo}: {type(e).__name__}")
    raise RuntimeError(f"{name}: all sources failed -> {errs}")

def read_jsonl_compressed(path):
    op = gzip.open if path.endswith(".gz") else (lzma.open if path.endswith(".xz") else open)
    with op(path, "rt", encoding="utf-8") as f:
        txt = f.read().strip()
    try:                                    # some are a single JSON array
        return json.loads(txt)
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.split("\n") if l.strip()]

def add(bench, task, idx, prompt, response, label, label_str, core=True, **meta):
    items.append(dict(item_id=f"{bench}::{task}::{idx}", benchmark=bench, task=task,
                      prompt=prompt, response=response or "", label=int(label),
                      label_str=label_str, core=core, meta=meta))

# ---------- 1. WildGuardTest : prompt harm / response harm / refusal ----------
wg = pd.read_parquet(dl_any("wildguardtest", [
    ("allenai/wildguardmix", "test/wildguard_test.parquet", None),
    ("iagoalves/wildguardmix_eval", "data/test-00000-of-00001.parquet", None),
]))
wg["prompt_type"] = wg["adversarial"].map(lambda a: "adversarial" if a else "vanilla")
for task, col, pos in [("prompt_harm","prompt_harm_label","harmful"),
                       ("response_harm","response_harm_label","harmful"),
                       ("response_refusal","response_refusal_label","refusal")]:
    d = wg.dropna(subset=[col])
    for i,(_,r) in enumerate(d.iterrows()):
        add("wildguardtest", task, i, r["prompt"],
            "" if task=="prompt_harm" else r["response"],
            r[col]==pos, r[col], prompt_type=r["prompt_type"], subcategory=r["subcategory"])

# ---------- 2. ToxicChat (toxicchat0124 test, human-annotated subset) ----------
tc = pd.read_csv(dl("lmsys/toxic-chat", "data/0124/toxic-chat_annotation_test.csv"))
i = 0
for _, r in tc.iterrows():
    if not r["human_annotation"]: continue
    cat, score = max(ast.literal_eval(r["openai_moderation"]), key=lambda x: x[1])
    if score < 0.5: cat = "low_confidence"
    tox = bool(r["toxicity"])
    add("toxicchat","prompt_harm",i, r["user_input"], "", tox, "harmful" if tox else "unharmful",
        category_adversarial="adversarial" if r["jailbreaking"] else "vanilla",
        category_openai_moderation=cat)
    i += 1

# ---------- 3. OpenAI Moderation ----------
om = pd.read_json(f"{SE}/openai_mod/openai_moderation_evaluation_dataset.jsonl", orient="records", lines=True)
HC = ["S","H","V","HR","SH","S3","H2","V2"]
for i,(_,r) in enumerate(om.iterrows()):
    h = any(r[c]==1 for c in HC)
    add("openai_mod","prompt_harm",i, r["prompt"], "", h, "harmful" if h else "unharmful",
        moderation_categories=[c for c in HC if r[c]])

# ---------- 4. AEGIS 1.0 test ----------
ae = pd.read_parquet(dl("nvidia/Aegis-AI-Content-Safety-Dataset-1.0",
                        "Content Moderation Extracted Annotations 02.08.24_test_release_0418_v1.parquet"))
i = 0
for _, r in ae.iterrows():
    if r["text_type"] != "user_message": continue
    n_safe = sum(int(r[f"labels_{k}"] == "Safe") for k in (0,1,2))
    h = n_safe <= 1
    add("aegis_v1","prompt_harm",i, str(r["text"]).strip(), "", h, "harmful" if h else "unharmful",
        annotations=[r["labels_0"], r["labels_1"], r["labels_2"]])
    i += 1

# ---------- 5. SimpleSafetyTests (extra: all-positive, near-ceiling) ----------
ss = pd.read_csv(dl("Bertievidgen/SimpleSafetyTests", "sst_test_cases.csv"))
for i,(_,r) in enumerate(ss.iterrows()):
    add("simplesafetytests","prompt_harm",i, r["prompt"], "", 1, "harmful", core=False,
        category=r.get("category"), harm_area=r.get("harm_area"))

# ---------- 6. HarmBench prompt + response ----------
hb = pd.read_csv(f"{SE}/harmbench/harmbench_behaviors_text_test.csv")
hb = hb[hb["FunctionalCategory"].isin(["standard","copyright"])]
for i,(_,r) in enumerate(hb.iterrows()):
    add("harmbench","prompt_harm",i, r["Behavior"], "", 1, "harmful", FunctionalCategory=r["FunctionalCategory"])
raw = json.load(open(f"{SE}/harmbench/harmbench_text_behaviors_val_set.json"))
i = 0
for k in raw:
    for x in raw[k]:
        h = (x["human_0"]+x["human_1"]+x["human_2"]) > 1
        add("harmbench","response_harm",i, x["test_case"], x["generation"], h,
            "harmful" if h else "unharmful", experiment=x["experiment"],
            prompt_type="adversarial" if x["experiment"]!="default" else "vanilla")
        i += 1

# ---------- 7. BeaverTails 30k test ----------
bt = read_jsonl_compressed(dl("PKU-Alignment/BeaverTails", "round0/30k/test.jsonl.gz"))
for i,r in enumerate(bt):
    c = r.get("category")
    add("beavertails","response_harm",i, r["prompt"], r["response"], not r["is_safe"],
        "unharmful" if r["is_safe"] else "harmful",
        category=[k for k,v in c.items() if v] if isinstance(c,dict) else c)

# ---------- 8. PKU-SafeRLHF (1000 safe + 1000 harmful, seed 42) ----------
sr = read_jsonl_compressed(dl("PKU-Alignment/PKU-SafeRLHF", "round0/test.jsonl.xz", rev="v0"))
safe_r, harm_r = [], []
for r in sr:
    s0, s1 = r["is_response_0_safe"], r["is_response_1_safe"]
    if s0 == s1: continue
    (safe_r if s0 else harm_r).append((r["prompt"], r["response_0"]))
    (safe_r if s1 else harm_r).append((r["prompt"], r["response_1"]))
random.seed(42)
idx = random.sample(range(len(safe_r)), 1000)
for i,j in enumerate(idx):
    p,r = safe_r[j];  add("saferlhf","response_harm",f"safe{i}", p, r, 0, "unharmful")
for i,j in enumerate(idx):
    p,r = harm_r[j];  add("saferlhf","response_harm",f"harm{i}", p, r, 1, "harmful")

# ---------- 9. XSTest-Response : response harm + refusal ----------
for split, task, pos in [("response_harmfulness","response_harm","harmful"),
                         ("response_refusal","response_refusal","refusal")]:
    x = pd.read_parquet(dl_any(f"xstest_response/{split}", [
        ("allenai/xstest-response", f"data/{split}-00000-of-00001.parquet", None),
        ("heegyu/xstest-response-ko", f"data/{split}-00000-of-00001.parquet", None),
    ]))
    for i,(_,r) in enumerate(x.iterrows()):
        add("xstest_response", task, i, r["prompt"], r["response"], r["label"]==pos, r["label"],
            prompt_type=r["prompt_type"], prompt_harm_category=r["prompt_harm_category"])

# ---------- extra: AEGIS 2.0 test (2025, CC-BY-4.0, 23 categories) ----------
try:
    a2 = json.load(open(dl("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "test.json")))
    if isinstance(a2, dict): a2 = a2.get("data", list(a2.values())[0])
    n = 0
    for i,r in enumerate(a2):
        lab = r.get("prompt_label") or r.get("label")
        p = r.get("prompt") or r.get("text")
        if lab is None or p is None: continue
        h = str(lab).lower().startswith("unsafe")
        add("aegis_v2","prompt_harm",i, p, "", h, "harmful" if h else "unharmful", core=False)
        n += 1
    print(f"aegis_v2: {n} items")
except Exception as e:
    print("aegis_v2 SKIPPED:", type(e).__name__, str(e)[:200])

for _n,_r,_f,_rv in [("toxicchat","lmsys/toxic-chat","data/0124/toxic-chat_annotation_test.csv",None),
    ("openai_mod","allenai/safety-eval(github)","openai_moderation_evaluation_dataset.jsonl",None),
    ("aegis_v1","nvidia/Aegis-AI-Content-Safety-Dataset-1.0","...test_release_0418_v1.parquet",None),
    ("simplesafetytests","Bertievidgen/SimpleSafetyTests","sst_test_cases.csv",None),
    ("harmbench","allenai/safety-eval(github)","harmbench_behaviors_text_test.csv + val_set.json",None),
    ("beavertails","PKU-Alignment/BeaverTails","round0/30k/test.jsonl.gz",None),
    ("saferlhf","PKU-Alignment/PKU-SafeRLHF","round0/test.jsonl.xz","v0"),
    ("aegis_v2","nvidia/Aegis-AI-Content-Safety-Dataset-2.0","test.json",None)]:
    PROVENANCE.setdefault(_n, dict(repo=_r, file=_f, revision=_rv, official=True))

os.makedirs(OUT, exist_ok=True)
json.dump(PROVENANCE, open(f"{OUT}/provenance.json","w"), indent=2)
with open(f"{OUT}/eval_items.jsonl","w") as f:
    for it in items: f.write(json.dumps(it, ensure_ascii=False, default=str)+"\n")

print(f"\nTOTAL ITEMS: {len(items)}   -> {OUT}/eval_items.jsonl\n")
print(f"{'benchmark':<20}{'task':<18}{'n':>7}{'pos':>7}{'neg':>7}   core")
for (b,t),n in sorted(collections.Counter((i['benchmark'],i['task']) for i in items).items()):
    sub = [i for i in items if i['benchmark']==b and i['task']==t]
    pos = sum(i['label'] for i in sub)
    print(f"{b:<20}{t:<18}{n:>7}{pos:>7}{n-pos:>7}   {sub[0]['core']}")
