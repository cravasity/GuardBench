import json, collections, os, datetime, html as H

D="/raid/MLP/hgkim/guardbench"
items=[json.loads(l) for l in open(f"{D}/data/eval_items.jsonl")]
prov=json.load(open(f"{D}/data/provenance.json"))
import sys; sys.path.insert(0,f"{D}/scripts")
from models_manifest import MODELS

SRC={"wildguardtest":"allenai/wildguardmix (WildGuard, NeurIPS 2024 D&B)",
 "toxicchat":"lmsys/toxic-chat, config toxicchat0124 (EMNLP 2023 Findings)",
 "openai_mod":"OpenAI Moderation eval set (Markov et al., AAAI 2023)",
 "aegis_v1":"nvidia/Aegis-AI-Content-Safety-Dataset-1.0",
 "harmbench":"cais/HarmBench (ICML 2024)",
 "beavertails":"PKU-Alignment/BeaverTails 30k test (NeurIPS 2023 D&B)",
 "saferlhf":"PKU-Alignment/PKU-SafeRLHF rev v0",
 "xstest_response":"allenai/xstest-response (XSTest, NAACL 2024)",
 "simplesafetytests":"Bertievidgen/SimpleSafetyTests",
 "aegis_v2":"nvidia/Aegis-AI-Content-Safety-Dataset-2.0"}
WHAT={"wildguardtest":"Adversarial + vanilla prompt/response pairs, human-annotated; the only set covering all three guard tasks at once.",
 "toxicchat":"Real in-the-wild user queries from an LLM demo; heavy class imbalance and implicit toxicity.",
 "openai_mod":"The original moderation benchmark, 8 OpenAI policy categories; the field's oldest common baseline.",
 "aegis_v1":"Human-annotated user turns over a 13-category risk taxonomy, 3 annotators per item.",
 "harmbench":"Adversarial red-team behaviors (prompts) + human-labelled attack generations (responses).",
 "beavertails":"QA pairs with human safety labels over 14 harm categories; the standard response-harm set.",
 "saferlhf":"Preference-annotated response pairs; balanced safe/unsafe slice for response harm.",
 "xstest_response":"Exaggerated-safety contrast set: benign prompts that look harmful. Measures over-refusal.",
 "simplesafetytests":"100 critical-risk prompts, all harmful; a smoke test, not a discriminating benchmark.",
 "aegis_v2":"2025 refresh of AEGIS, 23 categories, CC-BY-4.0."}
ORDER=["wildguardtest","toxicchat","openai_mod","aegis_v1","harmbench","beavertails","saferlhf","xstest_response"]
EXTRA=["simplesafetytests","aegis_v2"]
TASKLBL={"prompt_harm":"prompt harm","response_harm":"response harm","response_refusal":"refusal"}

cells=collections.OrderedDict()
for it in items:
    k=(it["benchmark"],it["task"]); c=cells.setdefault(k,[0,0])
    c[0]+=1; c[1]+=it["label"]

def bench_rows(names):
    rows=[]
    for i,b in enumerate(names,1):
        ts=[(t,cells[(bb,t)]) for (bb,t) in cells if bb==b]
        tasks=" / ".join(TASKLBL[t] for t,_ in ts)
        n=" / ".join(str(c[0]) for _,c in ts)
        pn=" / ".join(f"{c[1]}·{c[0]-c[1]}" for _,c in ts)
        rows.append((str(i),b,tasks,n,pn,SRC[b],WHAT[b]))
    return rows

runnable=[m for m in MODELS if not m.get("blocked")]
blocked=[m for m in MODELS if m.get("blocked")]
fams=collections.OrderedDict()
for m in MODELS: fams.setdefault(m["family"],[]).append(m)

DATE=datetime.date.today().isoformat()
TOTAL=len(items); CORE=sum(1 for i in items if i["core"])

# ---------------- markdown ----------------
md=[]
A=md.append
A(f"# Guard-Model Benchmark Suite — setup summary\n")
A(f"*Compiled {DATE}. Working dir `/raid/MLP/hgkim/guardbench` on the 8×H100 node.*\n")
A("## What this is\n")
A("A run of every open-weight LLM safety **guard model** we could obtain against the standard guard-model")
A("evaluation suite, capturing raw outputs and decision probabilities per item. The item-level output feeds an")
A("Item Response Theory (IRT) analysis whose goal is a condensed, harder benchmark.\n")
A("The unit of analysis is a **classification item**, not a generation. That is a deliberate departure from")
A("Rivera et al. (2026), *Item Response Theory for AI Safety*, which fits IRT over generation benchmarks scored")
A("by LLM judges; they name unmodelled judge error as their largest limitation. Guard classification against")
A("human gold labels removes that confound.\n")
A("---\n")
A(f"## 1. Benchmarks — 8 core + 2 extras, {TOTAL:,} items ({CORE:,} core)\n")
A("Selection rule: the suite implemented in **`allenai/safety-eval`**, the reference harness whose datasets are")
A("the de-facto standard for guard-model evaluation — reused by WildGuard, Llama Guard 3, AEGIS 2.0, Qwen3Guard,")
A("PolyGuard and GuardReasoner. It is 9 datasets over 12 task-instances; we take 8 as core and run the rest anyway.\n")
A("Three tasks: **prompt harmfulness**, **response harmfulness**, **response refusal**.")
A("`pos·neg` = positive (harmful, or refusal) vs negative class counts. All counts reproduce the published splits.\n")
A("| # | Benchmark | Task(s) | n | pos·neg | Source |")
A("|---|---|---|---|---|---|")
for r in bench_rows(ORDER): A(f"| {r[0]} | **{r[1]}** | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
A("\n**Extras (run, not counted in the core 8):**\n")
A("| Benchmark | Task | n | pos·neg | Source |")
A("|---|---|---|---|---|")
for r in bench_rows(EXTRA): A(f"| *{r[1]}* | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
A("\n### What each one measures\n")
for b in ORDER+EXTRA: A(f"- **{b}** — {WHAT[b]}")
A("\n### Two selection calls worth flagging\n")
A("- **SimpleSafetyTests is demoted to an extra.** 100 items, all one label, near-ceiling for every model — it")
A("  carries almost no item-response information and would be dropped by the IRT fit regardless. It stays in the")
A("  run so the exclusion is empirical rather than asserted.")
A("- **AEGIS 2.0 is carried as a candidate addition.** Newer (2025), 23 categories, and CC-BY-4.0 — the cleanest")
A("  license in the suite, which matters if we redistribute a condensed derivative.\n")
A("---\n")
A(f"## 2. Guard models — {len(MODELS)} total, {len(runnable)} runnable\n")
A("Release date = HuggingFace repo creation date (to be cross-checked against arXiv for the paper).\n")
A("| Family | Model | Params | Released | Runner |")
A("|---|---|---|---|---|")
for f,ms in fams.items():
    for j,m in enumerate(ms):
        star=" ⛔" if m.get("blocked") else ""
        A(f"| {f if j==0 else ''} | `{m['repo']}`{star} | {m['params']} | {m['released']} | {m['runner']} |")
A("\n`vllm_gen` = generative guard served by vLLM, decision-token logprobs recorded.")
A("`hf_cls` = classification head, softmax probabilities recorded.\n")
A("### Blocked (⛔) — needs a license accept on the HF account holding our token\n")
for m in blocked: A(f"- **`{m['repo']}`** — {m['blocked']}")
A("\nThe two `allenai` repos (`wildguard` the model, `wildguardmix` the dataset) are one-click auto-approve.")
A("`meta-llama/LlamaGuard-7b` is manual review and slower.\n")
A("---\n")
A("## 3. Data provenance\n")
A("The build script tries the official source first and falls back to a schema-identical ungated mirror,")
A("recording which was used in `data/provenance.json`. Current state:\n")
A("| Dataset | Source used | Official? |")
A("|---|---|---|")
for k,v in prov.items(): A(f"| {k} | `{v['repo']}` | {'yes' if v['official'] else '**mirror**'} |")
A("\nThe two mirrors in play are `iagoalves/wildguardmix_eval` (1,725 rows, columns identical to the official")
A("wildguardtest split) and `heegyu/xstest-response-ko` (a Korean translation that retains the original English")
A("columns verbatim). Both are stopgaps — once the licenses above are accepted the build switches to the")
A("canonical sources with no other change.\n")
A("---\n")
A("## 4. Output format\n")
A("Per (model, benchmark, task), one JSON file recording for every item: ground-truth label, parsed prediction,")
A("probability of the positive class, the raw model output string, and the decision-position logprob")
A("distribution. That per-item matrix is the direct input to the IRT fit.\n")
A("## 5. Status\n")
A("| Stage | State |")
A("|---|---|")
A("| Environment (8×H100, vLLM 0.11.0 / torch 2.8) | done |")
A(f"| Benchmark items built ({TOTAL:,}) | done |")
A(f"| Model downloads | in progress |")
A("| Inference harness (per-family prompt adapters) | in progress |")
A("| Inference runs | not started |")
mdtext="\n".join(md)

OUTDIR="/home/seoultech/MLP/hgkim/Safety Measures/Safety measures"
open(f"{OUTDIR}/guardbench_setup_summary.md","w").write(mdtext)

# ---------------- html for pdf ----------------
def md2html(t):
    import re
    out=[]; intable=False
    for line in t.split("\n"):
        if line.startswith("|"):
            cells=[c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells))<=set("-: "):
                continue
            if not intable: out.append("<table>"); intable=True; tag="th"
            else: tag="td"
            out.append("<tr>"+"".join(f"<{tag}>{c}</{tag}>" for c in cells)+"</tr>")
            continue
        if intable: out.append("</table>"); intable=False
        if line.startswith("### "): out.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "): out.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "): out.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("---"): out.append("<hr>")
        elif line.startswith("- "): out.append(f"<li>{line[2:]}</li>")
        elif line.strip()=="": out.append("")
        else: out.append(f"<p>{line}</p>")
    if intable: out.append("</table>")
    h="\n".join(out)
    h=re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", h)
    h=re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<em>\1</em>", h)
    h=re.sub(r"`(.+?)`", r"<code>\1</code>", h)
    h=re.sub(r"(<li>.*?</li>\n?)+", lambda m:"<ul>"+m.group(0)+"</ul>", h, flags=re.S)
    return h

CSS="""
@page { size: A4; margin: 16mm 14mm; }
body{font-family:"DejaVu Sans",Helvetica,Arial,sans-serif;font-size:9.2pt;line-height:1.45;color:#16181d;}
h1{font-size:19pt;margin:0 0 2px;letter-spacing:-.3px;}
h2{font-size:12.5pt;margin:20px 0 7px;padding-bottom:4px;border-bottom:2px solid #d8dbe0;}
h3{font-size:10.4pt;margin:14px 0 5px;color:#33383f;}
p{margin:5px 0;} hr{border:0;border-top:1px solid #e4e6ea;margin:16px 0;}
table{border-collapse:collapse;width:100%;margin:8px 0 12px;font-size:8.1pt;}
th{background:#eef0f3;text-align:left;font-weight:600;border:1px solid #ccd0d6;padding:4px 6px;}
td{border:1px solid #dcdfe4;padding:3.5px 6px;vertical-align:top;}
tr:nth-child(even) td{background:#fafbfc;}
code{font-family:"DejaVu Sans Mono",monospace;font-size:7.9pt;background:#f2f3f5;padding:.5px 3px;border-radius:2px;}
th code{background:transparent;}
ul{margin:5px 0 5px 0;padding-left:17px;} li{margin:2.5px 0;}
strong{font-weight:600;} em{color:#4a5058;}
"""
htm=f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{md2html(mdtext)}</body></html>"
open("/tmp/gb_summary.html","w").write(htm)
print("wrote md +", OUTDIR)
