import os, sys, json, time, traceback
sys.path.insert(0, os.path.dirname(__file__))
from models_manifest import MODELS
from huggingface_hub import snapshot_download

TOK = os.environ.get("HF_TOKEN")
IGNORE = ["*.pth","*.msgpack","*.h5","original/*","*.gguf","*consolidated*"]
status = {}
outp = "/raid/MLP/hgkim/guardbench/models/download_status.json"
for m in MODELS:
    k, repo = m["key"], m["repo"]
    if m.get("blocked"):
        print(f"SKIP    {k:<22} {repo}  ({m['blocked']})", flush=True)
        status[k] = dict(repo=repo, ok=False, reason=m["blocked"]); continue
    t0 = time.time()
    try:
        p = snapshot_download(repo, token=TOK, ignore_patterns=IGNORE, max_workers=8)
        sz = sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fs in os.walk(p) for f in fs
                 if not os.path.islink(os.path.join(dp,f)))
        print(f"OK      {k:<22} {repo:<52} {sz/1e9:6.1f}GB  {time.time()-t0:5.0f}s", flush=True)
        status[k] = dict(repo=repo, ok=True, path=p, gb=round(sz/1e9,2))
    except Exception as e:
        print(f"FAIL    {k:<22} {repo}  {type(e).__name__}: {str(e)[:120]}", flush=True)
        status[k] = dict(repo=repo, ok=False, reason=f"{type(e).__name__}: {str(e)[:200]}")
    json.dump(status, open(outp,"w"), indent=2)
print("\nDONE ok=%d fail=%d" % (sum(1 for v in status.values() if v['ok']), sum(1 for v in status.values() if not v['ok'])))
