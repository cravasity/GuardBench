"""Run many models across a pool of GPUs, one model per GPU at a time."""
import os, sys, json, time, subprocess, threading, queue, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_manifest import MODELS
import adapters as AD

D = "/raid/MLP/hgkim/guardbench"
for variable, directory in {
    "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "TORCH_HOME": "torch",
    "TRITON_CACHE_DIR": "triton", "VLLM_CACHE_ROOT": "vllm",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer", "TORCH_EXTENSIONS_DIR": "torch_extensions",
}.items():
    path = f"{D}/runtime/{directory}"
    os.makedirs(path, exist_ok=True)
    os.environ[variable] = path
os.environ.setdefault("OMP_NUM_THREADS", "4")
# models needing >1 GPU or a smaller context
BIG = {"shieldgemma-27b": dict(tp=2), "gptoss-safeguard-20b": dict(tp=1),
       "harmbench-cls-13b": dict(max_len=2048)}

def wait_for_memory(gpus, timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        used = [int(subprocess.check_output([
            "nvidia-smi", "-i", str(g), "--query-gpu=memory.used",
            "--format=csv,noheader,nounits"], text=True).strip()) for g in gpus]
        if all(m < 2048 for m in used):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPUs {gpus} did not release memory: {used} MiB")
        time.sleep(1)

def worker(q, args, results, lock, available, condition):
    while True:
        try: mkey = q.get_nowait()
        except queue.Empty: return
        cfg = BIG.get(mkey, {})
        count = cfg.get("tp", 1)
        with condition:
            condition.wait_for(lambda: len(available) >= count)
            allocated = sorted(available)[:count]
            available.difference_update(allocated)
        python = f"{D}/.venv-shieldstral/bin/python" if mkey == "shieldstral-3b" else f"{D}/.venv/bin/python"
        cmd = [python, f"{D}/scripts/runner.py", "--model", mkey,
               "--outdir", args.outdir, "--tp", str(cfg.get("tp",1)),
               "--gpu-mem", str(args.gpu_mem), "--max-len", str(cfg.get("max_len", args.max_len))]
        if args.limit: cmd += ["--limit", str(args.limit)]
        gpus = ",".join(map(str, allocated))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, VLLM_LOGGING_LEVEL="ERROR",
                   HF_HOME="/raid/MLP/.cache/huggingface", TOKENIZERS_PARALLELISM="false")
        if os.environ.get("HF_MJ_READ_TOKEN"):
            env["HF_TOKEN"] = os.environ["HF_MJ_READ_TOKEN"]
        if mkey.startswith("shieldgemma-"):
            env["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        if mkey == "shieldstral-3b":
            for variable in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "TRITON_CACHE_DIR",
                             "VLLM_CACHE_ROOT", "FLASHINFER_WORKSPACE_BASE", "TORCH_EXTENSIONS_DIR"):
                env[variable] += "_shieldstral"
                os.makedirs(env[variable], exist_ok=True)
        log = f"{D}/logs/{args.tag}_{mkey}.log"
        t0 = time.time()
        try:
            wait_for_memory(allocated)
            with open(log,"w") as lf:
                rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
            validation = None
            if rc == 0:
                from validate_outputs import validate_model
                validation = validate_model(mkey, args.outdir, args.limit, since=t0)
                if not validation["ok"]:
                    rc = 2
                    with open(log, "a") as lf:
                        lf.write("\nOUTPUT VALIDATION: " + json.dumps(validation) + "\n")
        except Exception as exc:
            rc = 1
            validation = None
            with open(log, "a") as lf:
                lf.write(f"\nScheduler failure: {exc}\n")
        finally:
            with condition:
                available.update(allocated)
                condition.notify_all()
        dt = time.time()-t0
        tail = ""
        try:
            lines = open(log).read().strip().split("\n")
            tail = " | ".join(l for l in lines if l.startswith("WROTE"))[:200] or lines[-1][:200]
        except Exception: pass
        with lock:
            results[mkey] = dict(rc=rc, secs=round(dt,1), gpu=gpus, log=log, tail=tail, validation=validation)
            print(f"[{'ok ' if rc==0 else 'FAIL'}] gpu{gpus} {mkey:<22} {dt:6.0f}s  {tail[:120]}", flush=True)
        q.task_done()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3"); ap.add_argument("--limit", type=int)
    ap.add_argument("--outdir", default=f"{D}/outputs"); ap.add_argument("--tag", default="run")
    ap.add_argument("--models", nargs="*"); ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--max-len", type=int, default=8192)
    a = ap.parse_args()
    keys = a.models or [m["key"] for m in MODELS
                        if not m.get("blocked") and (AD.pick(m["key"]) or m["runner"] == "hf_cls")]
    if not a.limit:
        from validate_outputs import validate_model
        gate = {k: validate_model(k, f"{D}/outputs/_smoke", 2) for k in keys}
        failed = {k: v for k, v in gate.items() if not v["ok"]}
        if failed:
            raise SystemExit("Full-run smoke gate failed: " + json.dumps(failed))
    # biggest first so the long pole starts immediately
    order = {m["key"]: i for i,m in enumerate(MODELS)}
    def sz(k):
        value = next(m for m in MODELS if m["key"] == k)["params"]
        return float(value[:-1]) / (1000 if value.endswith("M") else 1)
    keys.sort(key=lambda k: -sz(k))
    q = queue.Queue()
    for k in keys: q.put(k)
    gpus = [int(g) for g in a.gpus.split(",")]
    if len(gpus) != len(set(gpus)) or not gpus:
        ap.error("GPU IDs must be unique and nonempty")
    if any(BIG.get(k, {}).get("tp", 1) > len(gpus) for k in keys):
        ap.error("Not enough GPUs for requested tensor parallelism")
    results, lock = {}, threading.Lock()
    available, condition = set(gpus), threading.Condition()
    print(f"{len(keys)} models over GPUs {gpus} (limit={a.limit})\n", flush=True)
    T0=time.time()
    ths = [threading.Thread(target=worker, args=(q,a,results,lock,available,condition)) for _ in gpus]
    for t in ths: t.start()
    for t in ths: t.join()
    json.dump(results, open(f"{D}/logs/{a.tag}_summary.json","w"), indent=1)
    ok = sum(1 for v in results.values() if v["rc"]==0)
    print(f"\n=== {ok}/{len(results)} ok, wall {(time.time()-T0)/60:.1f} min ===")
    for k,v in results.items():
        if v["rc"]!=0: print(f"  FAIL {k}: see {v['log']}")
    sys.exit(0 if len(results) == len(keys) and ok == len(keys) else 1)
