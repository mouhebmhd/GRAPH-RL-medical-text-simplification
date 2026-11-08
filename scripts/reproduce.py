"""Task 22 - write manifest.json: git sha, library versions, config hash, seeds, checkpoint + output hashes, LLM model/cache stats,
optional pre-registration hash. Used by reproduce.sh.   python scripts/reproduce.py --out eval [--prereg prereg.md] [--seeds 0,1,2,3,4]"""
import argparse, dataclasses, glob, hashlib, json, os, platform, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


ap = argparse.ArgumentParser(); ap.add_argument("--out", default="eval"); ap.add_argument("--prereg", default=None); ap.add_argument("--seeds", default="0")
ap.add_argument("--llm-model", default=None)
a = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), "..")
def run(c):
    try: return subprocess.check_output(c, cwd=root, stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "n/a"
import torch, spacy, numpy
from sipg_rl.env import EnvConfig
cfg = dataclasses.asdict(EnvConfig())
cache = os.path.join(a.out, "llm_cache.jsonl")
man = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha=run(["git", "rev-parse", "HEAD"]), python=platform.python_version(),
           libs=dict(torch=torch.__version__, spacy=spacy.__version__, numpy=numpy.__version__),
           env_config=cfg, env_config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
           seeds=[int(s) for s in a.seeds.split(",")], llm_model=a.llm_model,
           checkpoints={os.path.basename(p): sha(p) for p in sorted(glob.glob(os.path.join(root, "sipg_policy*.pt")))},
           reward_calibration_sha256=sha(os.path.join(root, "reward_calibration.json")) if os.path.exists(os.path.join(root, "reward_calibration.json")) else None,
           prereg_sha256=sha(a.prereg) if a.prereg and os.path.exists(a.prereg) else None,
           llm_cache_entries=sum(1 for _ in open(cache)) if os.path.exists(cache) else 0,
           outputs={os.path.relpath(p, a.out): sha(p) for p in sorted(glob.glob(os.path.join(a.out, "*.json")) + glob.glob(os.path.join(a.out, "*.md")))})
os.makedirs(a.out, exist_ok=True)
json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
print(f"manifest -> {a.out}/manifest.json  (git {man['git_sha'][:8]}, {len(man['outputs'])} output hashes, {len(man['checkpoints'])} checkpoints)")
