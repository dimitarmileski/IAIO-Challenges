# # Trace Twins — BASELINE notebook
# 
# Run top to bottom. Replace the `Submission` below's `score_A`/`score_B` with your real methods**, then build and submit `submission.pkl`.

# ### 1. Setup (unzip the train data)

# The train data is in this notebook's folder as train_data.zip — unzip it.
!unzip -o train_data.zip          # -> public_traces.csv

# ### 2. Your `Submission` (edit `score_A`/`score_B`)

import io, pickle
import numpy as np
from collections import Counter

class Submission:
    def __init__(self):
        # Load or train your models here. The baseline currently needs nothing.
        pass

    # ===== DO NOT EDIT: the cloud calls this to collect your scores =====
    def __call__(self, data: bytes) -> bytes:
        req = pickle.loads(data)
        fn = self.score_A if req["part"] == "A" else self.score_B
        scores = fn(req["windows"], req["pairs"])
        buf = io.BytesIO(); np.save(buf, np.asarray(list(scores), dtype=np.float64))
        return buf.getvalue()
    # ===================================================================

    def lcs(self,c1,c2):
        m,n=len(c1),len(c2)
        dp=[[0]*(n+1) for _ in range(m+1)]
        best=0
        for i in range(1,m+1):
            for j in range (1,n+1):
                if c1[i-1]==c2[j-1]:
                    dp[i][j]=dp[i-1][j-1]+1
                    if dp[i][j]>best:
                        best=dp[i][j]
        return best

    def score_A(self, windows, pairs):
        # Part A (real names): placeholder (AUC 0.5 -> 0 pts). REPLACE with a real method.
        scores=[]
        for i,j in pairs:
            c1,c2=windows[i],windows[j]
            LCS=self.lcs(c1,c2)
            scores.append(LCS/200.0)
        return scores

    def score_B(self, windows, pairs):
        def freq_vec(window):
            counts = Counter(window)
            vec = sorted(counts.values(), reverse=True)
            vec += [0] * (200 - len(vec))
            return np.array(vec[:200], dtype=np.float32)

        vecs = [freq_vec(w) for w in windows]
        scores = []
        for i, j in pairs:
            v1, v2 = vecs[i], vecs[j]
            dot = np.dot(v1, v2)
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            sim = dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0
            scores.append(float(sim))
        return scores

# ### 3. Train your solution (run once)

sol = Submission()   # loads/trains everything

# ### 4. (optional) Estimate your score locally

# OPTIONAL local check — self-contained; mirrors the grader (disjoint Part A / Part B
# programs, per-window scramble for B, 50+50 bands). Needs only public_traces.csv.
import csv, random
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

WINDOW = 200; WPP = 8
def _load(p):
    out=[]
    with open(p) as f:
        r=csv.reader(f); next(r)
        for pid,cat,toks in r: out.append({"program_id":int(pid),"category":cat,"tokens":toks.split()})
    return out
def _windows(traces):
    out=[]
    for tr in traces:
        s=tr["tokens"]; n=(len(s)//WINDOW)*WINDOW
        out.extend([{"program_id":tr["program_id"],"category":tr["category"],"wid":j//WINDOW,
                     "tokens":s[j:j+WINDOW]} for j in range(0,n,WINDOW)][:WPP])
    return out
def _pairs(ws,n,seed):
    rng=random.Random(seed); bc=defaultdict(list); bp=defaultdict(list)
    for k,w in enumerate(ws): bc[w["category"]].append(k); bp[w["program_id"]].append(k)
    cats=sorted(bc); multi=[p for p in bp if len(bp[p])>=2]; P=[]; L=[]
    while len(P)<n:
        if rng.random()<0.5:
            p=rng.choice(multi); a,b=rng.sample(bp[p],2); P.append((a,b)); L.append(1)
        else:
            pool=bc[rng.choice(cats)]
            for _ in range(50):
                a,b=rng.sample(pool,2)
                if ws[a]["program_id"]!=ws[b]["program_id"]: P.append((a,b)); L.append(0); break
    return P,L
def _scramble(ws,off):
    vocab=sorted({t for w in ws for t in w["tokens"]}); out=[]
    for w in ws:
        r=random.Random((w["program_id"]*1_000_000+w["wid"])^off); sh=list(vocab); r.shuffle(sh)
        m=dict(zip(vocab,sh)); out.append([m[t] for t in w["tokens"]])
    return out

tr=_load("/home/jovyan/work/t2/public_traces.csv")
by_prog={}
for t in tr: by_prog.setdefault(t["program_id"], t)
ids=sorted(by_prog); random.Random(7).shuffle(ids)
val=ids[:int(len(ids)*0.30)]; h=len(val)//2
wA=_windows([by_prog[i] for i in val[:h]]); WA=[w["tokens"] for w in wA]; pA,lA=_pairs(wA,3000,101)
wB=_windows([by_prog[i] for i in val[h:]]); WB=_scramble(wB,303); pB,lB=_pairs(wB,3000,202)
pts=lambda a,b: max(0.0, min(50.0,(a-0.5)/b*50.0))
aucA=roc_auc_score(lA, sol.score_A(WA,pA)); aucB=roc_auc_score(lB, sol.score_B(WB,pB))
print(f"Part A: AUC {aucA:.3f} -> {pts(aucA,0.34):.1f}/50")
print(f"Part B: AUC {aucB:.3f} -> {pts(aucB,0.28):.1f}/50")
print(f"ESTIMATED TOTAL ~ {pts(aucA,0.34)+pts(aucB,0.28):.1f}/100  (secret set differs slightly)")

# ### 5. Build submission.pkl  (run LAST)

# Build submission.pkl  (this is what you upload as the Output)
import cloudpickle, os
# `sol` was trained above. Re-run the train cell first if you restarted the kernel.
with open("submission.pkl", "wb") as f:
    cloudpickle.dump(sol, f)          
mb = os.path.getsize("submission.pkl") / 1e6
print(f"wrote submission.pkl  ({mb:.1f} MB)  -- must be < 50 MB")
assert mb < 50, "too big: cap model size (fewer trees / depth)"

# ### 6. Submit
# Click the **🦆 Submit to Judge** button in the toolbar and choose `submission.pkl` as the Output.


