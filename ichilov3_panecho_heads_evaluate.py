"""Patient-held-out test of PanEcho measurement heads for next-visit deterioration."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ichilov3_train_fusion import atomic, hashfile


SOURCE = Path(r"D:\DS\ichilov3_panecho_heads_20261004\visit_measurements.parquet")
INPUT = Path(r"D:\DS\ichilov3_nonphilips_future_20260930\aligned")
LABELS = Path(r"D:\us\amber_full_105_preprocessed\Ichilov_july_visits.parquet")
VISITS = Path(r"D:\DS\ichilov3_aligned_20260927\visit_alignment.parquet")
OUTPUT = Path(r"D:\us\output\ichilov_panecho_heads_20261004")
TASKS = ("EF", "LVEDV", "LVESV", "LAVol", "LAIDs2D", "TAPSE", "RVSVel")
ARMS = ("compact_reference", "heads_current", "compact_plus_heads",
        "heads_history", "compact_plus_heads_history")
CS = (.001, .01, .1, 1.)


def load_arrays():
    cohort = pd.read_parquet(INPUT / "cohort.parquet")
    heads = pd.read_parquet(SOURCE).set_index(["patient_id", "visit_date"])
    clinical_visits = pd.read_parquet(LABELS).set_index("visit_id")
    alignment = pd.read_parquet(VISITS).set_index("visit_id")
    with np.load(INPUT / "features.npz") as f:
        if not np.array_equal(f["transition_ids"], cohort.transition_id.to_numpy(str)):
            raise ValueError("Transition features not aligned")
        raw = {k: f[k].copy() for k in ("clinical", "strain_scalars", "strain_curves")}
    raw["strain_curves"] = raw["strain_curves"].reshape(len(cohort), -1)
    current, history, availability = [], [], []
    for r in cohort.itertuples():
        current_key = (r.patient_id, r.current_visit_date)
        if current_key not in heads.index:
            raise ValueError(f"Missing PanEcho current visit: {current_key}")
        cur = heads.loc[current_key, list(TASKS)].to_numpy(dtype=np.float32)
        same = clinical_visits[clinical_visits.patient_id.eq(r.patient_id)].sort_values("visit_order")
        earlier = same[same.visit_order.lt(int(r.current_visit_order))]
        prev = None
        if len(earlier):
            previous_id = earlier.iloc[-1].name
            previous_visit = alignment.loc[previous_id]
            previous_key = (str(previous_visit.patient_id), str(previous_visit.visit_date))
            if previous_key in heads.index:
                prev = heads.loc[previous_key, list(TASKS)].to_numpy(dtype=np.float32)
        current.append(cur)
        history.append(np.concatenate([cur, cur-prev if prev is not None else np.zeros_like(cur),
                                       np.array([float(prev is not None)], np.float32)]))
        availability.append(prev is not None)
    raw["heads_current"] = np.stack(current)
    raw["heads_history"] = np.stack(history)
    if not np.isfinite(raw["heads_current"]).all() or sum(availability) != 102:
        raise ValueError(f"Invalid head coverage: {sum(availability)} histories")
    return cohort, raw, int(sum(availability))


def preprocess(raw, train, test, seed):
    processed = {}
    for name, x in raw.items():
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                 ("scale", StandardScaler())]
        if name in {"strain_scalars", "strain_curves"}:
            steps += [("pca", PCA(n_components=min(8, len(train)-1, x.shape[1]),
                                   svd_solver="randomized", random_state=seed)),
                      ("component_scale", StandardScaler())]
        pipe = Pipeline(steps)
        processed[name] = (pipe.fit_transform(x[train]), pipe.transform(x[test]))
    base_names = ("clinical", "strain_scalars", "strain_curves")
    base = [np.concatenate([processed[n][i] for n in base_names], axis=1) for i in (0, 1)]
    current = processed["heads_current"]
    history = processed["heads_history"]
    return {
        "compact_reference": base,
        "heads_current": current,
        "compact_plus_heads": [np.concatenate([base[i], current[i]], axis=1) for i in (0, 1)],
        "heads_history": history,
        "compact_plus_heads_history": [np.concatenate([base[i], history[i]], axis=1) for i in (0, 1)],
    }


def ppv_top20(y, p):
    return float(y[np.argsort(p)[-max(1, math.ceil(.2*len(y))):]].mean())


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    folds_dir = OUTPUT / "folds"; folds_dir.mkdir(exist_ok=True)
    cohort, raw, history_count = load_arrays()
    y = cohort.label.to_numpy(int)
    groups = cohort.patient_id.to_numpy(str)
    protocol = {"tasks": TASKS, "aggregation": "LV: A2C/A3C/A4C; LA: A2C/A4C; RV: A4C; per-window mean",
                "arms": ARMS, "Cs": CS, "outer": "retained 3x5 patient-held-out",
                "inner": "3 StratifiedGroupKFold, AP selection", "history_available": history_count,
                "missing_history": "zero delta plus availability indicator",
                "base": "clinical standardized, strain scalars and curves each PCA8, fitted on training patients only",
                "source_sha256": hashfile(SOURCE), "feature_sha256": hashfile(INPUT / "features.npz"),
                "cohort_sha256": hashfile(INPUT / "cohort.parquet"),
                "fold_sha256": hashfile(INPUT / "patient_folds.parquet")}
    path = OUTPUT / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(protocol)):
        raise ValueError("Evaluation protocol changed")
    atomic(path, protocol)
    assignments = pd.read_parquet(INPUT / "patient_folds.parquet")
    with threadpool_limits(limits=4):
        for (repeat, fold), assignment in assignments.groupby(["repeat", "fold"]):
            held = np.flatnonzero(np.isin(groups, assignment.patient_id))
            train = np.flatnonzero(~np.isin(groups, assignment.patient_id))
            if set(groups[train]) & set(groups[held]) or len(np.unique(y[held])) != 2:
                raise ValueError("Invalid outer fold")
            seed = 20261004 + int(repeat)*5 + int(fold)
            pending = [arm for arm in ARMS if not (folds_dir/f"{arm}_r{repeat}_f{fold}.parquet").exists()]
            if not pending: continue
            scores = {arm: {c: [] for c in CS} for arm in pending}
            inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                         .split(np.zeros(len(train)), y[train], groups[train]))
            for a,b in inner:
                fit,val = train[a],train[b]
                if set(groups[fit]) & set(groups[val]): raise ValueError("Inner patient overlap")
                matrices = preprocess(raw, fit, val, seed)
                for arm in pending:
                    u,v = matrices[arm]
                    for c in CS:
                        model = LogisticRegression(C=c,max_iter=3000).fit(u,y[fit])
                        scores[arm][c].append(average_precision_score(y[val],model.predict_proba(v)[:,1]))
            matrices = preprocess(raw, train, held, seed)
            for arm in pending:
                best = max(CS, key=lambda c: (np.mean(scores[arm][c]),-c))
                u,v = matrices[arm]
                pred = LogisticRegression(C=best,max_iter=3000).fit(u,y[train]).predict_proba(v)[:,1]
                result = cohort.iloc[held][["transition_id","patient_id","label"]].copy()
                result = result.assign(model=arm,repeat=int(repeat),fold=int(fold),score=pred)
                dest = folds_dir/f"{arm}_r{repeat}_f{fold}.parquet"
                temp=dest.with_suffix(".tmp");result.to_parquet(temp,index=False);temp.replace(dest)
                atomic(dest.with_suffix(".json"),{"C":best,"inner_ap":{str(c):float(np.mean(v)) for c,v in scores[arm].items()}})
                print(f"{arm} r{repeat} f{fold}",flush=True)
            atomic(OUTPUT/"status.json",{"folds_complete":len(list(folds_dir.glob('*.parquet'))),"folds_total":75,"pid":os.getpid()})
    files=list(folds_dir.glob("*.parquet"))
    if len(files)!=75: raise ValueError("Missing folds")
    raw_pred=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    if raw_pred.groupby(["model","transition_id"]).size().ne(3).any():raise ValueError("Missing repeated predictions")
    raw_pred.to_parquet(OUTPUT/"fold_predictions.parquet",index=False)
    oof=raw_pred.groupby(["model","transition_id","patient_id","label"],as_index=False).score.mean()
    retained=pd.read_parquet(INPUT/"retained_baseline_oof.parquet")[["transition_id","patient_id","label","score"]]
    retained=retained.assign(model="retained_cnn_moment_reference")
    rows=[oof,retained]
    for arm in ("heads_current","heads_history"):
        t=oof[oof.model.eq(arm)].merge(retained,on=["transition_id","patient_id","label"],suffixes=("_heads","_reference"),validate="one_to_one")
        fused=t[["transition_id","patient_id","label"]].copy()
        fused["model"]=arm+"_late25"
        fused["score"]=.25*t.score_heads+.75*t.score_reference
        rows.append(fused)
    all_pred=pd.concat(rows,ignore_index=True)
    all_pred.to_parquet(OUTPUT/"oof_predictions.parquet",index=False)
    wide=all_pred.pivot(index=["transition_id","patient_id","label"],columns="model",values="score")
    if wide.isna().any().any() or len(wide)!=len(cohort):raise ValueError("Unequal prediction cohort")
    yy=wide.index.get_level_values("label").to_numpy(int)
    patients=wide.index.get_level_values("patient_id").to_numpy(str)
    metrics=[]
    for name in wide.columns:
        p=wide[name].to_numpy()
        metrics.append({"model":name,"n":len(yy),"events":int(yy.sum()),
                        "auc":roc_auc_score(yy,p),"ap":average_precision_score(yy,p),
                        "ppv_top20":ppv_top20(yy,p),"brier":brier_score_loss(yy,p)})
    metrics=pd.DataFrame(metrics).sort_values("model")
    metrics.to_csv(OUTPUT/"metrics.csv",index=False)
    rng=np.random.default_rng(20261004)
    unique=np.unique(patients);index={p:np.flatnonzero(patients==p) for p in unique}
    paired=[]
    comparisons=(("compact_plus_heads","compact_reference"),("compact_plus_heads_history","compact_reference"),
                 ("heads_current_late25","retained_cnn_moment_reference"),
                 ("heads_history_late25","retained_cnn_moment_reference"))
    for arm,base in comparisons:
        a=wide[arm].to_numpy();b=wide[base].to_numpy()
        rows=[]
        for _ in range(1000):
            sample=np.concatenate([index[p] for p in rng.choice(unique,len(unique),replace=True)])
            if len(np.unique(yy[sample]))<2:continue
            rows.append([roc_auc_score(yy[sample],a[sample])-roc_auc_score(yy[sample],b[sample]),
                         average_precision_score(yy[sample],a[sample])-average_precision_score(yy[sample],b[sample])])
        samples=np.asarray(rows)
        paired.append({"model":arm,"reference":base,
                       "delta_auc":roc_auc_score(yy,a)-roc_auc_score(yy,b),
                       "delta_auc_lo":np.quantile(samples[:,0],.025),"delta_auc_hi":np.quantile(samples[:,0],.975),
                       "delta_ap":average_precision_score(yy,a)-average_precision_score(yy,b),
                       "delta_ap_lo":np.quantile(samples[:,1],.025),"delta_ap_hi":np.quantile(samples[:,1],.975)})
    pd.DataFrame(paired).to_csv(OUTPUT/"paired_deltas.csv",index=False)
    (OUTPUT/"results.md").write_text("# PanEcho measurement-head prediction test\n\n"+metrics.to_string(index=False)+"\n\nPaired changes:\n\n"+pd.DataFrame(paired).to_string(index=False)+"\n",encoding="utf-8")
    atomic(OUTPUT/"run_complete.json",{"models":metrics.model.tolist(),"transitions":len(cohort),"events":int(yy.sum())})
    print(metrics.to_string(index=False),flush=True)
    print(pd.DataFrame(paired).to_string(index=False),flush=True)


if __name__=="__main__":main()
