"""
Retrain all baselines with their Optuna-best params on RAW-16, save per-fold
weight bundles, write fold CSVs, and report standard + deployable F1.
"""
import json, pickle, logging, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore"); logging.basicConfig(level=logging.INFO, format="%(message)s")
ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef
from src.feature_engineering import build_features
from src.pinn import RAW_CHANNELS

cfg = yaml.safe_load(open(ROOT/"config.yaml")); seed = cfg["features"]["random_seed"]
_, df = build_features(str(ROOT/"config.yaml")); df = df.reset_index(drop=True)
y = df["flight"].values; X = df[RAW_CHANNELS].values.astype(np.float32)
best = json.load(open(ROOT/"outputs/results/optuna_best_params.json"))
skf = StratifiedKFold(10, shuffle=True, random_state=seed)
RES = ROOT/"outputs/results"; FB = ROOT/"reproducibility/folds/base_models"

def make(name):
    p = best[name]["best_params"]
    if name=="SVM":
        from sklearn.svm import SVC; return SVC(C=p["C"],gamma=p["gamma"],kernel="rbf",probability=True,random_state=seed)
    if name=="XGBoost":
        from xgboost import XGBClassifier; return XGBClassifier(**{k:p[k] for k in p},random_state=seed,verbosity=0,eval_metric="logloss")
    if name=="LightGBM":
        from lightgbm import LGBMClassifier; return LGBMClassifier(**{k:p[k] for k in p},random_state=seed,verbose=-1)
    if name=="Random Forest":
        from sklearn.ensemble import RandomForestClassifier; return RandomForestClassifier(**{k:p[k] for k in p},random_state=seed,n_jobs=4)
    if name=="CatBoost":
        from catboost import CatBoostClassifier; return CatBoostClassifier(**{k:p[k] for k in p},random_seed=seed,verbose=0)

CSV = {"SVM":"svm","XGBoost":"xgboost","LightGBM":"lightgbm","Random Forest":"randomforest","CatBoost":"catboost"}

def bt(p,yy):
    b,bf=0.5,0.0
    for t in np.arange(0.2,0.8,0.005):
        f=f1_score(yy,(p>=t).astype(int),zero_division=0)
        if f>bf: bf,b=f,t
    return b

print(f"{'Model':<14}{'stdF1':>8}{'deployF1':>10}{'AUC':>8}")
for name, sub in CSV.items():
    oof=np.full(len(y),np.nan); rows=[]; bdir=FB/sub; bdir.mkdir(parents=True,exist_ok=True)
    for fi,(tr,te) in enumerate(skf.split(X,y),1):
        sc=StandardScaler().fit(X[tr]); m=make(name); m.fit(sc.transform(X[tr]),y[tr])
        prob=m.predict_proba(sc.transform(X[te]))[:,1]; oof[te]=prob; pr=(prob>=0.5).astype(int)
        rows.append({"fold":fi,"accuracy":accuracy_score(y[te],pr),"f1":f1_score(y[te],pr,zero_division=0),
                     "roc_auc":roc_auc_score(y[te],prob),"balanced_acc":balanced_accuracy_score(y[te],pr),
                     "mcc":matthews_corrcoef(y[te],pr)})
        pickle.dump({"fold_idx":fi,"train_idx":tr.tolist(),"test_idx":te.tolist(),"model":m,"scaler":sc},
                    open(bdir/f"fold_{fi:02d}.pkl","wb"))
    pd.DataFrame(rows).to_csv(RES/f"{sub}_folds.csv",index=False)
    dep=[]
    for tr,te in skf.split(X,y):
        t=bt(oof[tr],y[tr]); dep.append(f1_score(y[te],(oof[te]>=t).astype(int),zero_division=0))
    print(f"{name:<14}{pd.DataFrame(rows).f1.mean():>8.4f}{np.mean(dep):>10.4f}{pd.DataFrame(rows).roc_auc.mean():>8.4f}")
print("Tuned baseline bundles + CSVs saved.")
