import pickle
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA")

# Charger le modèle 12h (t=048) — le plus impacté par le PV
model = pickle.load(open(BASE / "models3" / "lgbm_t048.pkl", "rb"))

# Sommer les importances par variable (toutes stations et horizons confondus)
names = model.feature_name()
imps  = model.feature_importance()

groups = {}
for name, imp in zip(names, imps):
    # Extraire la variable principale
    if name.startswith("pred_"):
        parts = name.split("_")
        # pred_glob_rad_h13_Sion_t01 → variable = glob_rad
        var = "_".join(parts[1:-3]) if len(parts) > 4 else parts[1]
    elif name.startswith("rmet_"):
        var = name.split("_")[2]
    elif name.startswith("load_"):
        var = "load_historique"
    else:
        var = name
    groups[var] = groups.get(var, 0) + imp

top = sorted(groups.items(), key=lambda x: -x[1])[:15]
for var, total in top:
    print(f"{total:8.0f}  {var}")