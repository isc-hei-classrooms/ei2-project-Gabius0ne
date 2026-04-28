"""
Régénère manifest.json en scannant les CSV à côté du script.
Usage : python update_manifest.py
"""
import json
from pathlib import Path

# Toujours résolu relativement au script, pas au dossier courant
SCRIPT_DIR = Path(__file__).resolve().parent

# Où chercher les CSV : "." = à côté du script ; "data" = sous-dossier data/
DATA_DIR = SCRIPT_DIR / "."

# Où écrire le manifeste (toujours à côté du script)
MANIFEST_PATH = SCRIPT_DIR / "manifest.json"

# Mapping libellés affichés. Clé = nom de fichier sans .csv.
PRETTY_NAMES = {
    "forecast_data13": "Forecast v13",
    # ajoute ici les futurs fichiers
}

def pretty(stem: str) -> str:
    return PRETTY_NAMES.get(stem, stem.replace("_", " "))

def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Dossier '{DATA_DIR}' introuvable.")

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"Aucun .csv trouvé dans '{DATA_DIR}'.")

    files = []
    for i, f in enumerate(csv_files):
        # Chemin relatif au script (= relatif à index.html)
        rel = f.relative_to(SCRIPT_DIR).as_posix()
        entry = {"name": pretty(f.stem), "path": rel}
        if i == 0:
            entry["default"] = True
        files.append(entry)

    manifest = {"files": files}
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"manifest.json régénéré dans : {MANIFEST_PATH}")
    for entry in files:
        marker = "  ← default" if entry.get("default") else ""
        print(f"  - {entry['name']}  →  {entry['path']}{marker}")

if __name__ == "__main__":
    main()