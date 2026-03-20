import pandas as pd

path = r"C:\Users\gab1a\OneDrive\Documents\energyinfo2\DATA\oiken-data-denom.csv"
df = pd.read_csv(path)

facteur = (108096.84 / 466) * 0.95
df["estimated OIKEN PV production [kWh]"] = df["central valais solar production [kWh]"] * facteur

df.to_csv(path, index=False)
print(f"Facteur appliqué : {facteur:.4f}")
print(f"Colonne ajoutée, max = {df['estimated OIKEN PV production [kWh]'].max():.2f} kWh")