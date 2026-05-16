import pandas as pd

df = pd.read_csv("data/processed/processed_players.csv")

positions = df["position"].fillna("UNK").astype(str).str.upper().value_counts()
total = len(df)
unk = positions.get("UNK", 0)

print("Distribución de posiciones:")
print(positions)

print("\nTotal jugadores:", total)
print("UNK:", unk)
print("Porcentaje UNK:", round(unk / total * 100, 2), "%")
