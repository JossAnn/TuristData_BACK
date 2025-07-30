import pandas as pd

# Cargar y verificar tu CSV
df = pd.read_csv("tu_dataset.csv")
print("Primeras 5 filas:")
print(df.head())

print(f"\nTotal de ejemplos: {len(df)}")
print(f"Distribución de valores nulos por categoría:")
for col in ["atencion", "limpieza", "precio", "comida"]:
    null_count = df[col].isnull().sum()
    valid_count = len(df) - null_count
    print(f"  {col}: {valid_count} válidos, {null_count} nulos")
