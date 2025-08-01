"""import csv

# Nombre del archivo de entrada y salida
archivo_txt = "opiniones.txt"
archivo_csv = "opiniones.csv"

# Leer el archivo txt y escribir en formato csv
with open(archivo_txt, "r", encoding="utf-8") as infile, open(
    archivo_csv, "w", newline="", encoding="utf-8"
) as outfile:

    # Usamos csv.reader y writer
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        writer.writerow(row)

print(f"Archivo convertido exitosamente: {archivo_csv}")
"""
import csv

# Nombre del archivo de entrada y salida
archivo_txt = "opiniones.txt"
archivo_csv = "opiniones.csv"

# Leer el archivo txt y escribir en formato csv
with open(archivo_txt, "r", encoding="utf-8") as infile, open(
    archivo_csv, "w", newline="", encoding="utf-8"
) as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # Rellenar con strings vacíos si la fila tiene menos de 5 columnas
        while len(row) < 5:
            row.append("")

        # Si hay una columna extra (posición 5 o más)
        if len(row) > 5:
            extra = row[5]
            if extra.strip():  # si no está vacía
                row[4] = extra  # mover a la columna E
            row = row[:5]  # asegurar solo 5 columnas

        writer.writerow(row)

print(f"Archivo convertido exitosamente: {archivo_csv}")
