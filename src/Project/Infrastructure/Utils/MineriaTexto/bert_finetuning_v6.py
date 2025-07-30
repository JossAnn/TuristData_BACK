import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    AutoConfig,
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from tqdm import tqdm
import os


# 1. Modelo personalizado para regresión multi-categoría con valores nulos
class BERTSentimentRegressor(nn.Module):
    def __init__(self, model_name, num_categories=4, dropout=0.3):
        super(BERTSentimentRegressor, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        # Capas para cada categoría (atencion, limpieza, precio, comida)
        hidden_size = self.bert.config.hidden_size

        # Cada categoría tiene dos salidas:
        # 1. Probabilidad de que la categoría esté presente (clasificación binaria)
        # 2. Puntuación 1-5 si está presente (regresión)
        self.category_modules = nn.ModuleDict()

        for category in ["atencion", "limpieza", "precio", "comida"]:
            self.category_modules[category] = nn.ModuleDict(
                {
                    # Detector de presencia (0 = no mencionado, 1 = mencionado)
                    "presence_classifier": nn.Sequential(
                        nn.Linear(hidden_size, 256),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(256, 1),
                        nn.Sigmoid(),
                    ),
                    # Puntuación 1-5 cuando está presente
                    "score_regressor": nn.Sequential(
                        nn.Linear(hidden_size, 256),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(256, 1),
                    ),
                }
            )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Usar el token [CLS] (primer token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Predicciones para cada categoría
        predictions = {}
        for category in ["atencion", "limpieza", "precio", "comida"]:
            # Probabilidad de presencia
            presence_prob = self.category_modules[category]["presence_classifier"](
                pooled_output
            ).squeeze(-1)

            # Puntuación (aplicar sigmoid y escalar a rango 1-5)
            raw_score = self.category_modules[category]["score_regressor"](
                pooled_output
            ).squeeze(-1)
            score = torch.sigmoid(raw_score) * 4 + 1

            predictions[category] = {"presence_prob": presence_prob, "score": score}

        return predictions


# Función personalizada para agrupar datos en batches
def custom_collate_fn(batch):
    """
    Función personalizada para agrupar los datos en batches
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # Agrupar labels por categoría
    labels = {}
    for category in ["atencion", "limpieza", "precio", "comida"]:
        labels[category] = {
            "present": torch.stack(
                [item["labels"][category]["present"] for item in batch]
            ),
            "score": torch.stack([item["labels"][category]["score"] for item in batch]),
            "has_label": torch.stack(
                [item["labels"][category]["has_label"] for item in batch]
            ),
        }

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# 2. Dataset personalizado que maneja valores nulos
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Preparar labels con información de presencia
        processed_labels = {}
        for category in ["atencion", "limpieza", "precio", "comida"]:
            if (
                pd.isna(self.labels[idx][category])
                or self.labels[idx][category] is None
            ):
                # Categoría no mencionada
                processed_labels[category] = {
                    "present": torch.tensor(0.0, dtype=torch.float),  # No presente
                    "score": torch.tensor(0.0, dtype=torch.float),  # Score irrelevante
                    "has_label": torch.tensor(False, dtype=torch.bool),
                }
            else:
                # Categoría mencionada
                processed_labels[category] = {
                    "present": torch.tensor(1.0, dtype=torch.float),  # Presente
                    "score": torch.tensor(
                        float(self.labels[idx][category]), dtype=torch.float
                    ),
                    "has_label": torch.tensor(True, dtype=torch.bool),
                }

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": processed_labels,
        }


# 3. Función de pérdida personalizada que ignora valores nulos
def compute_loss(predictions, labels, presence_weight=1.0, score_weight=1.0):
    """
    Calcula la pérdida solo para las categorías que tienen etiquetas válidas
    """
    total_loss = 0
    total_samples = 0

    bce_criterion = nn.BCELoss()
    mse_criterion = nn.MSELoss()

    for category in ["atencion", "limpieza", "precio", "comida"]:
        # Extraer tensores del batch para esta categoría
        pred_presence = predictions[category]["presence_prob"]
        pred_scores = predictions[category]["score"]

        # Extraer etiquetas verdaderas
        true_presence = labels[category]["present"]
        true_scores = labels[category]["score"]
        has_label_mask = labels[category]["has_label"]

        # Pérdida de detección de presencia (para todas las muestras)
        presence_loss = bce_criterion(pred_presence, true_presence)

        # Pérdida de puntuación (solo para muestras presentes y con etiqueta válida)
        # Crear máscara para muestras que están presentes Y tienen etiqueta válida
        present_and_valid_mask = has_label_mask & (true_presence > 0.5)

        if present_and_valid_mask.sum() > 0:
            pred_scores_valid = pred_scores[present_and_valid_mask]
            true_scores_valid = true_scores[present_and_valid_mask]

            score_loss = mse_criterion(pred_scores_valid, true_scores_valid)
            category_loss = presence_weight * presence_loss + score_weight * score_loss
        else:
            category_loss = presence_weight * presence_loss

        total_loss += category_loss
        total_samples += 1

    return total_loss / max(total_samples, 1)


# 4. Función de entrenamiento
def train_model(model, train_dataloader, val_dataloader, device, epochs=3, lr=2e-5):
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            predictions = model(input_ids, attention_mask)

            # Calcular pérdida personalizada
            loss = compute_loss(predictions, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # Validación
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                predictions = model(input_ids, attention_mask)
                loss = compute_loss(predictions, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {total_train_loss/len(train_dataloader):.4f}")
        print(f"Val Loss: {val_loss/len(val_dataloader):.4f}")
        print("-" * 50)


# 5. Función para cargar datos desde CSV con valores nulos
def load_data_from_csv(csv_path):
    """
    Carga datos desde CSV manejando valores nulos correctamente
    """
    df = pd.read_csv(csv_path)

    texts = df["texto"].tolist()
    labels = []

    for _, row in df.iterrows():
        label_dict = {}
        for category in ["atencion", "limpieza", "precio", "comida"]:
            value = row[category]

            # Verificar si es nulo, vacío, o solo espacios en blanco
            if (
                pd.isna(value)
                or value == ""
                or (isinstance(value, str) and value.strip() == "")
            ):
                label_dict[category] = None
            else:
                try:
                    # Intentar convertir a float
                    label_dict[category] = float(str(value).strip())
                except (ValueError, TypeError):
                    print(
                        f"Advertencia: No se pudo convertir '{value}' a número en categoría '{category}'. Se tratará como nulo."
                    )
                    label_dict[category] = None
        labels.append(label_dict)

    return texts, labels


# 6. Función principal de entrenamiento
def main():
    # Configuración optimizada para dataset de ~700 ejemplos
    MODEL_NAME = "bert-base-multilingual-uncased"
    MAX_LENGTH = 512
    BATCH_SIZE = 8  # Reducido para evitar problemas de memoria en CPU
    EPOCHS = 5  # Aumentado para dataset pequeño
    LEARNING_RATE = 1e-5  # Reducido para evitar overfitting

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar datos desde tu CSV
    print("Cargando datos desde CSV...")

    # CAMBIA 'opiniones.csv' por la ruta a tu archivo CSV
    CSV_PATH = "opiniones.csv"  # <-- CAMBIAR ESTA RUTA SI ES DIFERENTE

    try:
        texts, labels = load_data_from_csv(CSV_PATH)
        print(f"Datos cargados exitosamente: {len(texts)} ejemplos")

        # Mostrar estadísticas del dataset
        category_counts = {"atencion": 0, "limpieza": 0, "precio": 0, "comida": 0}
        total_scores = {"atencion": [], "limpieza": [], "precio": [], "comida": []}

        for label in labels:
            for category in category_counts:
                if label[category] is not None:
                    category_counts[category] += 1
                    total_scores[category].append(label[category])

        print("\nDistribución de categorías:")
        for category, count in category_counts.items():
            if count > 0:
                avg_score = sum(total_scores[category]) / len(total_scores[category])
                print(
                    f"  {category}: {count} ejemplos ({count/len(texts)*100:.1f}%) - Promedio: {avg_score:.2f}"
                )
            else:
                print(f"  {category}: 0 ejemplos (0.0%)")

        # Verificar que tenemos suficientes datos
        if min(category_counts.values()) < 50:
            print(
                "\n⚠️  ADVERTENCIA: Algunas categorías tienen muy pocos ejemplos (<50)."
            )
            print("   El modelo podría no funcionar bien para esas categorías.")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {CSV_PATH}")
        print("Asegúrate de que el archivo esté en el directorio actual.")
        return
    except Exception as e:
        print(f"Error al cargar el CSV: {e}")
        return

    # División train/validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # DataLoaders con función de collate personalizada
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn
    )

    # Modelo
    model = BERTSentimentRegressor(MODEL_NAME)

    print(f"Modelo cargado: {MODEL_NAME}")
    print(f"Ejemplos de entrenamiento: {len(train_texts)}")
    print(f"Ejemplos de validación: {len(val_texts)}")

    # Entrenamiento
    train_model(model, train_dataloader, val_dataloader, device, EPOCHS, LEARNING_RATE)

    # Guardar modelo
    output_dir = "bert-sentiment-finetuned"
    os.makedirs(output_dir, exist_ok=True)

    # Guardar el modelo completo
    torch.save(model.state_dict(), f"{output_dir}/model.pth")

    # Guardar tokenizer
    tokenizer.save_pretrained(output_dir)

    # Guardar configuración
    config = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "categories": ["atencion", "limpieza", "precio", "comida"],
        "presence_threshold": 0.3,  # Umbral más bajo para detectar categorías
    }

    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Modelo guardado en: {output_dir}")


# 7. Función para cargar y usar el modelo entrenado con manejo de nulos
def load_and_predict(model_path, text, presence_threshold=0.3):  # Reducido de 0.5 a 0.3
    # Cargar configuración
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)

    presence_threshold = config.get("presence_threshold", 0.3)

    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Cargar modelo
    model = BERTSentimentRegressor(config["model_name"])
    model.load_state_dict(torch.load(f"{model_path}/model.pth", map_location="cpu"))
    model.eval()

    # Tokenizar texto
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=config["max_length"],
        return_tensors="pt",
    )

    # Predicción
    with torch.no_grad():
        predictions = model(encoding["input_ids"], encoding["attention_mask"])

    # Procesar resultados considerando presencia
    results = {}
    raw_predictions = {}  # Para debugging

    for category in config["categories"]:
        presence_prob = predictions[category]["presence_prob"].item()
        score = predictions[category]["score"].item()

        raw_predictions[category] = {
            "presence_prob": round(presence_prob, 3),
            "raw_score": round(score, 2),
        }

        if presence_prob >= presence_threshold:
            # La categoría está presente
            results[category] = {
                "score": round(score, 2),
                "confidence": round(presence_prob, 3),
            }
        else:
            # La categoría no está presente
            results[category] = None

    return results, raw_predictions


# 8. Función para calcular promedios ignorando nulos
def calculate_average_scores(predictions_list):
    """
    Calcula promedios de puntuaciones ignorando valores nulos
    """
    averages = {}

    for category in ["atencion", "limpieza", "precio", "comida"]:
        scores = []
        for pred in predictions_list:
            if pred[category] is not None:
                scores.append(pred[category]["score"])

        if scores:
            averages[category] = {
                "average": round(sum(scores) / len(scores), 2),
                "count": len(scores),
            }
        else:
            averages[category] = None

    return averages


# Ejemplo de uso para predicción
def predict_example():
    texts = [
        "El restaurante tiene excelente comida pero el servicio es lento",
        "Lugar muy limpio y barato",
        "La atención fue horrible, tardaron mucho",
        "Comida deliciosa, vale cada peso",
    ]

    try:
        all_predictions = []
        for text in texts:
            predictions, raw_preds = load_and_predict("bert-sentiment-finetuned", text)
            all_predictions.append(predictions)

            print(f"\nTexto: {text}")
            print("Probabilidades RAW (para debugging):")
            for category, raw_data in raw_preds.items():
                print(
                    f"  {category}: presencia={raw_data['presence_prob']}, score={raw_data['raw_score']}"
                )

            print("Puntuaciones finales:")
            for category, result in predictions.items():
                if result is not None:
                    print(
                        f"  {category.capitalize()}: {result['score']}/5 (confianza: {result['confidence']})"
                    )
                else:
                    print(f"  {category.capitalize()}: No mencionado")

        # Calcular promedios solo si hay predicciones válidas
        valid_predictions = [
            p for p in all_predictions if any(v is not None for v in p.values())
        ]

        if valid_predictions:
            print("\n" + "=" * 50)
            print("PROMEDIOS GENERALES:")
            averages = calculate_average_scores(valid_predictions)
            for category, avg_data in averages.items():
                if avg_data is not None:
                    print(
                        f"{category.capitalize()}: {avg_data['average']}/5 (basado en {avg_data['count']} menciones)"
                    )
                else:
                    print(f"{category.capitalize()}: Sin menciones")
        else:
            print("\n⚠️  NO SE DETECTARON CATEGORÍAS. Posibles problemas:")
            print("1. El modelo no se entrenó correctamente")
            print("2. El umbral de confianza es muy alto")
            print("3. Los datos de entrenamiento no fueron suficientes")

    except FileNotFoundError:
        print("❌ ERROR: No se encontró el modelo entrenado.")
        print("Asegúrate de que el entrenamiento se haya completado exitosamente.")
        print(
            "Debe existir la carpeta 'bert-sentiment-finetuned' con los archivos del modelo."
        )
    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    # Primero entrenar el modelo
    print("=== INICIANDO ENTRENAMIENTO ===")
    main()

    # Después probar predicciones
    print("\n=== PROBANDO PREDICCIONES ===")
    predict_example()
