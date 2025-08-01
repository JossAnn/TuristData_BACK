import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
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


# 1. Modelo personalizado para regresión multi-categoría
class BERTSentimentRegressor(nn.Module):
    def __init__(self, model_name, num_categories=4, dropout=0.3):
        super(BERTSentimentRegressor, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        # Capas para cada categoría (atencion, limpieza, precio, comida)
        hidden_size = self.bert.config.hidden_size

        self.regressors = nn.ModuleDict(
            {
                "atencion": nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 1),
                ),
                "limpieza": nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 1),
                ),
                "precio": nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 1),
                ),
                "comida": nn.Sequential(
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
        for category, regressor in self.regressors.items():
            predictions[category] = regressor(pooled_output).squeeze(-1)
            # Aplicar sigmoid y escalar a rango 1-5
            predictions[category] = torch.sigmoid(predictions[category]) * 4 + 1

        return predictions


# 2. Dataset personalizado
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

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": {
                "atencion": torch.tensor(
                    self.labels[idx]["atencion"], dtype=torch.float
                ),
                "limpieza": torch.tensor(
                    self.labels[idx]["limpieza"], dtype=torch.float
                ),
                "precio": torch.tensor(self.labels[idx]["precio"], dtype=torch.float),
                "comida": torch.tensor(self.labels[idx]["comida"], dtype=torch.float),
            },
        }


# 3. Función de entrenamiento
def train_model(model, train_dataloader, val_dataloader, device, epochs=3, lr=2e-5):
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}

            predictions = model(input_ids, attention_mask)

            # Calcular pérdida para cada categoría
            loss = 0
            for category in ["atencion", "limpieza", "precio", "comida"]:
                category_loss = criterion(predictions[category], labels[category])
                loss += category_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # Validación
        model.eval()
        val_predictions = {
            cat: [] for cat in ["atencion", "limpieza", "precio", "comida"]
        }
        val_targets = {cat: [] for cat in ["atencion", "limpieza", "precio", "comida"]}

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = {k: v.to(device) for k, v in batch["labels"].items()}

                predictions = model(input_ids, attention_mask)

                for category in ["atencion", "limpieza", "precio", "comida"]:
                    val_predictions[category].extend(
                        predictions[category].cpu().numpy()
                    )
                    val_targets[category].extend(labels[category].cpu().numpy())

        # Métricas de validación
        val_metrics = {}
        for category in ["atencion", "limpieza", "precio", "comida"]:
            mse = mean_squared_error(val_targets[category], val_predictions[category])
            mae = mean_absolute_error(val_targets[category], val_predictions[category])
            val_metrics[f"{category}_mse"] = mse
            val_metrics[f"{category}_mae"] = mae

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {total_train_loss/len(train_dataloader):.4f}")
        for metric, value in val_metrics.items():
            print(f"Val {metric}: {value:.4f}")
        print("-" * 50)


# 4. Función principal de entrenamiento
def main():
    # Configuración
    MODEL_NAME = "bert-base-multilingual-uncased"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar datos (ejemplo de formato esperado)
    # Tu dataset debe tener columnas: 'texto', 'atencion', 'limpieza', 'precio', 'comida'
    """
    Ejemplo de formato CSV:
    texto,atencion,limpieza,precio,comida
    "El servicio fue excelente pero la comida regular",5,4,3,2
    "Lugar limpio pero caro",3,5,1,3
    """

    # Cargar y preparar datos
    # df = pd.read_csv('tu_dataset.csv')

    # Para este ejemplo, creamos datos sintéticos
    print("Creando datos de ejemplo...")
    sample_data = []
    sample_texts = [
        "El servicio fue excelente, el lugar muy limpio, precio justo y comida deliciosa",
        "Atención terrible, sucio, muy caro, comida horrible",
        "Servicio regular, limpio, precio alto, comida buena",
        "Excelente atención, muy limpio, barato, comida excelente",
        "Mala atención, lugar sucio, precio regular, comida mala",
    ]

    sample_labels = [
        {"atencion": 5, "limpieza": 5, "precio": 4, "comida": 5},
        {"atencion": 1, "limpieza": 1, "precio": 1, "comida": 1},
        {"atencion": 3, "limpieza": 4, "precio": 2, "comida": 4},
        {"atencion": 5, "limpieza": 5, "precio": 5, "comida": 5},
        {"atencion": 2, "limpieza": 2, "precio": 3, "comida": 2},
    ]

    # Multiplicar datos para tener más ejemplos (en la práctica tendrías más datos reales)
    texts = sample_texts * 100
    labels = sample_labels * 100

    # División train/validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

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
    }

    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Modelo guardado en: {output_dir}")


# 5. Función para cargar y usar el modelo entrenado
def load_and_predict(model_path, text):
    # Cargar configuración
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)

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

    # Convertir a valores 1-5
    results = {}
    for category in config["categories"]:
        score = predictions[category].item()
        results[category] = round(score, 2)

    return results


# Ejemplo de uso para predicción
def predict_example():
    text = "El restaurante tiene excelente comida pero el servicio es lento"
    try:
        predictions = load_and_predict("bert-sentiment-finetuned", text)
        print(f"Texto: {text}")
        print("Puntuaciones:")
        for category, score in predictions.items():
            print(f"  {category.capitalize()}: {score}/5")
    except FileNotFoundError:
        print("Primero ejecuta el entrenamiento para crear el modelo")


if __name__ == "__main__":
    main()
    # predict_example()  # Descomenta para probar predicción
