import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import mlflow
import mlflow.pytorch

# --- CONFIGURACIÓN ---
EXPERIMENT_NAME = "Oxford_Pets_Transfer_Learning"
RUN_NAME = "MobileNetV2_Run_1"
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.001
IMG_SIZE = 224
SEED = 42


def set_seed(seed):
    """Garantizar reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Preparar Entorno
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Configurar MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Preparar Datos (Oxford-IIIT Pet)
    print("Descargando y preparando datos...")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Usamos dataset oficial de Torchvision. Se descargará en ./data
    train_dataset = torchvision.datasets.OxfordIIITPet(
        root='./data', split='trainval', transform=transform, download=True
    )
    test_dataset = torchvision.datasets.OxfordIIITPet(
        root='./data', split='test', transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Guardar las etiquetas de las clases (JSON)
    class_labels = train_dataset.classes
    classes_path = "classes.json"
    with open(classes_path, "w") as f:
        json.dump(class_labels, f)

    # 3. Iniciar Run de MLflow
    with mlflow.start_run(run_name=RUN_NAME) as run:

        # Log de hiperparámetros
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("model", "mobilenet_v2")
        mlflow.log_param("seed", SEED)

        # 4. Configurar Modelo (Transfer Learning)
        print("Configurando MobileNetV2...")
        # Cargar pesos pre-entrenados
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights)

        # Congelar extractor de características
        for param in model.features.parameters():
            param.requires_grad = False

        # Modificar la última capa clasificadora
        num_classes = len(class_labels)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

        model = model.to(device)

        # Loss y Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

        # 5. Bucle de Entrenamiento
        print("Iniciando entrenamiento...")
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if i % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # Métricas por época
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            print(f"Epoch {epoch + 1} Final Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            # Log de métricas a MLflow [cite: 194]
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)

        # 6. Registrar Modelo y Artefactos
        print("Guardando modelo en MLflow...")
        # Registrar el modelo entrenado [cite: 200]
        mlflow.pytorch.log_model(model, "model")

        # Registrar el archivo JSON con las etiquetas
        mlflow.log_artifact(classes_path)

        print(f"Entrenamiento finalizado. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()