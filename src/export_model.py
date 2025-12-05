import torch
import mlflow
import json
import os
from mlflow.tracking import MlflowClient

# Configuración
EXPERIMENT_NAME = "Oxford_Pets_Transfer_Learning"
MODEL_NAME = "mobilenet_v2_onnx"


def main():
    print("🔎 Buscando el mejor modelo en MLflow...")

    # 1. Conectar con MLflow
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print(f"Error: No se encuentra el experimento '{EXPERIMENT_NAME}'")
        return

    # 2. Buscar la mejor ejecución (Run)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.train_accuracy DESC"]
    )

    if not runs:
        print("No se encontraron ejecuciones.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_acc = best_run.data.metrics['train_accuracy']

    print(f"🏆 Mejor Run ID: {best_run_id}")
    print(f"📈 Precisión: {best_acc:.4f}")

    # 3. Cargar modelo desde MLflow
    print("📥 Cargando modelo...")
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)

    # 4. Pasar a CPU para exportar
    model.to("cpu")
    model.eval()

    # 5. Exportar a ONNX (Formato universal)
    print("📦 Exportando a ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = "model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Modelo guardado: {onnx_path}")

    # 6. Descargar etiquetas (classes.json)
    print("📎 Recuperando etiquetas...")
    client.download_artifacts(best_run_id, "classes.json", ".")
    print("✅ Etiquetas descargadas.")


if __name__ == "__main__":
    main()