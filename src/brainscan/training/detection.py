"""Entraînement YOLOv8 pour la détection de tumeurs cérébrales.

Usage:
    python -m brainscan.training.detection \
        --data-yaml data/processed/detection/data.yaml \
        --models-dir models/detection \
        --artifacts-dir artifacts/reports/detection \
        --epochs 100 \
        --batch-size 16 \
        --img-size 640 \
        --model-size n  # n, s, m, l, x
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import torch
from ultralytics import YOLO

# MLflow est optionnel
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement YOLOv8 - BrainScan-v2")
    parser.add_argument("--data-yaml", type=Path, required=True, help="Chemin vers data.yaml")
    parser.add_argument("--models-dir", type=Path, default=Path("models/detection"), help="Dossier de sortie pour le modèle")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/reports/detection"), help="Dossier pour les artefacts")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques")
    parser.add_argument("--batch-size", type=int, default=16, help="Taille de batch")
    parser.add_argument("--img-size", type=int, default=640, help="Taille d'image (carrée)")
    parser.add_argument("--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="Taille du modèle YOLOv8")
    parser.add_argument("--lr0", type=float, default=0.01, help="Learning rate initial")
    parser.add_argument("--lrf", type=float, default=0.01, help="Learning rate final (lrf * lr0)")
    parser.add_argument("--momentum", type=float, default=0.937, help="Momentum SGD")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Epochs de warmup")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="", help="Device (cuda, cpu, mps, ou vide pour auto)")
    parser.add_argument("--workers", type=int, default=8, help="Nombre de workers pour le dataloader")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    parser.add_argument("--mlflow-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--mlflow-experiment", type=str, default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "brainscan-v2"))
    parser.add_argument("--run-name", type=str, default=None, help="Nom du run MLflow")
    parser.add_argument("--resume", type=str, default=None, help="Reprendre depuis un checkpoint (chemin .pt)")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Utiliser les poids pré-entraînés COCO")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Ne pas utiliser les poids pré-entraînés")
    parser.add_argument("--augment", action="store_true", default=True, help="Activer l'augmentation de données")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Désactiver l'augmentation")
    return parser.parse_args()


def setup_reproducibility(seed: int):
    """Configure la reproductibilité."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data_yaml(path: Path) -> Dict:
    """Charge le fichier data.yaml."""
    if yaml is None:
        raise ImportError("PyYAML requis pour charger data.yaml. Installez-le avec: pip install PyYAML")
    with open(path) as f:
        return yaml.safe_load(f)


def get_model_path(model_size: str, pretrained: bool) -> str:
    """Retourne le chemin/nom du modèle YOLOv8."""
    suffix = "" if pretrained else "-no-pretrained"
    return f"yolov8{model_size}{suffix}.pt"


def train_yolo(
    data_yaml: Path,
    models_dir: Path,
    artifacts_dir: Path,
    epochs: int,
    batch_size: int,
    img_size: int,
    model_size: str,
    lr0: float,
    lrf: float,
    momentum: float,
    weight_decay: float,
    warmup_epochs: float,
    patience: int,
    device: str,
    workers: int,
    seed: int,
    pretrained: bool,
    augment: bool,
    mlflow_uri: str,
    mlflow_experiment: str,
    run_name: Optional[str],
    resume: Optional[str],
) -> Dict:
    """Entraîne le modèle YOLOv8 et retourne les métriques."""

    # Configuration
    setup_reproducibility(seed)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Chargement data.yaml
    data_config = load_data_yaml(data_yaml)
    class_names = data_config.get("names", [])
    n_classes = data_config.get("nc", len(class_names))

    print(f"[Detection] Classes: {class_names} (nc={n_classes})")
    print(f"[Detection] Device: {device or 'auto'}")
    print(f"[Detection] Modèle: yolov8{model_size} {'(pretrained)' if pretrained else '(from scratch)'}")

    # MLflow (optionnel)
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(mlflow_experiment)

    run_name = run_name or f"detection-yolov8{model_size}-{time.strftime('%Y%m%d-%H%M%S')}"

    mlflow_context = mlflow.start_run(run_name=run_name) if MLFLOW_AVAILABLE else nullcontext()
    with mlflow_context as run:
        # Log hyperparamètres
        if MLFLOW_AVAILABLE:
            mlflow.log_params({
                "model": f"yolov8{model_size}",
                "pretrained": pretrained,
                "epochs": epochs,
                "batch_size": batch_size,
                "img_size": img_size,
                "lr0": lr0,
                "lrf": lrf,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "warmup_epochs": warmup_epochs,
                "patience": patience,
                "device": device or "auto",
                "workers": workers,
                "seed": seed,
                "augment": augment,
                "n_classes": n_classes,
                "class_names": str(class_names),
            })

        # Modèle
        model_name = get_model_path(model_size, pretrained)
        if resume:
            print(f"[Detection] Reprise depuis {resume}")
            model = YOLO(resume)
        else:
            model = YOLO(model_name)

        # Callbacks pour MLflow
        if MLFLOW_AVAILABLE:
            def on_train_epoch_end(trainer):
                mlflow.log_metrics({
                    "train/box_loss": float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0,
                    "train/cls_loss": float(trainer.loss_items[1]) if len(trainer.loss_items) > 1 else 0,
                    "train/dfl_loss": float(trainer.loss_items[2]) if len(trainer.loss_items) > 2 else 0,
                }, step=trainer.epoch)

            def on_fit_epoch_end(trainer):
                # Métriques de validation
                if hasattr(trainer, "metrics") and trainer.metrics:
                    metrics = trainer.metrics
                    mlflow.log_metrics({
                        "val/mAP50": metrics.get("metrics/mAP50(B)", 0),
                        "val/mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
                        "val/precision": metrics.get("metrics/precision(B)", 0),
                        "val/recall": metrics.get("metrics/recall(B)", 0),
                    }, step=trainer.epoch)

            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        # Entraînement
        print(f"[Detection] Début entraînement ({epochs} epochs)...")
        start_time = time.time()

        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            patience=patience,
            device=device if device else None,
            workers=workers,
            seed=seed,
            augment=augment,
            project=str(models_dir),
            name="train",
            exist_ok=True,
            verbose=True,
            plots=True,
            save=True,
            save_period=10,  # Sauvegarde tous les 10 epochs
        )

        train_time = time.time() - start_time
        print(f"[Detection] Entraînement terminé en {train_time/60:.1f} min")

        # Meilleur modèle
        best_model_path = Path(models_dir) / "train" / "weights" / "best.pt"
        last_model_path = Path(models_dir) / "train" / "weights" / "last.pt"

        # Copie du meilleur modèle vers models_dir
        final_model_path = models_dir / f"yolov8{model_size}_best.pt"
        if best_model_path.exists():
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            print(f"[Detection] Modèle sauvegardé : {final_model_path}")

        # Évaluation finale sur test
        print("[Detection] Évaluation sur l'ensemble de test...")
        test_results = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=img_size,
            batch=batch_size,
            device=device if device else None,
            workers=workers,
            verbose=True,
        )

        # Extraction métriques
        metrics = {
            "train_time_seconds": train_time,
            "best_model_path": str(final_model_path) if final_model_path.exists() else str(best_model_path),
            "last_model_path": str(last_model_path) if last_model_path.exists() else None,
        }

        # Métriques d'entraînement (dernière époque)
        if hasattr(results, "results_dict"):
            for k, v in results.results_dict.items():
                metrics[f"train_{k}"] = float(v) if isinstance(v, (int, float)) else v

        # Métriques de test
        if hasattr(test_results, "results_dict"):
            for k, v in test_results.results_dict.items():
                metrics[f"test_{k}"] = float(v) if isinstance(v, (int, float)) else v

        # Métriques clés YOLO
        key_metrics = {
            "mAP50": metrics.get("test_metrics/mAP50(B)", metrics.get("val_metrics/mAP50(B)", 0)),
            "mAP50-95": metrics.get("test_metrics/mAP50-95(B)", metrics.get("val_metrics/mAP50-95(B)", 0)),
            "precision": metrics.get("test_metrics/precision(B)", metrics.get("val_metrics/precision(B)", 0)),
            "recall": metrics.get("test_metrics/recall(B)", metrics.get("val_metrics/recall(B)", 0)),
        }
        metrics.update(key_metrics)

        # Log métriques finales (MLflow)
        if MLFLOW_AVAILABLE:
            for k, v in key_metrics.items():
                mlflow.log_metric(f"final_{k.lower()}", v)

            mlflow.log_metric("train_time_minutes", train_time / 60)

            # Artefacts
            runs_dir = Path(models_dir) / "train"
            if runs_dir.exists():
                # Figures
                for fig_name in ["results.png", "confusion_matrix.png", "F1_curve.png", "PR_curve.png", "P_curve.png", "R_curve.png"]:
                    fig_path = runs_dir / fig_name
                    if fig_path.exists():
                        mlflow.log_artifact(str(fig_path), artifact_path="figures")

                # Fichiers de résultats
                for file_name in ["results.csv", "opt.yaml"]:
                    file_path = runs_dir / file_name
                    if file_path.exists():
                        mlflow.log_artifact(str(file_path), artifact_path="reports")

            # Log du modèle
            if final_model_path.exists():
                mlflow.log_artifact(str(final_model_path), artifact_path="model")

            # Sauvegarde métadonnées locales
            metadata = {
                "task": "brain_tumor_detection",
                "architecture": f"YOLOv8{model_size.upper()}",
                "framework": f"ultralytics {YOLO.__module__.split('.')[0]}",
                "data_yaml": str(data_yaml),
                "classes": class_names,
                "hyperparameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "img_size": img_size,
                    "lr0": lr0,
                    "lrf": lrf,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "warmup_epochs": warmup_epochs,
                    "patience": patience,
                    "pretrained": pretrained,
                    "augment": augment,
                    "seed": seed,
                },
                "metrics": metrics,
                "mlflow_run_id": run.info.run_id,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
            mlflow.log_artifact(str(models_dir / "metadata.json"), artifact_path="reports")

            print(f"[Detection] Run MLflow: {run.info.run_id}")
        else:
            # Sauvegarde métadonnées locales sans MLflow
            metadata = {
                "task": "brain_tumor_detection",
                "architecture": f"YOLOv8{model_size.upper()}",
                "framework": f"ultralytics {YOLO.__module__.split('.')[0]}",
                "data_yaml": str(data_yaml),
                "classes": class_names,
                "hyperparameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "img_size": img_size,
                    "lr0": lr0,
                    "lrf": lrf,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "warmup_epochs": warmup_epochs,
                    "patience": patience,
                    "pretrained": pretrained,
                    "augment": augment,
                    "seed": seed,
                },
                "metrics": metrics,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        print(f"[Detection] Métriques finales: mAP50={key_metrics['mAP50']:.4f}, mAP50-95={key_metrics['mAP50-95']:.4f}")

        return metrics


def main():
    args = parse_args()
    train_yolo(
        data_yaml=args.data_yaml,
        models_dir=args.models_dir,
        artifacts_dir=args.artifacts_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        model_size=args.model_size,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        pretrained=args.pretrained,
        augment=args.augment,
        mlflow_uri=args.mlflow_uri,
        mlflow_experiment=args.mlflow_experiment,
        run_name=args.run_name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()