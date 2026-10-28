#!/usr/bin/env python
"""Création d'un dataset de détection synthétique depuis les données de classification.

Puisque le dataset original n'a pas d'annotations bounding box, ce script crée
des annotations YOLO synthétiques en utilisant l'image entière comme boîte
englobante pour la classe correspondante.

Usage:
    python scripts/create_synthetic_detection.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from PIL import Image

# Mapping classification -> détection
CLASS_MAPPING = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,  # Pas de tumeur -> pas d'annotation ou classe "no_tumor"
    "pituitary": 3,
}

DETECTION_CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

def create_synthetic_detection_dataset():
    """Crée un dataset YOLO synthétique depuis data/processed/classification."""
    src_root = Path("data/processed/classification")
    dst_root = Path("data/detection_raw")
    
    if not src_root.exists():
        print(f"❌ Source non trouvée : {src_root}")
        print("   Exécutez d'abord les notebooks de classification (étape 1).")
        return
    
    # Nettoyage destination
    if dst_root.exists():
        shutil.rmtree(dst_root)
    
    images_dir = dst_root / "images"
    labels_dir = dst_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    total_annotations = 0
    
    for split in ["train", "valid", "test"]:
        split_dir = src_root / split
        if not split_dir.exists():
            continue
            
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            if class_name not in CLASS_MAPPING:
                continue
                
            class_id = CLASS_MAPPING[class_name]
            
            # Pour "notumor", on ne crée pas d'annotation (ou classe background)
            # Ici on crée une annotation "no_tumor" couvrant toute l'image
            # pour que le modèle apprenne à détecter l'absence de tumeur
            
            for img_path in class_dir.glob("*.jpg"):
                # Copie image
                dst_img = images_dir / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # Création annotation YOLO (boîte = image entière, normalisée)
                # Format: class_id x_center y_center width height (tout en 0-1)
                # Image entière -> center=(0.5, 0.5), size=(1.0, 1.0)
                label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"
                
                dst_lbl = labels_dir / f"{img_path.stem}.txt"
                dst_lbl.write_text(label_content)
                
                total_images += 1
                total_annotations += 1
    
    print(f"✅ Dataset synthétique créé : {dst_root}")
    print(f"   Images : {total_images}")
    print(f"   Annotations : {total_annotations}")
    print(f"   Classes : {DETECTION_CLASS_NAMES}")
    
    # Création data.yaml
    data_yaml = {
        "path": str(dst_root.absolute()),
        "train": "images",  # Note: on n'a pas encore split, prepare.py le fera
        "val": "images",
        "test": "images",
        "nc": len(DETECTION_CLASS_NAMES),
        "names": DETECTION_CLASS_NAMES,
    }
    
    import yaml
    yaml_path = dst_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"   data.yaml : {yaml_path}")
    
    return dst_root


if __name__ == "__main__":
    create_synthetic_detection_dataset()