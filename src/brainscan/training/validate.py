"""Validation de la structure et qualité des données pour BrainScan-v2.

Usage:
    # Classification
    python -m brainscan.training.validate --task classification --data-dir data/processed/classification

    # Détection
    python -m brainscan.training.validate --task detection --data-dir data/processed/detection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

ALLOWED_EXTENSIONS = {".jpeg", ".jpg", ".bmp", ".png", ".JPEG", ".JPG", ".BMP", ".PNG"}


def validate_classification(data_dir: Path) -> Dict:
    """Valide le dataset de classification préparé."""
    print(f"[Validation Classification] Vérification de {data_dir}")

    data_dir = Path(data_dir)
    splits = ["train", "valid", "test"]
    results = {"splits": {}, "total_images": 0, "classes": [], "errors": []}

    all_classes = set()

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            results["errors"].append(f"Dossier manquant : {split_dir}")
            continue

        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        split_classes = [d.name for d in class_dirs]
        all_classes.update(split_classes)

        split_count = 0
        class_counts = {}
        for class_dir in class_dirs:
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
                     list(class_dir.glob("*.png")) + list(class_dir.glob("*.bmp"))
            valid_images = 0
            for img_path in images:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                    with Image.open(img_path) as im:
                        im.load()
                    valid_images += 1
                except Exception as e:
                    results["errors"].append(f"Image corrompue {img_path}: {e}")
            class_counts[class_dir.name] = valid_images
            split_count += valid_images

        results["splits"][split] = {
            "total": split_count,
            "classes": class_counts,
            "n_classes": len(class_counts),
        }
        results["total_images"] += split_count
        print(f"  {split}: {split_count} images valides, classes: {class_counts}")

    results["classes"] = sorted(all_classes)
    print(f"  Total: {results['total_images']} images, {len(results['classes'])} classes: {results['classes']}")

    # Vérification cohérence des classes entre splits
    for split in splits:
        if split in results["splits"]:
            split_classes = set(results["splits"][split]["classes"].keys())
            missing = all_classes - split_classes
            if missing:
                results["errors"].append(f"Split {split} manque des classes: {missing}")

    return results


def validate_yolo_format(label_path: Path, img_width: int, img_height: int, n_classes: int) -> List[str]:
    """Valide un fichier d'annotation YOLO."""
    errors = []
    try:
        with open(label_path) as f:
            lines = f.readlines()
    except Exception as e:
        return [f"Impossible de lire {label_path}: {e}"]

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Ligne {i+1}: format invalide (attendu 5 valeurs, trouvé {len(parts)})")
            continue
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            errors.append(f"Ligne {i+1}: valeurs non numériques")
            continue

        if class_id < 0 or class_id >= n_classes:
            errors.append(f"Ligne {i+1}: class_id {class_id} hors limites [0, {n_classes-1}]")

        for name, val in [("x_center", x_center), ("y_center", y_center), ("width", width), ("height", height)]:
            if val < 0 or val > 1:
                errors.append(f"Ligne {i+1}: {name}={val:.4f} hors plage [0, 1]")

        # Vérification cohérence boîte
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
            errors.append(f"Ligne {i+1}: boîte dépasse de l'image (x_min={x_min:.3f}, y_min={y_min:.3f}, x_max={x_max:.3f}, y_max={y_max:.3f})")

    return errors


def validate_detection(data_dir: Path, class_names: List[str] = None) -> Dict:
    """Valide le dataset de détection au format YOLO."""
    print(f"[Validation Detection] Vérification de {data_dir}")

    data_dir = Path(data_dir)
    splits = ["train", "valid", "test"]
    results = {"splits": {}, "total_images": 0, "total_labels": 0, "classes": [], "errors": [], "warnings": []}

    # Charger data.yaml si présent
    yaml_path = data_dir / "data.yaml"
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f)
            if class_names is None:
                class_names = yaml_data.get("names", [])
            n_classes = yaml_data.get("nc", len(class_names) if class_names else 1)
        except Exception:
            n_classes = 1
    else:
        n_classes = len(class_names) if class_names else 1
        results["warnings"].append("data.yaml non trouvé, validation des classes limitée")

    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    all_class_ids = set()

    for split in splits:
        img_dir = data_dir / split / "images"
        lbl_dir = data_dir / split / "labels"

        if not img_dir.exists():
            results["errors"].append(f"Dossier images manquant : {img_dir}")
            continue
        if not lbl_dir.exists():
            results["errors"].append(f"Dossier labels manquant : {lbl_dir}")
            continue

        images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS])
        labels = sorted([f for f in lbl_dir.iterdir() if f.suffix == ".txt"])

        # Correspondance image <-> label
        img_stems = {f.stem: f for f in images}
        lbl_stems = {f.stem: f for f in labels}

        matched = set(img_stems.keys()) & set(lbl_stems.keys())
        orphan_images = set(img_stems.keys()) - matched
        orphan_labels = set(lbl_stems.keys()) - matched

        split_errors = []
        split_warnings = []
        class_counts = {i: 0 for i in range(n_classes)}

        # Validation des paires
        for stem in matched:
            img_path = img_stems[stem]
            lbl_path = lbl_stems[stem]

            # Validation image
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as e:
                split_errors.append(f"Image corrompue {img_path.name}: {e}")
                continue

            # Validation label
            lbl_errors = validate_yolo_format(lbl_path, w, h, n_classes)
            split_errors.extend([f"{lbl_path.name}: {e}" for e in lbl_errors])

            # Comptage classes
            try:
                with open(lbl_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            class_id = int(line.split()[0])
                            if 0 <= class_id < n_classes:
                                class_counts[class_id] += 1
                                all_class_ids.add(class_id)
            except Exception:
                pass

        # Avertissements orphelins
        if orphan_images:
            split_warnings.append(f"{len(orphan_images)} images sans annotation")
        if orphan_labels:
            split_warnings.append(f"{len(orphan_labels)} annotations sans image")

        results["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "matched": len(matched),
            "orphan_images": len(orphan_images),
            "orphan_labels": len(orphan_labels),
            "class_distribution": class_counts,
            "errors": split_errors,
            "warnings": split_warnings,
        }
        results["total_images"] += len(images)
        results["total_labels"] += len(labels)
        results["errors"].extend(split_errors)
        results["warnings"].extend(split_warnings)

        print(f"  {split}: {len(images)} images, {len(labels)} labels, {len(matched)} paires valides")
        if split_errors:
            print(f"    ❌ {len(split_errors)} erreurs")
        if split_warnings:
            print(f"    ⚠️  {len(split_warnings)} avertissements")

    results["classes"] = class_names
    results["n_classes"] = n_classes
    results["class_ids_found"] = sorted(all_class_ids)

    # Vérification cohérence classes
    expected_ids = set(range(n_classes))
    missing_ids = expected_ids - all_class_ids
    if missing_ids:
        results["warnings"].append(f"Classes non présentes dans les annotations: {missing_ids}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validation des datasets BrainScan-v2")
    parser.add_argument("--task", choices=["classification", "detection"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--class-names", nargs="+", default=None, help="Noms des classes (détection)")
    parser.add_argument("--output", type=Path, default=None, help="Fichier JSON de sortie pour le rapport")

    args = parser.parse_args()

    if args.task == "classification":
        results = validate_classification(args.data_dir)
    else:
        results = validate_detection(args.data_dir, args.class_names)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nRapport sauvegardé : {args.output}")

    # Code de sortie selon les erreurs
    if results["errors"]:
        print(f"\n❌ Validation échouée : {len(results['errors'])} erreur(s)")
        for e in results["errors"][:10]:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n✅ Validation réussie")
        if results["warnings"]:
            print(f"⚠️  {len(results['warnings'])} avertissement(s)")
        sys.exit(0)


if __name__ == "__main__":
    main()