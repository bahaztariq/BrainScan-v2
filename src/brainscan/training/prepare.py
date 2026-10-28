"""Préparation des datasets pour BrainScan-v2.

Usage:
    # Classification (split 70/15/15 depuis data/Data)
    python -m brainscan.training.prepare --task classification \
        --data-dir data/Data \
        --output-dir data/processed/classification

    # Détection (organisation format YOLO depuis un dossier source avec images + labels)
    python -m brainscan.training.prepare --task detection \
        --data-dir data/detection_raw \
        --output-dir data/processed/detection \
        --split 0.7 0.15 0.15
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Extensions autorisées
ALLOWED_EXTENSIONS = {".jpeg", ".jpg", ".bmp", ".png", ".JPEG", ".JPG", ".BMP", ".PNG"}
SEED = 42


def validate_image(path: Path) -> bool:
    """Vérifie qu'une image peut être ouverte sans erreur."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def prepare_classification(
    data_dir: Path,
    output_dir: Path,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    image_size: int = 224,
) -> pd.DataFrame:
    """Prépare le dataset de classification : split stratifié + écriture images nettoyées."""
    print(f"[Classification] Lecture depuis {data_dir}")
    print(f"[Classification] Écriture vers {output_dir}")

    # 1. Inventaire des fichiers valides
    rows = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for f in class_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
                rows.append({"label": class_dir.name, "file": f.name, "path": str(f)})

    if not rows:
        raise ValueError(f"Aucune image trouvée dans {data_dir}")

    df = pd.DataFrame(rows)
    print(f"[Classification] {len(df)} fichiers candidats trouvés")

    # 2. Validation try/except
    valid, broken = [], []
    for rec in df.to_dict("records"):
        if validate_image(Path(rec["path"])):
            with Image.open(rec["path"]) as im:
                valid.append({**rec, "width": im.width, "height": im.height, "mode": im.mode})
        else:
            broken.append({**rec, "error": "Failed to load/verify"})

    manifest_df = pd.DataFrame(valid)
    broken_df = pd.DataFrame(broken)
    print(f"[Classification] Images valides : {len(manifest_df)} | Écartées : {len(broken_df)}")

    if len(manifest_df) == 0:
        raise ValueError("Aucune image valide après vérification")

    # 3. Split stratifié 70/15/15
    train_ratio, valid_ratio, test_ratio = split_ratios
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-9

    train_df, rest_df = train_test_split(
        manifest_df, test_size=1 - train_ratio, stratify=manifest_df["label"], random_state=SEED
    )
    valid_df, test_df = train_test_split(
        rest_df,
        test_size=test_ratio / (valid_ratio + test_ratio),
        stratify=rest_df["label"],
        random_state=SEED,
    )

    splits = {"train": train_df, "valid": valid_df, "test": test_df}

    # 4. Écriture du dataset nettoyé (RGB, redimensionné)
    output_dir = Path(output_dir)
    for split_name, split_df in splits.items():
        for rec in split_df.to_dict("records"):
            dst_dir = output_dir / split_name / rec["label"]
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / Path(rec["file"]).with_suffix(".jpg").name
            try:
                with Image.open(rec["path"]) as im:
                    im = im.convert("RGB")
                    im = im.resize((image_size, image_size), Image.Resampling.LANCZOS)
                    im.save(dst_path, format="JPEG", quality=95)
            except Exception as e:
                print(f"[Warning] Échec écriture {rec['path']} -> {dst_path}: {e}")

    # 5. Sauvegarde du manifeste de split
    split_manifest = pd.concat([df.assign(split=name) for name, df in splits.items()], ignore_index=True)
    split_manifest[["split", "label", "file", "path"]].to_csv(
        output_dir / "classification_split.csv", index=False
    )

    # 6. Résumé
    summary_rows = []
    for name, df in splits.items():
        vc = df["label"].value_counts()
        summary_rows.append({
            "ensemble": name,
            **{c: int(v) for c, v in vc.items()},
            "total": int(vc.sum()),
            "part": round(vc.sum() / len(manifest_df), 4),
        })
    summary_df = pd.DataFrame(summary_rows).set_index("ensemble")
    summary_df.to_csv(output_dir / "split_summary.csv")
    print(f"[Classification] Split summary:\n{summary_df}")

    return split_manifest


def prepare_detection(
    data_dir: Path,
    output_dir: Path,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    class_names: List[str] = None,
) -> dict:
    """Prépare le dataset de détection au format YOLO.

    Attend une structure source :
    data_dir/
        images/     # fichiers images
        labels/     # fichiers .txt YOLO (même nom de base que l'image)

    Ou une structure plate avec images et labels mélangés.
    """
    print(f"[Detection] Lecture depuis {data_dir}")
    print(f"[Detection] Écriture vers {output_dir}")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Détection des dossiers images/labels
    images_src = data_dir / "images"
    labels_src = data_dir / "labels"

    if images_src.exists() and labels_src.exists():
        image_files = sorted(images_src.rglob("*"))
    else:
        # Structure plate : on sépare par extension
        all_files = sorted(data_dir.rglob("*"))
        image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]

    print(f"[Detection] {len(image_files)} fichiers images trouvés")

    # Association image <-> label
    pairs = []
    orphan_images = []
    orphan_labels = []

    for img_path in image_files:
        if not validate_image(img_path):
            continue
        label_path = img_path.with_suffix(".txt")
        # Chercher le label au même endroit ou dans labels/
        possible_labels = [
            label_path,
            data_dir / "labels" / label_path.name,
            labels_src / label_path.name if labels_src.exists() else None,
        ]
        found_label = None
        for lp in possible_labels:
            if lp and lp.exists():
                found_label = lp
                break

        if found_label:
            pairs.append((img_path, found_label))
        else:
            orphan_images.append(img_path)

    # Labels orphelins
    if labels_src.exists():
        for lbl in labels_src.rglob("*.txt"):
            img_candidates = [lbl.with_suffix(ext) for ext in ALLOWED_EXTENSIONS]
            if not any(c.exists() for c in img_candidates):
                orphan_labels.append(lbl)

    print(f"[Detection] Paires valides : {len(pairs)} | Images orphelines : {len(orphan_images)} | Labels orphelins : {len(orphan_labels)}")

    if len(pairs) == 0:
        raise ValueError("Aucune paire image/annotation valide trouvée")

    # Split
    train_ratio, valid_ratio, test_ratio = split_ratios
    indices = np.arange(len(pairs))
    labels_for_stratify = []
    for _, lbl_path in pairs:
        try:
            with open(lbl_path) as f:
                first_line = f.readline().strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    labels_for_stratify.append(class_id)
                else:
                    labels_for_stratify.append(-1)
        except Exception:
            labels_for_stratify.append(-1)

    train_idx, rest_idx = train_test_split(
        indices, test_size=1 - train_ratio, stratify=labels_for_stratify, random_state=SEED
    )
    valid_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_ratio / (valid_ratio + test_ratio),
        stratify=[labels_for_stratify[i] for i in rest_idx],
        random_state=SEED,
    )

    split_map = {}
    for i in train_idx:
        split_map[i] = "train"
    for i in valid_idx:
        split_map[i] = "valid"
    for i in test_idx:
        split_map[i] = "test"

    # Copie vers la structure YOLO
    for split_name in ["train", "valid", "test"]:
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    for idx, (img_path, lbl_path) in enumerate(pairs):
        split_name = split_map[idx]
        dst_img = output_dir / split_name / "images" / img_path.name
        dst_lbl = output_dir / split_name / "labels" / lbl_path.name
        shutil.copy2(img_path, dst_img)
        shutil.copy2(lbl_path, dst_lbl)

    # Détection des classes depuis les labels
    if class_names is None:
        class_names = _infer_classes_from_labels(pairs)

    # Création de data.yaml
    data_yaml = {
        "path": str(output_dir.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = output_dir / "data.yaml"
    _write_yaml(data_yaml, yaml_path)
    print(f"[Detection] data.yaml créé : {yaml_path}")

    # Résumé
    summary = {}
    for split_name in ["train", "valid", "test"]:
        img_count = len(list((output_dir / split_name / "images").glob("*")))
        lbl_count = len(list((output_dir / split_name / "labels").glob("*.txt")))
        summary[split_name] = {"images": img_count, "labels": lbl_count}
        print(f"[Detection] {split_name}: {img_count} images, {lbl_count} labels")

    return {"pairs": len(pairs), "orphan_images": len(orphan_images), "orphan_labels": len(orphan_labels), "summary": summary, "classes": class_names}


def _infer_classes_from_labels(pairs: List[Tuple[Path, Path]]) -> List[str]:
    """Infère les noms de classes depuis les fichiers d'annotation YOLO."""
    class_ids = set()
    for _, lbl_path in pairs:
        try:
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_ids.add(int(parts[0]))
        except Exception:
            pass
    # Par défaut, on utilise des noms génériques
    max_id = max(class_ids) if class_ids else 0
    return [f"class_{i}" for i in range(max_id + 1)]


def _write_yaml(data: dict, path: Path):
    """Écrit un fichier YAML simple sans dépendance PyYAML."""
    lines = []
    for key, value in data.items():
        if isinstance(value, str):
            lines.append(f"{key}: {value}")
        elif isinstance(value, list):
            lines.append(f"{key}: [{', '.join(repr(v) for v in value)}]")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Préparation des datasets BrainScan-v2")
    parser.add_argument("--task", choices=["classification", "detection"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Dossier source des données brutes")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dossier de sortie pour le dataset préparé")
    parser.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15], help="Ratios train/val/test")
    parser.add_argument("--image-size", type=int, default=224, help="Taille de redimensionnement (classification)")
    parser.add_argument("--class-names", nargs="+", default=None, help="Noms des classes (détection)")

    args = parser.parse_args()

    if args.task == "classification":
        prepare_classification(args.data_dir, args.output_dir, tuple(args.split), args.image_size)
    else:
        prepare_detection(args.data_dir, args.output_dir, tuple(args.split), args.class_names)

    print("\n✅ Préparation terminée")


if __name__ == "__main__":
    main()