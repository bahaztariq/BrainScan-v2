"""GoogLeNet (Inception v1) implémenté en Keras 3 avec transfert des poids ImageNet.

Keras 3 (et ``tf_keras`` 2.21) ne fournissent plus ``keras.applications.googlenet`` :
l'architecture a été retirée des ``keras.applications`` (Keras >= 2.13) et les poids
``.h5`` historiques ne sont plus distribués.

Ce module reconstruit l'architecture GoogLeNet *à l'identique* de l'implémentation
``torchvision.models.googlenet`` et transfère les poids ImageNet pré-entraînés
(``IMAGENET1K_V1``) vers les couches Keras.

Écarts assumés par rapport à l'architecture décrite dans l'article original
(ils correspondent au checkpoint ImageNet de torchvision) :

* la branche « 5x5 » utilise en réalité des noyaux **3x3** (bug connu de torchvision,
  cf. https://github.com/pytorch/vision/issues/906) ;
* un **quatrième** max-pooling (``maxpool4``, noyau 2) est appliqué après
  ``inception4e`` (14x14 -> 7x7) ;
* les classifieurs auxiliaires (``aux1``/``aux2``) ne sont pas pré-entraînés et sont
  donc omis.

Choix d'implémentation (pour garantir l'équivalence numérique avec torchvision) :

* les convolutions à padding explicite (``padding=3/2/1``) sont rendues par un
  ``ZeroPadding2D`` suivi d'une convolution ``padding='valid'`` — le découpage
  des fenêtres est alors strictement identique à PyTorch ;
* les ``MaxPool2d(ceil_mode=True)`` de torchvision (112->56, 56->28, 28->14) sont
  rendus par ``ZeroPadding2D((0, 1))`` + ``MaxPooling2D(padding='valid')`` ;
* les ``BatchNorm2d(eps=1e-3)`` de torchvision sont repris tels quels dans les
  ``BatchNormalization`` de Keras (``epsilon=1e-3``) ;
* le padding zéro du ``MaxPool`` de torchvision (``-inf``) est sans effet sur des
  activations post-ReLU (>= 0).

Le module est utilisé par les notebooks de classification et réutilisable par le
pipeline d'entraînement (``brainscan.training.classification``) et l'API FastAPI.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import numpy as np
from tensorflow import keras
from keras import layers

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "GOOGLENET_CLASSES",
    "build_input_pipeline",
    "build_googlenet",
    "load_imagenet_weights",
    "convert_imagenet_weights_to_keras",
    "set_reproducible",
]

#: Moyenne/écart-type ImageNet (utilisés par ``torchvision`` avec ``transform_input=False``).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Correspondance entre les dossiers du dataset et les classes du projet.
GOOGLENET_CLASSES = ("glioma", "meningioma", "notumor", "pituitary")

#: (couche Keras, préfixe de l'attribut torchvision) pour la stem.
_STEM_CONVS = (
    ("conv1", "conv1"),
    ("conv2", "conv2"),
    ("conv3", "conv3"),
)

#: (nom du bloc, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj)
_INCEPTION_BLOCKS = (
    ("inception3a", 64, 96, 128, 16, 32, 32),
    ("inception3b", 128, 128, 192, 32, 96, 64),
    ("inception4a", 192, 96, 208, 16, 48, 64),
    ("inception4b", 160, 112, 224, 24, 64, 64),
    ("inception4c", 128, 128, 256, 24, 64, 64),
    ("inception4d", 112, 144, 288, 32, 64, 64),
    ("inception4e", 256, 160, 320, 32, 128, 128),
    ("inception5a", 256, 160, 320, 32, 128, 128),
    ("inception5b", 384, 192, 384, 48, 128, 128),
)


# --------------------------------------------------------------------------------------
# Couches de base
# --------------------------------------------------------------------------------------
def _zero_pad(x, padding: tuple[int, int, int, int]):
    """Padding explicite (haut, bas, gauche, droite) via ``ZeroPadding2D``."""
    top, bottom, left, right = padding
    return layers.ZeroPadding2D(padding=((top, bottom), (left, right)))(x)


def _conv_bn(
    x,
    filters: int,
    kernel_size: int,
    name: str,
    *,
    stride: int = 1,
    pad: int = 0,
):
    """Convolution (sans biais) + BatchNorm + ReLU.

    Le padding est explicite : ``ZeroPadding2D`` puis convolution ``valid``.
    """
    if pad:
        x = _zero_pad(x, (pad, pad, pad, pad))
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="valid",
        use_bias=False,
        name=name,
    )(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.99, name=f"{name}_bn")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def _inception(
    x,
    name: str,
    ch1x1: int,
    ch3x3red: int,
    ch3x3: int,
    ch5x5red: int,
    ch5x5: int,
    pool_proj: int,
):
    """Bloc Inception (4 branches) avec l'ordre de concaténation de torchvision.

    ``branch1`` (1x1), ``branch2`` (1x1 -> 3x3), ``branch3`` (1x1 -> 3x3, « 5x5 »
    historique) et ``branch4`` (max-pool -> 1x1).
    """
    branch1x1 = _conv_bn(x, ch1x1, 1, f"{name}_branch1x1")

    branch3x3dbl_reduced = _conv_bn(x, ch3x3red, 1, f"{name}_branch3x3dbl_reduced")
    branch3x3dbl = _conv_bn(branch3x3dbl_reduced, ch3x3, 3, f"{name}_branch3x3dbl", pad=1)

    branch5x5_reduced = _conv_bn(x, ch5x5red, 1, f"{name}_branch5x5_reduced")
    branch5x5 = _conv_bn(branch5x5_reduced, ch5x5, 3, f"{name}_branch5x5", pad=1)

    # torchvision : nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
    pooled = _zero_pad(x, (1, 1, 1, 1))
    pooled = layers.MaxPooling2D(pool_size=3, strides=1, padding="valid", name=f"{name}_branch_pool_pool")(pooled)
    branch_pool = _conv_bn(pooled, pool_proj, 1, f"{name}_branch_pool")

    return layers.Concatenate(name=f"{name}_concat")([branch1x1, branch3x3dbl, branch5x5, branch_pool])


def _maxpool_ceil_mode(x, name: str, pool_size: int = 3, extra_pad: int = 1):
    """``nn.MaxPool2d(pool_size, stride=2, ceil_mode=True)`` de torchvision.

    ``ceil_mode=True`` arrondit la taille de sortie à l'unité supérieure ; pour un
    noyau de côté impair et un pas de 2, on l'obtient en ajoutant une ligne/colonne
    de padding (``extra_pad=1``) avant un pooling ``valid``.
    """
    if extra_pad:
        x = _zero_pad(x, (0, extra_pad, 0, extra_pad))
    return layers.MaxPooling2D(pool_size=pool_size, strides=2, padding="valid", name=name)(x)


# --------------------------------------------------------------------------------------
# Pipeline d'entrée (prétraitement + augmentation)
# --------------------------------------------------------------------------------------
def build_input_pipeline(
    image_size: int = 224,
    *,
    augment: bool = False,
    blur_sigma: tuple[float, float] = (0.1, 1.5),
    blur_factor: float = 0.5,
    noise_stddev: float = 0.05,
    name: str = "input_pipeline",
) -> keras.Sequential:
    """Construit le pipeline d'entrée Keras du classifieur.

    Ordre des opérations :

    1. ``Resizing`` — redimensionnement à la taille attendue par le backbone ;
    2. ``RandomFlip`` / ``RandomGaussianBlur`` — **augmentation** (flou, retournement) ;
    3. ``Rescaling(1/255)`` — conversion au format ``float32`` normalisé ``[0, 1]`` ;
    4. ``GaussianNoise`` — **augmentation** (bruit gaussien) ;
    5. ``Normalization`` — centrage/réduction selon les statistiques ImageNet.

    Les couches d'augmentation de Keras ne s'activent que lorsque ``training=True``.
    Keras utilise ``training=False`` pendant ``evaluate()`` et ``predict()`` : les
    ensembles de validation et de test ne sont donc **jamais** augmentés, alors que
    l'ensemble d'entraînement l'est systématiquement.

    Parameters
    ----------
    image_size:
        Taille (côté) des images attendues par le backbone.
    augment:
        Active ou désactive les augmentations.
    blur_sigma:
        Bornes du sigma du flou gaussien.
    blur_factor:
        Probabilité d'application du flou.
    noise_stddev:
        Écart-type du bruit gaussien (sur des valeurs dans ``[0, 1]``).

    Returns
    -------
    keras.Sequential
    """
    layers_list: list[Any] = [layers.Resizing(image_size, image_size)]

    if augment:
        layers_list.append(layers.RandomFlip("horizontal"))
        layers_list.append(
            layers.RandomGaussianBlur(
                kernel_size=3,
                sigma=blur_sigma,
                factor=blur_factor,
                value_range=(0.0, 255.0),
            )
        )

    layers_list.append(layers.Rescaling(1.0 / 255.0))

    if augment:
        layers_list.append(layers.GaussianNoise(stddev=noise_stddev))

    layers_list.append(layers.Normalization(mean=IMAGENET_MEAN, variance=IMAGENET_STD))
    return keras.Sequential(layers_list, name=name)


# --------------------------------------------------------------------------------------
# Construction du modèle
# --------------------------------------------------------------------------------------
def build_googlenet(
    num_classes: int = 4,
    class_names: Iterable[str] | None = None,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    *,
    image_size: int = 224,
    weights: str | None = "imagenet",
    freeze_backbone: bool = True,
    dense_units: int = 128,
    dropout: float = 0.5,
    include_preprocessing: bool = True,
    augment: bool = False,
    name: str = "brainscan_googlenet",
):
    """Construit le classifieur GoogLeNet adapté à la classification de tumeurs.

    Parameters
    ----------
    num_classes:
        Nombre de classes de sortie.
    class_names:
        Noms des classes (métadonnées du modèle).
    input_shape:
        Forme d'entrée ``(H, W, C)``.
    weights:
        ``"imagenet"`` pour transférer les poids pré-entraînés, ``None`` pour un
        modèle entraîné de zéro.
    freeze_backbone:
        Si ``True``, les couches convolutionnelles sont gelées (transfer learning
        « feature extraction »). Passer à ``False`` pour un fine-tuning complet.
    dense_units:
        Largeur de la couche dense cachée du classifieur.
    dropout:
        Taux de dropout appliqué avant la couche de sortie.
    include_preprocessing:
        Si ``True``, le prétraitement (redimensionnement + normalisation) est
        intégré au modèle : celui-ci accepte alors directement des images
        ``uint8`` ou ``float32`` en ``[0, 255]``.
    augment:
        Si ``True``, les couches d'augmentation (flip, flou, bruit) sont intégrées
        au modèle. Elles ne s'activent qu'en mode entraînement (``training=True``),
        ce qui laisse les ensembles de validation et de test inchangés.

    Returns
    -------
    keras.Model
    """
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    class_names = list(class_names)

    inp = keras.Input(shape=input_shape, name="input")
    x = inp

    if include_preprocessing:
        x = build_input_pipeline(image_size=image_size, augment=augment)(x)

    # --- Stem -----------------------------------------------------------------------
    x = _conv_bn(x, 64, 7, "conv1", stride=2, pad=3)
    x = _maxpool_ceil_mode(x, "maxpool1")
    x = _conv_bn(x, 64, 1, "conv2")
    x = _conv_bn(x, 192, 3, "conv3", pad=1)
    x = _maxpool_ceil_mode(x, "maxpool2")

    # --- Inception blocks ------------------------------------------------------------
    for block in _INCEPTION_BLOCKS:
        x = _inception(x, *block)
        if block[0] == "inception3b":
            x = _maxpool_ceil_mode(x, "maxpool3")
        if block[0] == "inception4e":
            x = _maxpool_ceil_mode(x, "maxpool4", pool_size=2, extra_pad=0)

    # --- Classifieur -----------------------------------------------------------------
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense_units, activation="relu", name="classifier_dense")(x)
    x = layers.Dropout(dropout, name="classifier_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier_output")(x)

    model = keras.Model(inp, outputs, name=name)

    if weights == "imagenet":
        load_imagenet_weights(model)

    if freeze_backbone:
        for layer in model.layers:
            if layer.name.startswith(("conv", "inception", "maxpool")):
                layer.trainable = False

    model.class_names = class_names  # type: ignore[attr-defined]
    model.image_size = image_size  # type: ignore[attr-defined]
    return model


# --------------------------------------------------------------------------------------
# Transfert des poids ImageNet
# --------------------------------------------------------------------------------------
def _torch_state_dict() -> dict[str, Any]:
    """Charge le ``state_dict`` torchvision ``googlenet(IMAGENET1K_V1)``."""
    import torch  # noqa: PLC0415  (import tardif : torch n'est pas requis pour l'inférence)
    from torchvision.models import googlenet  # noqa: PLC0415

    net = googlenet(weights="IMAGENET1K_V1", transform_input=False, aux_logits=True, init_weights=False)
    net.eval()
    return net.state_dict()


def _conv_weight(torch_w: np.ndarray) -> np.ndarray:
    """(out, in, kh, kw) -> (kh, kw, in, out) attendu par Keras."""
    return np.transpose(torch_w, (2, 3, 1, 0))


def _iter_conv_bn_pairs() -> list[tuple[str, str]]:
    """Liste des couples (couche Keras, chemin torchvision) des convolution+BN."""
    pairs: list[tuple[str, str]] = list(_STEM_CONVS)
    for block in _INCEPTION_BLOCKS:
        name = block[0]
        pairs.extend(
            [
                (f"{name}_branch1x1", f"{name}.branch1"),
                (f"{name}_branch3x3dbl_reduced", f"{name}.branch2.0"),
                (f"{name}_branch3x3dbl", f"{name}.branch2.1"),
                (f"{name}_branch5x5_reduced", f"{name}.branch3.0"),
                (f"{name}_branch5x5", f"{name}.branch3.1"),
                (f"{name}_branch_pool", f"{name}.branch4.1"),
            ]
        )
    return pairs


def convert_imagenet_weights_to_keras(state_dict: dict[str, Any], model: keras.Model) -> list[str]:
    """Copie les poids torchvision dans les couches Keras correspondantes.

    Returns
    -------
    list[str]
        Noms des couches effectivement transférées.
    """
    by_name = {layer.name: layer for layer in model.layers}
    transferred: list[str] = []

    for keras_name, torch_prefix in _iter_conv_bn_pairs():
        conv_layer = by_name.get(keras_name)
        bn_layer = by_name.get(f"{keras_name}_bn")
        if conv_layer is None or bn_layer is None:
            continue

        torch_conv = state_dict.get(f"{torch_prefix}.conv.weight")
        torch_bn_w = state_dict.get(f"{torch_prefix}.bn.weight")
        torch_bn_b = state_dict.get(f"{torch_prefix}.bn.bias")
        torch_mean = state_dict.get(f"{torch_prefix}.bn.running_mean")
        torch_var = state_dict.get(f"{torch_prefix}.bn.running_var")
        if torch_conv is None or torch_mean is None or torch_var is None:
            continue

        conv_layer.set_weights([_conv_weight(torch_conv.detach().cpu().numpy())])
        bn_layer.set_weights(
            [
                torch_bn_w.detach().cpu().numpy(),
                torch_bn_b.detach().cpu().numpy(),
                torch_mean.detach().cpu().numpy(),
                torch_var.detach().cpu().numpy(),
            ]
        )
        transferred.append(keras_name)

    return transferred


def load_imagenet_weights(model: keras.Model, verbose: bool = True) -> list[str]:
    """Transfère les poids ImageNet de ``torchvision`` vers un modèle Keras construit
    par :func:`build_googlenet`.

    Les poids sont mis en cache dans ``~/.cache/torch`` (variable ``TORCH_HOME``).
    """
    state_dict = _torch_state_dict()
    transferred = convert_imagenet_weights_to_keras(state_dict, model)
    if verbose:
        total = sum(int(np.prod(w.shape)) for w in model.get_weights())
        print(
            f"[googlenet] {len(transferred)} couches transférées depuis "
            f"torchvision (IMAGENET1K_V1) - {total:,} paramètres au total."
        )
    return transferred


def set_reproducible(seed: int = 42) -> None:
    """Fixe les graines aléatoires (Python / NumPy / TensorFlow)."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    import random  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf  # noqa: PLC0415

    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover - indisponible sur certaines builds
        pass
