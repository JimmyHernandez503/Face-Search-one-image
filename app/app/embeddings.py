import os
from typing import Optional, Tuple, List

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Modelo global cacheado
_MODEL: Optional[FaceAnalysis] = None


def _parse_det_size(val: str) -> Tuple[int, int]:
    """
    Parsea DET_SIZE desde el entorno con formato "ancho,alto".
    """
    try:
        a, b = val.split(",")
        return int(a), int(b)
    except Exception:
        return 640, 640


def get_face_app() -> FaceAnalysis:
    """
    Inicializa y devuelve el objeto FaceAnalysis (InsightFace).

    - Usa MODEL_NAME (por defecto: buffalo_l)
    - Usa DET_SIZE (por defecto: 640,640)
    - Usa /models como root por defecto para descargar los packs.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    root = os.environ.get("INSIGHTFACE_MODELS", "/models")
    # por defecto buffalo_l para evitar antelopev2 roto
    model_name = os.environ.get("MODEL_NAME", "buffalo_l")
    det_size = _parse_det_size(os.environ.get("DET_SIZE", "640,640"))

    app = FaceAnalysis(name=model_name, root=root, providers=providers)
    app.prepare(ctx_id=0, det_size=det_size)
    _MODEL = app
    return _MODEL


def _get_resize_params() -> Tuple[int, int]:
    """
    Lee MAX_SIDE y DOWNSCALE_TO. Si no están, usa 1600 / 1280.
    """
    try:
        max_side = int(os.environ.get("MAX_SIDE", "1600"))
        down_to = int(os.environ.get("DOWNSCALE_TO", "1280"))
    except Exception:
        max_side = 1600
        down_to = 1280
    return max_side, down_to


def read_image(path: str) -> Optional[np.ndarray]:
    """
    Lee una imagen desde disco (BGR) y la reescala si es muy grande.
    Usa imdecode(np.fromfile(...)) para soportar paths con Unicode.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = None

    if img is None:
        return None

    max_side, down_to = _get_resize_params()
    h, w = img.shape[:2]
    m = max(h, w)
    if m > max_side:
        scale = down_to / float(m)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def best_face_embedding(img_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Devuelve (embedding_512, bbox) del rostro más grande en la imagen, o None si no hay rostros.
    """
    app = get_face_app()
    faces = app.get(img_bgr)
    if not faces:
        return None

    # Ordenar por área del bounding box (rostro más grande primero)
    faces.sort(
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    f = faces[0]
    emb = f.normed_embedding  # 512-D ya normalizado
    return emb.astype(np.float32), f.bbox


def best_face_embedding_tta(
    img_bgr: np.ndarray,
    do_flip: bool = True,
    do_brightness: bool = True,
    do_rotate: bool = True,
    rot_deg: float = 15.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    TTA ligero para CONSULTAS (no usar en ingesta masiva):
      - original
      - flip horizontal
      - +/- brillo
      - +/- pequeña rotación

    Devuelve (embedding_512, bbox_principal) promediando todos los embeddings válidos.
    """
    variants: List[np.ndarray] = [img_bgr]

    if do_flip:
        variants.append(cv2.flip(img_bgr, 1))

    if do_brightness:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        for factor in (0.85, 1.15):
            hsv2 = hsv.copy()
            hsv2[..., 2] = np.clip(hsv2[..., 2] * factor, 0, 255).astype(np.uint8)
            variants.append(cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR))

    if do_rotate:
        h, w = img_bgr.shape[:2]
        center = (w // 2, h // 2)
        for angle in (-rot_deg, rot_deg):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                img_bgr,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            variants.append(rotated)

    embs: List[np.ndarray] = []
    bbox0: Optional[np.ndarray] = None
    for v in variants:
        out = best_face_embedding(v)
        if out is None:
            continue
        emb, bbox = out
        embs.append(emb)
        if bbox0 is None:
            bbox0 = bbox

    if not embs:
        return None

    emb_mean = np.mean(embs, axis=0)
    # Re-normalizamos porque cada emb ya venía normalizado
    norm = float(np.linalg.norm(emb_mean) + 1e-8)
    emb_mean = (emb_mean / norm).astype(np.float32)
    return emb_mean, bbox0  # type: ignore[return-value]


def embed_path(path: str) -> Optional[np.ndarray]:
    """
    Helper para ingesta: lee la imagen de disco, calcula el embedding del mejor rostro
    y devuelve únicamente el vector 512-D (o None si falla).
    """
    img = read_image(path)
    if img is None:
        return None
    out = best_face_embedding(img)
    if out is None:
        return None
    emb, _ = out
    return emb
