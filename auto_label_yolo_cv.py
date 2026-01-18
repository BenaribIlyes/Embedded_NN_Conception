
"""
Auto-label ASL alphabet hands with YOLOv8 + MediaPipe Hands
Génère un fichier bounding_boxes.labels dans chaque sous-dossier (A, B)
compatible Edge Impulse (object detection).
"""
# cd /Users/mohammedaminebendaou/Downloads/
# source env_asl/bin/activate 
# python auto_label_yolo_cv.py 
import os
import json
import cv2
from ultralytics import YOLO
import mediapipe as mp

# ========= PARAMÈTRES GÉNÉRAUX =========
BASE_FOLDER = "/Users/mohammedaminebendaou/Downloads/Data" #ASL_Alphabet_Dataset"
SUBFOLDERS = ["A","B","C","D","E","F","G","H","I","K"]#,"L","M","N","O","P","Q","R","S","T","U","V","W","X"] # classes / sous-dossiers
DEFAULT_CATEGORY = "split"       # non utilisé par Edge Impulse dans ce fichier

# Réglages YOLO (rappel élevé, petites mains)
YOLO_MODEL_PATH = "/Users/mohammedaminebendaou/Downloads/Projet_NN_Embarqués/env_asl/yolov8n.pt"   # ou un modèle plus gros si tu veux
CONF_PRIMARY   = 0.20            # seuil principal plus bas
CONF_FALLBACK  = 0.10            # second passage très permissif
IOU_NMS        = 0.40            # NMS moins agressif pour ne pas louper des mains
MAX_DET        = 2               # max 2 mains par image

# Réglages boîtes
MIN_REL_AREA   = 0.008           # surface min ~0.8% de l'image
MP_MARGIN      = 0.15            # marge autour de la main (15%)

# =======================================

# Charger YOLO
yolo = YOLO(YOLO_MODEL_PATH)

# MediaPipe Hands pour refinement / fallback
mp_hands = mp.solutions.hands
mp_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.30
)


def clamp_box(x1, y1, x2, y2, W, H):
    """Force la boîte dans les bornes de l'image."""
    x1 = max(0.0, min(float(x1), float(W - 1)))
    y1 = max(0.0, min(float(y1), float(H - 1)))
    x2 = max(0.0, min(float(x2), float(W - 1)))
    y2 = max(0.0, min(float(y2), float(H - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def detect_with_yolo(img_path, W, H):
    """Renvoie une liste de boîtes (x1, y1, w, h) avec YOLO, ou [] si rien."""
    # Passage 1
    r = yolo(
        img_path,
        conf=CONF_PRIMARY,
        iou=IOU_NMS,
        max_det=MAX_DET,
        agnostic_nms=True
    )
    boxes = r[0].boxes

    # Passage 2 si vide
    if boxes is None or len(boxes) == 0:
        r = yolo(
            img_path,
            conf=CONF_FALLBACK,
            iou=IOU_NMS,
            max_det=MAX_DET,
            agnostic_nms=True
        )
        boxes = r[0].boxes

    dets = []
    if boxes is None:
        return dets

    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
        w, h = x2 - x1, y2 - y1
        if w * h >= MIN_REL_AREA * W * H:
            dets.append((x1, y1, w, h))
    return dets


def refine_with_mediapipe(img_bgr, roi=None):
    """
    Utilise MediaPipe pour serrer la boîte autour de la main.
    - roi : (x1, y1, w, h) en pixels (optionnel). Si fourni, MediaPipe travaille
      dans cette région et renvoie des boîtes remappées dans le repère global.
    """
    H, W = img_bgr.shape[:2]

    if roi is not None:
        rx1, ry1, rw, rh = roi
        rx1, ry1, rx2, ry2 = clamp_box(rx1, ry1, rx1 + rw, ry1 + rh, W, H)
        crop = img_bgr[int(ry1):int(ry2), int(rx1):int(rx2)]
        if crop.size == 0:
            return []
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = mp_detector.process(rgb)
        base_x, base_y = rx1, ry1
        crop_W, crop_H = crop.shape[1], crop.shape[0]
    else:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = mp_detector.process(rgb)
        base_x, base_y = 0.0, 0.0
        crop_W, crop_H = W, H

    dets = []
    if res and res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            xs = [p.x * crop_W for p in lm.landmark]
            ys = [p.y * crop_H for p in lm.landmark]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)

            # Ajoute une petite marge pour ne pas couper les doigts
            mx = MP_MARGIN * (x2 - x1)
            my = MP_MARGIN * (y2 - y1)
            gx1, gy1, gx2, gy2 = clamp_box(
                base_x + x1 - mx,
                base_y + y1 - my,
                base_x + x2 + mx,
                base_y + y2 + my,
                W, H
            )
            w, h = gx2 - gx1, gy2 - gy1
            if w * h >= MIN_REL_AREA * W * H:
                dets.append((gx1, gy1, w, h))

    return dets


for sub in SUBFOLDERS:
    folder = os.path.join(BASE_FOLDER, sub)
    if not os.path.isdir(folder):
        print(f"⚠️ Dossier introuvable: {folder}")
        continue

    print(f"📂 Traitement du dossier: {folder}")

    bb_map = {}
    total, with_box = 0, 0

    for img_name in os.listdir(folder):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        total += 1

        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        # 1) YOLO pour proposer 0–MAX_DET ROI
        yolo_dets = detect_with_yolo(img_path, W, H)

        # 2) MediaPipe sur chaque ROI pour serrer la boîte
        final_dets = []
        if yolo_dets:
            for roi in yolo_dets:
                refined = refine_with_mediapipe(img, roi=roi)
                if refined:
                    final_dets.extend(refined)
                else:
                    # fallback: garder la boîte YOLO si MP ne voit rien
                    final_dets.append(roi)

        # 3) Fallback global MediaPipe si encore aucune boîte
        if not final_dets:
            mp_dets = refine_with_mediapipe(img, roi=None)
            final_dets.extend(mp_dets)

        # 4) Option: limiter le nombre final de boîtes par image (ex. 2)
        if final_dets:
            final_dets = final_dets[:MAX_DET]
            with_box += 1
            bb_map[img_name] = [
                {
                    "label": sub,  # classe OD = nom du sous-dossier (A ou B)
                    "x": float(x),
                    "y": float(y),
                    "width": float(w),
                    "height": float(h)
                }
                for (x, y, w, h) in final_dets
            ]

    # Écriture du bounding_boxes.labels pour ce dossier
    out_path = os.path.join(folder, "bounding_boxes.labels")
    with open(out_path, "w") as f:
        json.dump(
            {
                "version": 1,
                "type": "bounding-box-labels",
                "boundingBoxes": bb_map
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"✅ {sub}: {with_box}/{total} images avec boîtes -> {out_path}")
