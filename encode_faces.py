import cv2
import mediapipe as mp
import os
import pickle
import numpy as np

# Inicializa o detector de rostos do Mediapipe
mp_face_detection = mp.solutions.face_detection

# Onde estão as pastas com fotos
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"

def extract_face_embedding(image_path):
    """Extrai embedding normalizado baseado nos pixels do rosto."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        results = face_detection.process(rgb)
        if not results.detections:
            return None

        detection = results.detections[0]
        box = detection.location_data.relative_bounding_box

        h, w, _ = image.shape
        x, y, w_box, h_box = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
        face = rgb[y:y+h_box, x:x+w_box]

        if face.size == 0:
            return None

        resized = cv2.resize(face, (128, 128)).flatten().astype("float32")

        # Normalização (0–1)
        return resized / np.linalg.norm(resized)

def main():
    encodings = {}

    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            face_embedding = extract_face_embedding(img_path)
            if face_embedding is not None:
                embeddings.append(face_embedding)

        if embeddings:
            encodings[person] = embeddings
            print(f"[INFO] {person}: {len(embeddings)} imagens processadas.")

    # Salva no arquivo
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(encodings, f)

    print(f"[INFO] Encodings salvos em {ENCODINGS_FILE}")

if __name__ == "__main__":
    main()
