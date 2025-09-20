import cv2
import mediapipe as mp
import os
import pickle
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"

def extract_face_embeddings(image_path):
    """Extrai embeddings normalizados a partir dos landmarks do FaceMesh."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        # Pegamos os landmarks do rosto detectado
        landmarks = results.multi_face_landmarks[0]
        coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark])  # normalizado [0,1]
        return coords.flatten()

def main():
    encodings = {}

    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            face_embedding = extract_face_embeddings(img_path)
            if face_embedding is not None:
                embeddings.append(face_embedding)

        if embeddings:
            # salva a média para maior estabilidade
            encodings[person] = [np.mean(embeddings, axis=0)]
            print(f"[INFO] {person}: {len(embeddings)} imagens processadas.")

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(encodings, f)

    print(f"[INFO] Encodings salvos em {ENCODINGS_FILE}")

if __name__ == "__main__":
    main()
