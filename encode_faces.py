import os
import cv2
import mediapipe as mp
from deepface import DeepFace
import pickle

dataset_dir = "dataset"
encodings = {}

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(min_detection_confidence=0.5)

for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_dir):
        continue

    encodings[person] = []

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Detecta rosto com Mediapipe
        results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.detections:
            try:
                # Extrai embedding com DeepFace (modelo leve)
                embedding = DeepFace.represent(img_path=img_path, model_name="Facenet")[0]["embedding"]
                encodings[person].append(embedding)
                print(f"[OK] {img_name} -> {person}")
            except Exception as e:
                print(f"[ERRO] {img_name}: {e}")

# Salva banco de embeddings
with open("faces.pkl", "wb") as f:
    pickle.dump(encodings, f)

print("✅ Encodings salvos em faces.pkl")
