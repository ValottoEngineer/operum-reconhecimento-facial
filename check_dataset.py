import os
import cv2
import mediapipe as mp

dataset_dir = "dataset"
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(min_detection_confidence=0.5)

for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_dir):
        continue

    print(f"\n📂 Verificando pasta: {person}")

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ ERRO ao abrir {img_name}")
            continue

        results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.detections:
            print(f"✅ {img_name} -> rosto detectado")
        else:
            print(f"⚠️ {img_name} -> nenhum rosto detectado")
