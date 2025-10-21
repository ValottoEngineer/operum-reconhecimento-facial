# app.py
import cv2
import mediapipe as mp
import pickle
import numpy as np
from numpy.linalg import norm
import requests

# URL do Flask bridge
APPROVE_URL = "http://127.0.0.1:5001/approve"
REJECT_URL = "http://127.0.0.1:5001/reject"

# Carregar banco de embeddings
with open("encodings.pkl", "rb") as f:
    encodings = pickle.load(f)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

# flag para não aprovar repetidamente
already_sent_approve = False

THRESHOLD = 0.40  # ajuste fino se necessário

with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as detector, \
     mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = max(0, int(bboxC.xmin * w))
                y = max(0, int(bboxC.ymin * h))
                w_box = int(bboxC.width * w)
                h_box = int(bboxC.height * h)

                face_img = rgb[y:y+h_box, x:x+w_box]
                if face_img.size == 0:
                    continue

                try:
                    embedding = cv2.resize(face_img, (128, 128)).flatten().astype("float32")
                    embedding = embedding / (norm(embedding) + 1e-8)

                    best_match = None
                    best_dist = 1.0

                    for person, embeds in encodings.items():
                        for ref_emb in embeds:
                            dist = 1 - np.dot(embedding, ref_emb) / (norm(embedding)*norm(ref_emb) + 1e-8)
                            if dist < best_dist:
                                best_dist = dist
                                best_match = person

                    label = best_match if best_dist < THRESHOLD else "Desconhecido"

                    # desenha
                    cv2.putText(frame, f"{label} ({best_dist:.4f})", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (255, 0, 0), 2)

                    # landmarks
                    mesh_results = face_mesh.process(rgb)
                    if mesh_results.multi_face_landmarks:
                        for face_landmarks in mesh_results.multi_face_landmarks:
                            for lm in face_landmarks.landmark:
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

                    # >>> APROVAÇÃO <<<
                    if (label != "Desconhecido") and (best_dist < THRESHOLD) and (not already_sent_approve):
                        try:
                            requests.post(APPROVE_URL, timeout=2)
                            already_sent_approve = True
                            print("[APPROVE] Enviado para o servidor.")
                        except Exception as e:
                            print("[APPROVE] Falha ao enviar:", e)

                except Exception as e:
                    print("Erro:", e)

        cv2.imshow("Reconhecimento Facial (com landmarks)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para sair
            # opcional: rejeitar ao sair
            try:
                requests.post(REJECT_URL, timeout=1)
            except:
                pass
            break

cap.release()
cv2.destroyAllWindows()
