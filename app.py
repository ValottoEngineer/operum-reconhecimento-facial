import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque, Counter

# Carregar embeddings salvos
with open("encodings.pkl", "rb") as f:
    encodings = pickle.load(f)

# Inicializa detector do MediaPipe
mp_face_detection = mp.solutions.face_detection

# Configura webcam
cap = cv2.VideoCapture(0)

# Histórico das últimas previsões para suavização
history = deque(maxlen=10)

# Threshold para considerar a mesma pessoa
THRESHOLD = 0.6

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        label = "Desconhecido"  # valor padrão

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                face_img = rgb[y:y+h_box, x:x+w_box]

                if face_img.size == 0:
                    continue

                # Redimensiona e gera embedding simples
                embedding = cv2.resize(face_img, (128, 128)).flatten()

                best_match = None
                best_dist = float("inf")

                # Compara com todas as pessoas
                for person, embeds in encodings.items():
                    for ref_emb in embeds:
                        dist = np.linalg.norm(embedding - ref_emb)
                        if dist < best_dist:
                            best_dist = dist
                            best_match = person

                if best_match and best_dist < THRESHOLD:
                    history.append(best_match)
                else:
                    history.append("Desconhecido")

                # Voto majoritário
                label = Counter(history).most_common(1)[0][0]

                # Caixa e nome na tela
                cv2.putText(frame, f"{label}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (255, 0, 0), 2)

        cv2.imshow("Reconhecimento Facial", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
            break

cap.release()
cv2.destroyAllWindows()
