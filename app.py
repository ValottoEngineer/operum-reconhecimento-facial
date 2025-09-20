import cv2
import mediapipe as mp
from deepface import DeepFace
import pickle
import numpy as np

# Carregar banco de embeddings
with open("faces.pkl", "rb") as f:
    encodings = pickle.load(f)

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

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
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                                 int(bboxC.width * w), int(bboxC.height * h)

            face_img = frame[y:y+h_box, x:x+w_box]

            try:
                embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]

                # Comparar com banco
                best_match = None
                best_dist = 100

                for person, embeds in encodings.items():
                    for ref_emb in embeds:
                        dist = np.linalg.norm(np.array(embedding) - np.array(ref_emb))
                        if dist < best_dist:
                            best_dist = dist
                            best_match = person

                label = best_match if best_dist < 10 else "Desconhecido"
                cv2.putText(frame, f"{label} ({best_dist:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (255, 0, 0), 2)

            except Exception as e:
                print("Erro:", e)

    cv2.imshow("Reconhecimento Facial", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
