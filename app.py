import cv2
import mediapipe as mp
import pickle
import numpy as np

with open("encodings.pkl", "rb") as f:
    encodings = pickle.load(f)

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

def normalize(vec):
    """Normaliza vetor para norma unitária."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
                embedding = normalize(coords.flatten())

                # Comparar com banco
                best_match = None
                best_dist = float("inf")

                for person, embeds in encodings.items():
                    for ref_emb in embeds:
                        ref_emb = normalize(ref_emb)
                        dist = np.linalg.norm(embedding - ref_emb)
                        if dist < best_dist:
                            best_dist = dist
                            best_match = person

                # Ajuste do limiar
                label = best_match if best_dist < 0.5 else "Desconhecido"

                # Desenhar landmarks
                h, w, _ = frame.shape
                for lm in landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                cv2.putText(frame, f"{label} ({best_dist:.4f})", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("Reconhecimento Facial com Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
