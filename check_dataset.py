# check_dataset.py
import cv2
import os
import face_recognition

dataset_dir = "dataset"

def check_images():
    print("[INFO] Iniciando validação do dataset...\n")
    for person in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_path):
            continue
        
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"[ERRO] Não foi possível abrir a imagem: {image_path}")
                continue

            h, w = image.shape[:2]

            # Verifica resolução mínima
            if h < 150 or w < 150:
                print(f"[AVISO] Imagem muito pequena ({w}x{h}): {image_path}")

            # Converte para RGB (garante compatibilidade)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detecta rostos na imagem
            boxes = face_recognition.face_locations(rgb, model="hog")

            if len(boxes) == 0:
                print(f"[ERRO] Nenhum rosto detectado em: {image_path}")
            else:
                print(f"[OK] {image_path} - {len(boxes)} rosto(s) detectado(s).")

    print("\n[INFO] Validação concluída!")

if __name__ == "__main__":
    check_images()
