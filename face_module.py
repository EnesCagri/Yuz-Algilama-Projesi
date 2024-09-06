import cv2 as cv
import mediapipe as mp

class FaceDetector:
    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        # Mediapipe yüz algılama modelini başlat
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=model_selection, 
                                                                        min_detection_confidence=min_detection_confidence)

    def yuzleri_algila(self, image):
        # Görüntüyü RGB formatına çevir
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Yüz algılama işlemi yap
        results = self.face_detection.process(image_rgb)
        faces = []
        if results.detections:
            # Görüntü boyutlarını al
            ih, iw, _ = image.shape
            # Algılanan her yüz için
            for detection in results.detections:
                # Yüzün göreli sınır kutusunu al
                bboxC = detection.location_data.relative_bounding_box
                # Sınır kutusunu piksel cinsine çevir
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                # Sınır kutusunu listeye ekle
                faces.append(bbox)
        return faces

    def yuz_say(self, image):
        # Görüntüdeki yüzleri algıla ve yüz sayısını döndür
        faces = self.yuzleri_algila(image)
        return len(faces)

    def yuz_ciz(self, image, faces, color=(0, 255, 0), thickness=2):
        # Algılanan yüzlerin etrafına dikdörtgen çiz
        for bbox in faces:
            cv.rectangle(image, bbox, color, thickness)
        return image
    
    def metin_goster(self, image, text, position, color=(98, 160, 3)):
        # Görüntüye belirli bir konumda metin ekle
        cv.putText(image, text, position, cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
    def bulaniklastir(self, image, face_bbox):
        # Sınır kutusundan yüz koordinatlarını al
        x, y, width, height = face_bbox
        
        # Koordinatların geçerli olup olmadığını kontrol et
        ih, iw, _ = image.shape
        if x < 0 or y < 0 or x + width > iw or y + height > ih:
            print("Yüz koordinatları geçersiz, görüntü dışında")
            return image

        # Yüz bölgesini çıkar ve bulanıklaştır
        face = image[y:y+height, x:x+width]
        
        if face.size == 0:
            print("Yüz bölgesi boş")
            return image

        # Bulanıklaştırma işlemi
        blurred_face = cv.GaussianBlur(face, (99, 99), 30)

        # Bulanıklaştırılmış yüzü orijinal görüntüye yerleştir
        image[y:y+height, x:x+width] = blurred_face

        return image
