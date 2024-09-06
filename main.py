import cv2 as cv
from face_module import FaceDetector

# FaceDetector sınıfını başlat
face_detector = FaceDetector()

# Kamerayı başlat
cap = cv.VideoCapture(0)

while True:
    # Kameradan görüntü okuma
    success, image = cap.read()
    
    if not success:
        # Eğer görüntü alınamıyorsa döngüden çık
        break
    
    # Görüntüyü yatayda çevir (ayna efekti)
    image = cv.flip(image, 1)
    
    # Görüntüdeki yüzleri algıla
    yuzler = face_detector.yuzleri_algila(image)
    
    if yuzler:  
        # Yüzlerin etrafına dikdörtgen çiz
        face_detector.yuz_ciz(image, yuzler)
        
    # 'q' tuşuna basıldığında programdan çık
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Görüntüyü ekranda göster
    cv.imshow("Yüz Algılayıcı", image)
    
# Kamerayı serbest bırak ve tüm pencereleri kapat
cap.release()
cv.destroyAllWindows()
