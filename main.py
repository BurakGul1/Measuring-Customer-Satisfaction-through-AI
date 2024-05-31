import numpy as np
import cv2
from keras.models import model_from_json
from transformers import pipeline
import time

# Duygu etiketleri
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"} 

# Hugging Face Transformers Sentiment Analysis modelini yükleme
nlp = pipeline("sentiment-analysis")

json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights('model/emotion_model.weights.h5')
print("Loaded model from disk")

# Kamera başlatma
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare oku
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (500, 500))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y+h, x:x+w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # Duygu tahmini
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            result = nlp(emotion_dict[maxindex])
            sentiment_label = result[0]['label']
            
            if sentiment_label == "POSITIVE":
                customer_satisfaction_sentence = "Müşteri memnun kaldı."
            elif sentiment_label == "NEGATIVE":
                customer_satisfaction_sentence = "Müşteri memnun kalmadı."
            else:
                customer_satisfaction_sentence = "Müşteri memnuniyeti belirsiz."
            
            print("Müşteri memnuniyeti cümlesi:", customer_satisfaction_sentence)

        cv2.imshow('Emotion Detection', frame)
    
    # bekle
    time.sleep(1)

    # Klavyeden 'q' tuşuna basılırsa döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı kapat
cap.release()
cv2.destroyAllWindows()
