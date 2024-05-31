from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import matplotlib.pyplot as plt

train_data = ImageDataGenerator(rescale=1./255)
validation_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    'dataSet/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')
validation_generator = validation_data.flow_from_directory(
    'dataSet/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25)) 

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=125,
    validation_data=validation_generator,
    validation_steps=7178 // 64)
model_json = emotion_model.to_json()
with open("emotion_model.json", 'w') as json_file:
    json_file.write(model_json)
emotion_model.save_weights('emotion_model.weights.h5')

# Modelin eğitim ve doğruluk verilerini çıkar
train_accuracy = emotion_model_info.history['accuracy']
validation_accuracy = emotion_model_info.history['val_accuracy']
train_loss = emotion_model_info.history['loss']
validation_loss = emotion_model_info.history['val_loss']

epochs = range(1, len(train_accuracy) + 1)

plt.plot(epochs, train_accuracy, 'bo', label='Eğitim Doğruluğu')
plt.plot(epochs, validation_accuracy, 'b', label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epok')
plt.ylabel('Doğruluk')
plt.legend()

plt.figure()

plt.plot(epochs, train_loss, 'bo', label='Eğitim Kaybı')
plt.plot(epochs, validation_loss, 'b', label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epok')
plt.ylabel('Kayıp')
plt.legend()
plt.savefig('training_plot.png')
plt.show()

