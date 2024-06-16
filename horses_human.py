import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.95:
            self.model.stop_training = True
            print("hai raggiunto una accuracy del 95%")


training_dir = "path\datasets\Horse or Human\Images\Training"
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode= 'binary'
)

validation_dir = "path\datasets\Horse or Human\Images\Validation"
validation_datagen = ImageDataGenerator(rescale=1/255.0)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode= 'binary'
    )

callback = myCallback()

model = keras.models.Sequential([
    keras.layers.Conv2D(16, (3,3), activation= 'relu', input_shape = (300, 300, 3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(32, (3,3), activation= 'relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation= 'relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation= 'relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation= 'relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss= 'binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    epochs = 15,
    validation_data= validation_generator,
    callbacks = callback
)

model.save('path\models\horse-human1')

print("insert path of the image you want to test")
path = input()
while path is not None:
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print(classes)
    print(classes[0])
    if classes[0]> 0.5:
        print("it's a human")
    else:
        print("it's a horse")
    print("insert path of the image you want to test")
    path = input()
