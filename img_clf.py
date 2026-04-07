from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import SGD
import numpy as np
import csv
import os

IMG_WIDTH, IMG_HEIGHT = 200, 200
BATCH_SIZE = 32
EPOCHS = 15

train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
test_data_dir = '../data/test'
model_weights_path = 'final_weights.h5'


def build_model():
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        verbose=2)

    model.save_weights(model_weights_path)


def predict_labels(model):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode=None)

    predictions = model.predict_generator(
        test_generator,
        steps=len(test_generator),
        verbose=1)

    with open('prediction.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for filename, prediction in zip(test_generator.filenames, predictions):
            pic_id = os.path.splitext(os.path.basename(filename))[0]
            writer.writerow([pic_id, float(prediction[0])])


if __name__ == '__main__':
    model = build_model()
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)
        print('Loaded existing weights from', model_weights_path)
    else:
        train_model(model)
    predict_labels(model)
