import numpy as np
import tensorflow as tf
import os
from PIL import Image


class Model:
    batch_size = 128
    epochs = 15
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    PATH = os.path.defpath
    checkpoint = 'checkpoints/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint,
                                                     save_weights_only=True,
                                                     verbose=1)
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Data set
    PATH = os.path.join(os.getcwd(), 'Data')
    PATH = os.path.join(PATH, 'cats_and_dogs_filtered')
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))
    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))
    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val
    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_dir,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary')

    def __init__(self):
        if self.latest is not None:
            self.core_load_cp()
        else:
            self.train()


    # Data set
    def train(self):
        self.model.fit(
            self.train_data_gen,
            steps_per_epoch=self.total_train // self.batch_size,
            epochs=self.epochs,
            validation_data=self.val_data_gen,
            validation_steps=self.total_val // self.batch_size,
            callbacks=[self.cp_callback]
        )

    def core_create(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)))

    def core_build(self):
        self.model.build()

    def core_load_cp(self):
        self.model.load_weights(self.latest)

    def core_recognize(self, path):
        file = Image.open(path).resize((150, 150))
        file = np.array(file)/255.0
        file.shape
        print(file)
        #result = self.model.predict(file[np.newaxis, ...])
        #result.shape
        #return result[0]

    def core_summary(self):
        slist = []
        self.model.summary(print_fn=lambda x: slist.append(x))
        return str(slist)

    def core_save(self, path):
        self.model.save(path)

    def core_load(self, path):
        self.model = tf.keras.models.load_model(path)

    def core_train(self):
        self.model.fit(
            self.train_data_gen,
            steps_per_epoch=self.total_train // self.batch_size,
            epochs=self.epochs,
            validation_data=self.val_data_gen,
            validation_steps=self.total_val // self.batch_size,
        )

    def core_training_set(self, train_dir):
        self.train_data_gen = self.train_image_generator.flow_from_directory(batch_size=self.batch_size,
                                                                             directory=train_dir,
                                                                             shuffle=True,
                                                                             target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                             class_mode='binary')

    def core_validation_set(self, val_dir):
        self.total_val = len(os.listdir(val_dir))
        self.val_data_gen = self.validation_image_generator.flow_from_directory(batch_size=self.batch_size,
                                                                                directory=val_dir,
                                                                                shuffle=True,
                                                                                target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                                class_mode='binary')

    def core_add_dense(self, n):
        self.model.add(tf.keras.layers.Dense(n, activation='relu'))

    def core_add_conv2d(self, n):
        self.model.add(tf.keras.layers.Conv2D(n, 3, padding='same', activation='relu'))

    def core_add_flattern(self):
        self.model.add(tf.keras.layers.Flatten())

    def core_add_dropout(self):
        self.model.add(tf.keras.layers.Dropout(0.2))

    def core_add_maxpooling2(self):
        self.model.add(tf.keras.layers.MaxPooling2D())

    def core_compile(self):
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])




