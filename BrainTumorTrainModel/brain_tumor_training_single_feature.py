import os
import tensorflow as tf
import pandas as pd
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation, AveragePooling2D, \
    GlobalAveragePooling2D
from keras.optimizer_v2.adam import Adam
from keras.utils.np_utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tqdm import tqdm
from keras.preprocessing import image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns


def train_run():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    img_size = 64
    batch_size = 128
    epochs = 1000
    num_classes = 2
    input_shape = (img_size, img_size, 1)
    color_mode = 'grayscale'  # One of "grayscale", "rgb", "rgba". The desired image format.

    ##
    # Load Dataset
    ##
    dataset_dir = "brain_tumor_dataset/"
    df = pd.read_csv("brain_tumor.csv")

    ##
    # Load Images
    ##
    train_image = []
    for i in tqdm(range(df.shape[0])):
        img = image.load_img(dataset_dir + df['Image'][i] + '.jpg',
                             target_size=input_shape, color_mode=color_mode)
        img = image.img_to_array(img)
        img = img / 255  # 0 - 1 (Normalization)
        train_image.append(img)
    x = np.array(train_image)
    print("Samples : ", len(x))

    y = df['Class'].values
    y = to_categorical(y)

    ##
    # Split (Train - Test - Validation)
    ##
    x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=0.9)
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_rem, y_rem, test_size=0.8)

    ##
    # Preprocessing
    ##
    train_datagen = ImageDataGenerator(height_shift_range=0.2,
                                       width_shift_range=0.2,
                                       zoom_range=0.2,
                                       brightness_range=(-0.8, 0.8),
                                       shear_range=0.2,
                                       fill_mode='nearest',
                                       horizontal_flip=True,
                                       rotation_range=40)

    train_gen = train_datagen.flow(x_train, y_train, batch_size=batch_size)

    other_datagen = ImageDataGenerator()
    test_gen = other_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)
    np_test_gen_y = np.argmax(test_gen.y, axis=1)
    valid_gen = other_datagen.flow(x_valid, y_valid, batch_size=batch_size)

    classes = df['Class'].astype('str').unique().tolist()
    labels = "\n".join(sorted(classes))
    open('labels/brain_tumor_labels.txt', 'w').write(labels)

    ##
    # Create Model
    ##
    model = Sequential()
    model.add(Conv2D(32, (5, 5), strides=(1, 1), name='conv0', input_shape=input_shape))

    model.add(BatchNormalization(axis=3, name='bn0'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), name='max_pool'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), name="conv1"))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3), name='avg_pool'))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(300, activation="relu", name='rl'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid', name='sm'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-5),
                  metrics=['accuracy'])

    ##
    # Train Model
    ##
    history = model.fit(
        train_gen.x, train_gen.y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(valid_gen.x, valid_gen.y),
        shuffle=True,
        verbose=1,
    )

    ##
    # Save Model
    ##
    saved_model_dir = 'models/brain_tumor_model/'
    tf.saved_model.save(model, saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    open(saved_model_dir + "brain_tumor_model.tflite", 'wb').write(tflite_model)

    ##
    # Accuracy
    ##
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.figure()
    ##
    # Loss
    ##
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.figure()

    ##
    # Confusion Matrix
    ##
    y_pred = model.predict(test_gen.x)
    y_pred = np.argmax(y_pred, axis=1)
    cf_matrix = confusion_matrix(np_test_gen_y, y_pred)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Greens')
    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['No', 'Yes'])
    ax.yaxis.set_ticklabels(['No', 'Yes'])
    plt.show()

    ##
    # F1 Score
    ##
    # Accuracy
    accuracy_res = accuracy_score(np_test_gen_y, y_pred, normalize=True, sample_weight=None)
    print("Accuracy Score : ", accuracy_res)
    # F1 Score
    f1_res = f1_score(np_test_gen_y, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None,
                      zero_division='warn')
    print("F1 Score : ", f1_res)
