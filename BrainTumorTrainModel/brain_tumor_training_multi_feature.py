import os
import tensorflow as tf
import pandas as pd
from keras import Sequential, Model, Input
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation, Flatten, concatenate
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

    hidden_activation = "relu"
    output_activation = "sigmoid"
    img_size = 64
    batch_size = 128
    epochs = 256
    num_classes = 2
    color_mode = 'grayscale'  # One of "grayscale", "rgb", "rgba". The desired image format.
    if color_mode == 'grayscale':
        color_mode_index = 1
    elif color_mode == 'rgb':
        color_mode_index = 3
    elif color_mode == 'rgba':
        color_mode_index = 4
    else:
        raise Exception("color_mode must be one of 'grayscale', 'rgb', 'rgba'.")
    input_shape = (img_size, img_size, color_mode_index)
    print("input shape :", input_shape)

    ##
    # Load Dataset
    ##
    dataset_dir = "brain_tumor_dataset/"
    df = pd.read_csv("brain_tumor.csv")

    # Classes
    y = df['Class'].values
    y = to_categorical(y, num_classes=num_classes)

    # No/Yes labellarının oranını gösteren grafik
    plt.pie(df["Class"].value_counts(), labels=["No", "Yes"], normalize=True, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()

    # Features
    cols = ["Mean", "Variance", "Standard Deviation", "Entropy", "Skewness", "Kurtosis", "Contrast", "Energy",
            "ASM", "Homogeneity", "Dissimilarity", "Correlation", "Coarseness"]
    df_num = pd.read_csv("brain_tumor.csv", usecols=cols)

    # Min Max Normalization
    df_num_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min())
    print(df_num_norm.head())

    # MEAN - STD - ENERGY değer dağılımları
    plt.figure(figsize=(10, 7))
    sns.kdeplot(df_num_norm["Mean"], label="Mean")
    sns.kdeplot(df_num_norm["Standard Deviation"], label="Standard Deviation")
    sns.kdeplot(df_num_norm["Energy"], label="Energy")
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('MEAN - STD - ENERGY Values Range')
    plt.legend()
    plt.show()

    # Numerical Dataset Split : %80 train - %10 test - %10 validation
    x_num_train, x_num_rem, y_num_train, y_num_rem = train_test_split(df_num_norm, y, train_size=0.8,
                                                                      shuffle=False)
    x_num_valid, x_num_test, y_num_valid, y_num_test = train_test_split(x_num_rem, y_num_rem, test_size=0.5,
                                                                        shuffle=False)

    # Load Images
    train_image = []
    for i in tqdm(range(df.shape[0])):
        img = image.load_img(dataset_dir + df['Image'][i] + '.jpg',
                             target_size=input_shape, color_mode=color_mode)
        img = image.img_to_array(img)
        img = img / 255  # 0 - 1 (Normalization)
        train_image.append(img)
    images = np.array(train_image)
    print("Samples : ", len(images))

    # Image Dataset Split : %80 train - %10 test - %10 validation
    x_img_train, x_img_rem, y_img_train, y_img_rem = train_test_split(images, y, train_size=0.8, shuffle=False)
    x_img_valid, x_img_test, y_img_valid, y_img_test = train_test_split(x_img_rem, y_img_rem, test_size=0.5,
                                                                        shuffle=False)

    print(f"Train :{x_img_train.shape}")
    print(f"Test :{x_img_test.shape}")
    print(f"Validation :{x_img_valid.shape}")

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

    train_gen = train_datagen.flow(x_img_train, y_img_train, batch_size=batch_size, shuffle=False)

    other_datagen = ImageDataGenerator()
    test_gen = other_datagen.flow(x_img_test, y_img_test, batch_size=batch_size, shuffle=False)
    np_test_gen_y = np.argmax(test_gen.y, axis=1)
    valid_gen = other_datagen.flow(x_img_valid, y_img_valid, batch_size=batch_size, shuffle=False)

    ###
    # Show Samples
    ###
    def plot_img_sample(img_x, img_y, classes):
        col = 4
        row = 5
        fig = plt.figure(figsize=(col, row), dpi=100)
        fig.tight_layout()
        fig.subplots_adjust(hspace=1, wspace=1)
        for index in range(0, row * col):
            fig.add_subplot(row, col, index + 1)
            plt.imshow(img_x[index])
            plt.xlabel(classes[int(img_y[index][1])])
        plt.show()

    plot_img_sample(train_gen.x, train_gen.y, ["No", "Yes"])
    print()

    ##
    # Create Models
    ##
    def create_numerical_model(dim):
        # define our MLP network
        num_model = Sequential()
        num_model.add(Dense(8, input_dim=dim, activation=hidden_activation))
        num_model.add(Dense(4, activation=hidden_activation))
        # return our model
        return num_model

    def create_image_model(width, height, depth, filters=(16, 32, 64)):
        img_input_shape = (height, width, depth)
        # Input Layer
        inputs = Input(shape=img_input_shape)
        # loop over the number of filters
        x_img = 0
        for (index, f) in enumerate(filters):
            # if this is the first CONV layer then set the input
            # appropriately
            if index == 0:
                x_img = inputs
            # Convolution Layer
            x_img = Conv2D(f, (3, 3), padding="same")(x_img)
            x_img = Activation(hidden_activation)(x_img)
            x_img = BatchNormalization()(x_img)
            # Pooling Layer
            x_img = MaxPooling2D(pool_size=(2, 2))(x_img)
        # Fully Connected Layer
        x_img = Flatten()(x_img)
        x_img = Dense(16)(x_img)
        x_img = Activation(hidden_activation)(x_img)
        x_img = BatchNormalization()(x_img)
        x_img = Dropout(0.5)(x_img)
        # Fully Connected Layer
        x_img = Dense(4)(x_img)
        x_img = Activation(hidden_activation)(x_img)
        # construct the CNN
        img_model = Model(inputs, x_img)
        # return the CNN
        return img_model

    mlp = create_numerical_model(x_num_train.shape[1])
    cnn = create_image_model(input_shape[0], input_shape[1], input_shape[2], filters=(32, 64))

    # Concatenate of numerical and image model
    combined_input = concatenate([mlp.output, cnn.output])

    # Fully Connected Layer
    x = Dense(4, activation=hidden_activation)(combined_input)
    # Fully Connected Layer - Softmax
    x = Dense(num_classes, activation=output_activation)(x)

    model = Model(inputs=[mlp.input, cnn.input], outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-5),  # lr -> 0.00001
                  metrics=['accuracy'])

    print(model.summary())
    print("\n")
    print("Number of Layers:", len(model.layers))

    ##
    # Train Model
    ##
    history = model.fit(
        [x_num_train, train_gen.x], train_gen.y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_num_valid, valid_gen.x], valid_gen.y),
        verbose=1
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
    # Test
    ##
    y_pred = model.predict([x_num_test, test_gen.x])
    y_pred = np.argmax(y_pred, axis=1)

    ##
    # Confusion Matrix
    ##
    cf_matrix = confusion_matrix(np_test_gen_y, y_pred)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Greens')
    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['No', 'Yes'])
    ax.yaxis.set_ticklabels(['No', 'Yes'])
    plt.show()

    ##
    # Scores
    ##
    # Accuracy Score
    accuracy_res = accuracy_score(np_test_gen_y, y_pred, normalize=True, sample_weight=None)
    print("Accuracy Score : ", accuracy_res)
    # F1 Score
    f1_res = f1_score(np_test_gen_y, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None,
                      zero_division='warn')
    print("F1 Score : ", f1_res)
