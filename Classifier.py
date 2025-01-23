import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras.applications import DenseNet121
from keras.layers import Input, Dense, Dropout, concatenate, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt


img_size = 256
types = ['B', 'Mgas', 'MgFe', 'Vgas', 'T', 'Mcdm']
num_types = len(types)

X_img = []  
X_params = []  
y = []  

for type_index, type_name in enumerate(types):
    folder = os.path.join(r'.\MapImages', type_name)
    params_df = pd.read_csv(os.path.join(folder, 'map_parameters.csv'))

    for index, row in params_df.iterrows():
        img_path = os.path.join(folder, f"Map_{int(row['Map_ID'])}.png")
        img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
        img = np.array(img)

        X_img.append(img)
        X_params.append(row[['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_SN2', 'A_AGN2']].values)
        y.append(type_index)

X_img = np.array(X_img) / 255.0
X_params = np.array(X_params)
y = to_categorical(y, num_types)

X_img_train, X_img_test, X_params_train, X_params_test, y_train, y_test = train_test_split(X_img, X_params, y, test_size=0.2, random_state=42)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])

def create_model(input_shape_img, input_shape_params, num_types):
    base_model = DenseNet121(include_top=False, input_shape=input_shape_img, weights='imagenet', pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False

    img_input = Input(shape=input_shape_img)
    x = data_augmentation(img_input)
    x = base_model(x)

    params_input = Input(shape=input_shape_params)
    y = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(params_input)

    combined = concatenate([x, y])
    z = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined)
    z = Dropout(0.25)(z)
    z = Dense(num_types, activation='softmax')(z)

    model = Model(inputs=[img_input, params_input], outputs=z)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = create_model(X_img_train[0].shape, X_params_train[0].shape, y_train.shape[1])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit([X_img_train, X_params_train], y_train, epochs=20, batch_size=32, validation_data=([X_img_test, X_params_test], y_test), callbacks=[early_stopping])

test_loss, test_acc = model.evaluate([X_img_test, X_params_test], y_test)
print(f'Test accuracy: {test_acc}')

def plot_history(history):
    fig, axs = plt.subplots(2, figsize=(10, 10))
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Model accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Test'], loc='upper left')
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Test'], loc='upper left')
    plt.show()

plot_history(history)

def load_and_preprocess_single_image(image_path, img_size):
    img = Image.open(image_path).convert('RGB').resize((img_size, img_size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(model, img_size, types):
    root = tk.Tk()
    root.withdraw()
    while True:
        image_path = filedialog.askopenfilename(title='Select an image to classify')
        if not image_path:
            break
        img = load_and_preprocess_single_image(image_path, img_size)
        prediction = model.predict([img, np.zeros((1, 6))])
        predicted_category = types[np.argmax(prediction)]
        print(f"The model predicts this image is of type: {predicted_category}")

predict_image(model, img_size, types)

