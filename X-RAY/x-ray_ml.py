# Library
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix,roc_curve, auc
import matplotlib.pyplot as plt

# Augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)
train =train_gen.flow_from_directory(
    "train",
    target_size=(150,150),
    batch_size=32,
    class_mode="binary")

test_gen = ImageDataGenerator(rescale=1./255)
test =test_gen.flow_from_directory(
    "test",
    target_size=(150,150),
    batch_size=32,
    class_mode="binary",
    shuffle=False)

val_gen = ImageDataGenerator(rescale=1./255)
val =val_gen.flow_from_directory(
    "val",target_size=(150,150),
    batch_size=32,
    class_mode="binary")

# Model
model = Sequential([
    #nuron block 1
    Conv2D(32,(3,3), padding='same',input_shape=(150,150,3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    
    #nuron block 2
    Conv2D(64,(3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),

    #nuron block 3
    Conv2D(128,(3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    
    #nuron block 4
    Conv2D(256,(3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    
    # Final Layer
    Flatten(),
    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])

# Compile 
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Call-Backs
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5)

# Train
history = model.fit(
    train,
    steps_per_epoch=train.samples // train.batch_size,
    validation_data=val,
    validation_steps=val.samples // val.batch_size,
    epochs=15,
    callbacks=[early_stop, reduce_lr]
)

test_loss, test_acc = model.evaluate(test)
print(f"Test Accuracy: {test_acc*100:.2f}%")


# Evaluation
y_test_true = test.classes
y_test_probs = model.predict(test).ravel()
y_pred_05 = (y_test_probs > 0.5).astype(int)
print(confusion_matrix(y_test_true, y_pred_05))
print(classification_report(y_test_true, y_pred_05, target_names=list(test.class_indices.keys())))
model.summary()

# ROC 
fpr, tpr, thresholds = roc_curve(y_test_true, y_test_probs)
roc_auc = auc(fpr, tpr)
print("ROC-AUC Score:", roc_auc)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - X-ray Classification")
plt.legend()
plt.show()

# Saving Model
model.save("Xray_model.h5")


