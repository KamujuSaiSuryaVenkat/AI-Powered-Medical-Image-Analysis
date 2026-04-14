import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications import MobileNetV2

def build_model(num_classes=6):
    # 1. Define Input Shape (RGB instead of Grayscale)
    inputs = Input(shape=(256, 256, 3))

    # 2. Data Augmentation (Slightly shifts/rotates images to prevent memorization)
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.05)(x)
    x = RandomZoom(0.05)(x)

    # 3. Load the Pre-Trained Google Model (MobileNetV2)
    # We freeze its weights so we don't accidentally erase its pre-existing knowledge
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = False 

    # 4. Add our Custom Classification Head for our 6 diseases
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # 5. Build and Compile
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model