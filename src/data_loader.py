import tensorflow as tf

def get_data_loaders(data_dir="data", batch_size=32, img_size=(256, 256)):
    print(f"Loading training data from {data_dir}/train...")
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=img_size,
        color_mode="rgb",  # UPDATED TO RGB FOR TRANSFER LEARNING
        batch_size=batch_size,
        label_mode="categorical" 
    )
    
    print(f"Loading Testing data from {data_dir}/val...")
    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_dir}/val",
        image_size=img_size,
        color_mode="rgb",  # UPDATED TO RGB
        batch_size=batch_size,
        label_mode="categorical"
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
    val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_data, val_data