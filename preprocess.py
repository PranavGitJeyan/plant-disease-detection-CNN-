import tensorflow as tf

def create_data_generators(dataset_dir="/kaggle/input/plantvillage-dataset/plantvillage dataset/color", batch_size=32, target_size=(224, 224)):
    
    # Automatically reserve 20% for validation using modern tf.data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=target_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=target_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    # Grab class_names before transformations
    class_names = train_ds.class_names

    # Normalization layer (rescales pixel values from 0-255 to 0-1)
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    AUTOTUNE = tf.data.AUTOTUNE

    # Apply normalization and prefetch (removed .cache() to prevent RAM overflow)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Attach class_names back to the train_ds dataset object
    train_ds.class_names = class_names

    return train_ds, val_ds