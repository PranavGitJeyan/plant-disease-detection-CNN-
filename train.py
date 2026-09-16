import argparse
import json
import os
import tensorflow as tf
try:
    from keras.applications import MobileNetV2
    from keras.layers import Dense, GlobalAveragePooling2D
    from keras.models import Model
except ImportError:
    from tensorflow.keras.applications import MobileNetV2  # type: ignore
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # type: ignore
    from tensorflow.keras.models import Model  # type: ignore
from preprocess import create_data_generators

def train_model(dataset_dir, epochs=5, batch_size=32, model_save_path="plant_disease_model.keras"):
    print("============================================================")
    print("Plant Disease Detection - MobileNetV2 Training Pipeline")
    print("============================================================")

    print("\n[Step 1/4] Initializing data generators...")
    train_ds, val_ds = create_data_generators(dataset_dir=dataset_dir, batch_size=batch_size)
    
    # Get class names from tf.data dataset
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Successfully loaded dataset with {num_classes} classes.")

    # Save class indices json for later inference in your local app
    class_indices = {class_name: i for i, class_name in enumerate(class_names)}
    with open("class_indices.json", "w") as f:
        json.dump(class_indices, f)
    print("Saved class_indices.json successfully.")

    print("\n[Step 2/4] Building MobileNetV2 model...")
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model for initial training

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("\n[Step 3/4] Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    print(f"\n[Step 4/4] Saving model to {model_save_path}...")
    model.save(model_save_path)
    print("Training complete and model successfully saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for Plant Disease Detection")
    parser.add_argument("--dataset_dir", type=str, default="./dataset/plantvillage dataset/color", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--model_save_path", type=str, default="plant_disease_model.keras", help="Path to save trained model")

    args = parser.parse_args()

    train_model(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_save_path
    )