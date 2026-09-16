import tensorflow as tf
try:
    from keras.applications import MobileNetV2
    from keras.models import Sequential
    from keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from keras.optimizers import Adam
except ImportError:
    from tensorflow.keras.applications import MobileNetV2  # type: ignore
    from tensorflow.keras.models import Sequential         # type: ignore
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense  # type: ignore
    from tensorflow.keras.optimizers import Adam           # type: ignore

def build_plant_disease_model(input_shape=(224, 224, 3), num_classes=38, learning_rate=0.001):
    """
    Builds a Transfer Learning model using pre-trained MobileNetV2 for Plant Disease Detection.
    """
    # 1. Load MobileNetV2 with ImageNet weights, without the top classification layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # 2. Freeze the base model layers so their weights are not updated during initial training
    base_model.trainable = False

    # 3. Build the custom classification head
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    # 4. Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_model(input_shape=(224, 224, 3), num_classes=38, learning_rate=0.001):
    """
    Alias for build_plant_disease_model to maintain compatibility.
    """
    return build_plant_disease_model(input_shape=input_shape, num_classes=num_classes, learning_rate=learning_rate)

def safe_load_model(model_path="plant_disease_model.keras"):
    """
    Safely loads a Keras model, automatically resolving version mismatches
    such as the Google Colab Keras 3.13 'quantization_config' deserialization bug in Keras 3.12.
    """
    import os
    import zipfile
    import json
    import tempfile
    import keras

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        return keras.models.load_model(model_path)
    except Exception as e:
        err_msg = str(e)
        if "quantization_config" in err_msg or "Deserialization" in err_msg or "deserialized" in err_msg:
            # Recursively strip unrecognized keys (like quantization_config) from the saved config.json
            def _clean_dict(d, bad_keys={"quantization_config"}):
                if isinstance(d, dict):
                    for k in bad_keys:
                        d.pop(k, None)
                    for v in d.values():
                        _clean_dict(v, bad_keys)
                elif isinstance(d, list):
                    for item in d:
                        _clean_dict(item, bad_keys)
                return d

            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                with zipfile.ZipFile(model_path, "r") as zin, zipfile.ZipFile(tmp_path, "w") as zout:
                    for item in zin.infolist():
                        if item.filename == "config.json":
                            cfg = json.loads(zin.read("config.json").decode("utf-8"))
                            cfg = _clean_dict(cfg)
                            zout.writestr("config.json", json.dumps(cfg))
                        else:
                            zout.writestr(item, zin.read(item.filename))

                loaded = keras.models.load_model(tmp_path)
                try:
                    os.replace(tmp_path, model_path)
                except Exception:
                    pass
                return loaded
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        raise e

load_plant_disease_model = safe_load_model

if __name__ == "__main__":
    model = build_plant_disease_model()
    model.summary()