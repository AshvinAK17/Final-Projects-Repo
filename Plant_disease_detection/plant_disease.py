import os, json, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2

# ── check for GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TRAIN_DIR = r'C:\Ashvin\AI ML\Project\Plant Disease\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train'
VAL_DIR   = r'C:\Ashvin\AI ML\Project\Plant Disease\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid'
IMG_SIZE  = (128,128)
BATCH     = 32
CLASSES   = 38
EPOCHS    = 20

# ── make output folders in same directory as this script ───────────────────────
BASE = os.getcwd()
MDIR = os.path.join(BASE, "models")
PDIR = os.path.join(BASE, "plots")
XDIR = os.path.join(BASE, "metrics_reports")
for d in (MDIR, PDIR, XDIR):
    os.makedirs(d, exist_ok=True)

# ── DATA GENERATORS ─────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    dtype="float32"                # force float32
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0, dtype="float32")

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical"
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical", shuffle=False
)

# ── BUILD CUSTOM CNN ───────────────────────────────────────────────────────────
def build_custom_cnn(input_shape, n_classes):
    reg = regularizers.l2(1e-4)
    inp = layers.Input(shape=input_shape, dtype="float32")  # explicit dtype
    x = layers.Conv2D(32, 3, activation="relu", kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inputs=inp, outputs=out, name="CustomCNN")

# ── TRANSFER LEARNING WRAPPER ───────────────────────────────────────────────────
def tl_model(base_cls):
    bm = base_cls(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    bm.trainable = False
    inp = bm.input
    x = bm.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(CLASSES, activation="softmax")(x)
    return models.Model(inputs=inp, outputs=out, name=base_cls.__name__)

models_to_train = {
    "CustomCNN": build_custom_cnn(IMG_SIZE + (3,), CLASSES),
    "MobileNetV2": tl_model(MobileNetV2),
    "VGG16": tl_model(VGG16),
    "ResNet50": tl_model(ResNet50),
}

# ── CALLBACKS ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
]

# ── TRAIN & EVALUATE ───────────────────────────────────────────────────────────
for name, model in models_to_train.items():
    print(f"\n Training {name} …")
    model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks)

    # — predict & classification report
    val_gen.reset()
    y_true = val_gen.classes
    y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(val_gen.class_indices.keys()),
        output_dict=True,
        zero_division=0,
    )

    # — clean metrics JSON
    clean = {
        "accuracy": round(report["accuracy"] * 100, 2),
        "macro_avg": {
            "precision": round(report["macro avg"]["precision"] * 100, 2),
            "recall": round(report["macro avg"]["recall"] * 100, 2),
            "f1_score": round(report["macro avg"]["f1-score"] * 100, 2),
        },
        "per_class": {},
    }
    for cls in val_gen.class_indices:
        c = report[cls]
        clean["per_class"][cls] = {
            "precision": round(c["precision"] * 100, 2),
            "recall": round(c["recall"] * 100, 2),
            "f1_score": round(c["f1-score"] * 100, 2),
            "support": int(c["support"]),
        }

    # — save JSON, plot, model
    with open(os.path.join(XDIR, f"{name}_metrics.json"), "w") as f:
        json.dump(clean, f, indent=2)

    plt.figure(figsize=(6, 3))
    plt.plot(hist.history["accuracy"], label="train")
    plt.plot(hist.history["val_accuracy"], label="val")
    plt.title(f"{name} Accuracy")
    plt.legend()
    plt.savefig(os.path.join(PDIR, f"{name}_acc.png"))
    plt.close()

    model.save(os.path.join(MDIR, f"{name}.h5"))
    print(f" {name}: Acc {clean['accuracy']}%, Macro F1 {clean['macro_avg']['f1_score']}%")

print("\n Done—all outputs are in ./models, ./plots, ./metrics_reports")
