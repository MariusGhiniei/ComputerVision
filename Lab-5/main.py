# ======== SETUP (imports + CPU only) ========
import os, glob, time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model   # <- needed for layers & Model
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
import cv2

print("TF version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU/Metal devices found:", gpus)
    DEVICE = "/GPU:0" # pe Mac cu tensorflow-metal
else:
    print("No GPU found, using CPU.")
    DEVICE = "/CPU:0"

# ======== PATHS + GENERAL SETTINGS ========
# TODO: modifică aceste căi pentru datasetul tău
DATASET_DIR = "/Users/marius/PycharmProjects/CV-lab1/Lab-5/dataset/Brain Tumor Segmentation"  # <- MODIFICA
IMAGE_DIR   = os.path.join(DATASET_DIR, "images")  # ex: "images"
MASK_DIR    = os.path.join(DATASET_DIR, "masks")   # ex: "masks"

IMG_HEIGHT   = 128
IMG_WIDTH    = 128
IMG_CHANNELS = 1        # 1 pentru MRI (grayscale); pune 3 pentru RGB

BATCH_SIZE = 8
EPOCHS     = 30

# ======== LOAD FILE LIST ========
# presupunem .png; schimbă în "*.jpg" dacă e cazul
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
if len(image_paths) == 0:
    raise RuntimeError("Nu s-au găsit imagini. Verifică IMAGE_DIR și extensia fișierelor.")

# presupunem că măștile au același nume de fișier și sunt în MASK_DIR
mask_paths = [os.path.join(MASK_DIR, os.path.basename(p)) for p in image_paths]
for p in mask_paths[:5]:
    if not os.path.exists(p):
        raise RuntimeError(f"Masca nu există: {p}. Verifică structura folderelor.")

print(f"Total imagini: {len(image_paths)}")

# ======== HELPERS: read image & mask ========
def read_image(path, grayscale=True):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = np.expand_dims(img, axis=-1)  # (H,W,1)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # optional
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    return img

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
    # presupunem 0/255 -> convertim la 0/1
    mask = (mask > 127).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)  # (H,W,1)
    return mask

# ======== LOAD ALL IMAGES & MASKS IN MEMORY ========
X = np.array([read_image(p, grayscale=(IMG_CHANNELS == 1)) for p in image_paths], dtype=np.float32)
y = np.array([read_mask(p) for p in mask_paths], dtype=np.float32)

print("X shape:", X.shape, "y shape:", y.shape)

# ======== TRAIN/TEST SPLIT 70/30 ========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# ======== TF.DATA WITH SIMPLE AUGMENTATION ========
def augment(image, mask):
    # flip stanga/dreapta
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)
    # flip sus/jos
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask  = tf.image.flip_up_down(mask)
    return image, mask

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = (train_dataset
                 .shuffle(1000)
                 .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(BATCH_SIZE)
                 .prefetch(tf.data.AUTOTUNE))

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# ======== U-NET MODEL ========
def double_conv(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = double_conv(inputs, 64)
    p1 = layers.MaxPool2D((2, 2))(c1)

    c2 = double_conv(p1, 128)
    p2 = layers.MaxPool2D((2, 2))(c2)

    c3 = double_conv(p2, 256)
    p3 = layers.MaxPool2D((2, 2))(c3)

    c4 = double_conv(p3, 512)
    p4 = layers.MaxPool2D((2, 2))(c4)

    # Bottleneck
    bn = double_conv(p4, 1024)

    # Decoder
    u6 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(bn)
    u6 = layers.Concatenate()([u6, c4])
    c6 = double_conv(u6, 512)

    u7 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = double_conv(u7, 256)

    u8 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = double_conv(u8, 128)

    u9 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = double_conv(u9, 64)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c9)  # binary mask

    model = Model(inputs, outputs, name="U-Net")
    return model

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return 1.0 - (2. * intersection + smooth) / (denom + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# ======== TRAINING (CPU only) ========
with tf.device(DEVICE):
    model = unet_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=bce_dice_loss,
                  metrics=['accuracy'])
    model.summary()

    start = time.time()
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS
    )
    end = time.time()
    print(f"Training time for {EPOCHS} epochs: {(end - start)/60:.2f} minutes")

# ======== EVALUATION: PIXEL ACCURACY, IoU, DICE ========
def compute_metrics_np(y_true, y_pred_bin):
    # y_true, y_pred_bin: (H,W,1) arrays 0/1
    y_true = y_true.astype(bool)
    y_pred = y_pred_bin.astype(bool)

    tp = np.logical_and(y_true,  y_pred).sum()
    tn = np.logical_and(~y_true, ~y_pred).sum()
    fp = np.logical_and(~y_true,  y_pred).sum()
    fn = np.logical_and(y_true,  ~y_pred).sum()

    pixel_acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)          # Jaccard / IoU
    dice      = (2 * tp) / (2 * tp + fp + fn + 1e-6)

    return pixel_acc, iou, dice

with tf.device(DEVICE):
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE)
y_pred = (y_prob > 0.5).astype(np.uint8)

pixel_accs, ious, dices = [], [], []
for i in range(len(X_test)):
    pa, iou, dc = compute_metrics_np(y_test[i], y_pred[i])
    pixel_accs.append(pa)
    ious.append(iou)
    dices.append(dc)

print("========== FINAL METRICS (MEAN OVER TEST SET) ==========")
print("Mean Pixel Accuracy:", np.mean(pixel_accs))
print("Mean IoU (Jaccard):", np.mean(ious))
print("Mean Dice Coefficient:", np.mean(dices))
