# ============================================================
# 🔬 STEGANALYSIS CNN — FINAL FIXED VERSION
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

# ============================================================
# ⚙️ CONFIG (UPDATED)
# ============================================================
IMG_SIZE   = 256
BATCH_SIZE = 4
EPOCHS     = 40
N_CHANNELS = 18   # 🔥 6 SRM × 3 RGB

# ============================================================
# 📁 PATHS
# ============================================================
clean_path  = "/home/pranj/Dataset/Clean-1"
stego_paths = [
    "/home/pranj/Dataset/JMiPOD",
    "/home/pranj/Dataset/JUNIWARD",
    "/home/pranj/Dataset/UERD",
]

# ============================================================
# 🔬 SRM KERNELS
# ============================================================
SRM_KERNELS = [
    np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],np.float32)/12,
    np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],np.float32)/4,
    np.array([[-1,0,1],[0,0,0],[1,0,-1]],np.float32)/2,
    np.array([[1,-2,1],[-2,4,-2],[1,-2,1]],np.float32)/4,
    np.array([[0,1,0],[1,-4,1],[0,1,0]],np.float32),
    np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],np.float32)/8,
]

# ============================================================
# 📊 LOAD PATHS + LABELS (BINARY)
# ============================================================
paths, labels = [], []

for f in os.listdir(clean_path):
    paths.append(os.path.join(clean_path, f))
    labels.append(0)

for sp in stego_paths:
    for f in os.listdir(sp):
        paths.append(os.path.join(sp, f))
        labels.append(1)

paths  = np.array(paths)
labels = np.array(labels)

# ============================================================
# 🔀 SPLIT
# ============================================================
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    paths, labels, test_size=0.3, stratify=labels, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, stratify=y_tmp)

# ============================================================
# ⚡ DATA PIPELINE (FIXED)
# ============================================================
def parse_fn(path, label):

    def _load(p):
        p = p.decode()
        img = cv2.imread(p)

        if img is None:
            return np.zeros((IMG_SIZE, IMG_SIZE, N_CHANNELS), np.float32)

        img = img.astype(np.float32)
        h, w, _ = img.shape

        # 🔥 RANDOM CROP (NO RESIZE)
        x = np.random.randint(0, w - IMG_SIZE + 1)
        y = np.random.randint(0, h - IMG_SIZE + 1)
        img = img[y:y+IMG_SIZE, x:x+IMG_SIZE]

        channels = []

        # 🔥 SRM PER RGB CHANNEL
        for c in range(3):
            ch = []
            for k in SRM_KERNELS:
                r = cv2.filter2D(img[:, :, c], -1, k)
                r = r / (np.std(r) + 1e-8)
                ch.append(r)
            channels.append(np.stack(ch, axis=-1))

        return np.concatenate(channels, axis=-1)

    img = tf.numpy_function(_load, [path], tf.float32)
    img.set_shape((IMG_SIZE, IMG_SIZE, N_CHANNELS))

    return img, tf.cast(label, tf.float32)


def build_ds(X, y, train=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if train:
        ds = ds.shuffle(5000)

    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = build_ds(X_tr, y_tr, True)
val_ds   = build_ds(X_val, y_val)
test_ds  = build_ds(X_test, y_test)

# ============================================================
# 🤖 MODEL (STEGO-READY)
# ============================================================
def build_model():
    inp = Input((IMG_SIZE, IMG_SIZE, N_CHANNELS))

    # =====================================================
    # 🔬 BLOCK 1 (LOW-LEVEL NOISE)
    # =====================================================
    x = Conv2D(64, 3, padding='same', use_bias=False)(inp)
    x = BatchNormalization()(x)

    # 🔥 TLU (CRITICAL)
    x = Lambda(lambda t: tf.clip_by_value(t, -3.0, 3.0))(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Downsample (NO pooling)
    x = Conv2D(64, 3, strides=2, padding='same')(x)

    # =====================================================
    # 🔬 BLOCK 2 (MID FEATURES)
    # =====================================================
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, 3, strides=2, padding='same')(x)

    # =====================================================
    # 🔬 BLOCK 3 (DEEP FEATURES)
    # =====================================================
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, 3, strides=2, padding='same')(x)

    # =====================================================
    # 🔬 BLOCK 4 (HIGH-LEVEL)
    # =====================================================
    x = Conv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # =====================================================
    # 🔬 CLASSIFIER
    # =====================================================
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)

    out = Dense(1, activation='sigmoid')(x)

    return Model(inp, out)

# ============================================================
# 🔧 COMPILE
# ============================================================
model = build_model()
model.compile(
    optimizer=Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# ============================================================
# 🚀 TRAIN
# ============================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        EarlyStopping(
            monitor='val_loss',      
            patience=8,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',      
            factor=0.3,
            patience=3,
            min_lr=1e-6,             
            verbose=1
        ),
        ModelCheckpoint(
            "best_model.keras",
            monitor='val_loss',
            save_best_only=True
        )
    ]
)

# ============================================================
# 📈 EVALUATION
# ============================================================
preds, true = [], []

for x,y in test_ds:
    p = model.predict(x, verbose=0).flatten()
    preds.extend(p)
    true.extend(y.numpy())

preds = np.array(preds)
true  = np.array(true)

print("\nFINAL RESULTS")
print("Accuracy:", np.mean((preds>0.5)==true))
print("ROC-AUC :", roc_auc_score(true, preds))
print("F1      :", f1_score(true, preds>0.5))