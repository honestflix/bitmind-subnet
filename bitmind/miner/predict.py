import tensorflow as tf
import numpy as np
import cv2


def predict(model, image, device='/CPU:0'):
    x = np.array(image, dtype=np.float64)

    if x.shape[0] != 256 or x.shape[1] != 256:
        x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_AREA)

    x /= 255.0
    if x.shape[-1] not in (1, 3):
        x = np.stack([x, x, x], axis=-1)
    x = np.expand_dims(x, axis=0)

    with tf.device(device):
        return model.predict(x)[0]
