import numpy as np
import tensorflow as tf
import cv2
import os

def get_gradcam(model, image, last_conv_layer_name="out_relu"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def create_gradcam_visualization(heatmap, original_image):
    os.makedirs("outputs/heatmaps", exist_ok=True)

    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (256,256))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply color map
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert original image
    original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # Overlay
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

    # Save all versions (VERY IMPORTANT FOR GITHUB PROOF)
    cv2.imwrite("outputs/heatmaps/heatmap.jpg", heatmap_color)
    cv2.imwrite("outputs/heatmaps/original.jpg", original_bgr)
    cv2.imwrite("outputs/heatmaps/overlay.jpg", overlay)

    return overlay