import gin
import os
import cv2
import numpy as np
import tensorflow as tf
import logging

@gin.configurable
def visualize(model, images, labels, step, run_paths):
    '''
    Generates and saves visualizations, including original and Grad-CAM images

    Args:
        model (keras.Model): Trained model
        images (Tensor): Batch of input images
        labels (Tensor): Corresponding labels for the images
        step (int): Current training step
        run_paths (dict): Paths for saving outputs
    '''
    vis_path = run_paths.get('path_deep_visualization', './visualizations')

    # Process original image
    orig_img_dir = os.path.join(vis_path, 'original')
    os.makedirs(orig_img_dir, exist_ok=True)
    orig_img = tf.cast(images[0] * 255, tf.uint8).numpy()
    orig_label = labels[0].numpy()
    orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    orig_img_path = os.path.join(orig_img_dir, f'original_step_{step}_label_{orig_label}.png')
    cv2.imwrite(orig_img_path, orig_img_bgr)
    logging.info(f"Original visualization saved at {orig_img_path}")

    # Generate and save Grad-CAM visualization
    grad_cam_img, _, _, grad_cam_save_dir = grad_cam(model, images, step, os.path.join(vis_path, 'grad_cam'))
    logging.info(f"Grad-CAM visualization saved at {grad_cam_save_dir}")

    # Generate and save Guided Grad-CAM visualization
    guided_grad_cam_img, _, _, guided_grad_cam_save_dir = guided_grad_cam(model, images, step, os.path.join(vis_path, 'guided_grad_cam'))
    logging.info(f"Guided Grad-CAM visualization saved at {guided_grad_cam_save_dir}")

@gin.configurable
def grad_cam(model, images, step, grad_cam_dir):
    '''
    Creates Grad-CAM visualizations for a given image batch and saves them

    Args:
        model (keras.Model): Trained model for visualization
        images (Tensor): Batch of input images
        step (int): Current training step for naming purposes
        grad_cam_dir (str): Directory to save Grad-CAM images

    Returns:
        Tuple: Grad-CAM image, heatmap, predicted class, Grad-CAM save directory.
    '''
    os.makedirs(grad_cam_dir, exist_ok=True)

    # Identify the last convolutional layer
    conv_layer_name = next(layer.name for layer in reversed(model.layers) if isinstance(layer, tf.keras.layers.Conv2D))
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(conv_layer_name).output, model.output])

    # Compute gradients for the predicted class
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(images)
        pred_class = tf.argmax(predictions[0])
        loss = predictions[:, pred_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Generate heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU-like operation
    heatmap /= heatmap.max() if heatmap.max() != 0 else 1
    heatmap = cv2.resize(heatmap, (images.shape[2], images.shape[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Overlay heatmap on the original image
    orig_img = tf.cast(images[0] * 255, tf.uint8).numpy()
    orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    grad_cam_img = cv2.addWeighted(orig_img_bgr, 0.6, heatmap, 0.4, 0)

    # Save Grad-CAM image
    grad_cam_path = os.path.join(grad_cam_dir, f'grad_cam_step_{step}.png')
    cv2.imwrite(grad_cam_path, grad_cam_img)

    return grad_cam_img, heatmap, pred_class.numpy(), grad_cam_path


@gin.configurable
def guided_grad_cam(model, images, step, guided_grad_cam_dir):
    '''
    Creates Guided Grad-CAM visualizations for a given image batch and saves them

    Args:
        model (keras.Model): Trained model for visualization
        images (Tensor): Batch of input images
        step (int): Current training step for naming purposes
        guided_grad_cam_dir (str): Directory to save Guided Grad-CAM images

    Returns:
        Tuple: Guided Grad-CAM image, heatmap, predicted class, Guided Grad-CAM save directory.
    '''
    os.makedirs(guided_grad_cam_dir, exist_ok=True)

    # Identify the last convolutional layer
    conv_layer_name = next(layer.name for layer in reversed(model.layers) if isinstance(layer, tf.keras.layers.Conv2D))
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(conv_layer_name).output, model.output])

    # Use a persistent GradientTape to calculate gradients multiple times
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(images)
        conv_outputs, predictions = grad_model(images)
        pred_class = tf.argmax(predictions[0])
        loss = predictions[:, pred_class]

    # Gradients with respect to convolutional layer outputs
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Generate Grad-CAM heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU-like operation
    heatmap /= heatmap.max() if heatmap.max() != 0 else 1
    heatmap = cv2.resize(heatmap, (images.shape[2], images.shape[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Gradients with respect to the input images
    guided_grads = tape.gradient(loss, images)[0].numpy()
    guided_grads = np.maximum(guided_grads, 0)  # ReLU operation
    guided_grads -= guided_grads.min()
    guided_grads /= guided_grads.max() if guided_grads.max() != 0 else 1
    guided_grads = np.uint8(guided_grads * 255)

    del tape  # Explicitly delete the persistent tape to free resources

    # Multiply Grad-CAM heatmap with Guided Backprop gradients
    guided_grad_cam = heatmap * guided_grads.astype('float64')
    guided_grad_cam -= guided_grad_cam.min()
    guided_grad_cam /= guided_grad_cam.max() if guided_grad_cam.max() != 0 else 1
    guided_grad_cam = np.uint8(guided_grad_cam * 255)  # Convert back to uint8

    # Save Guided Grad-CAM image
    guided_grad_cam_path = os.path.join(guided_grad_cam_dir, f'guided_grad_cam_step_{step}.png')
    cv2.imwrite(guided_grad_cam_path, guided_grad_cam)

    return guided_grad_cam, heatmap, pred_class.numpy(), guided_grad_cam_path