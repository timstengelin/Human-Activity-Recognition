import gin
import os
import cv2
import numpy as np
import tensorflow as tf
import logging


@gin.configurable
def visualize(model, images, labels, step, run_paths):
    '''
    Execute visualization for deep learning models

    Args:
        model (keras.Model): Neural network to analyze
        images (Tensor): Batch of input images
        labels (Tensor): Corresponding labels for input images
        step (int): Current training step
        run_paths (dict): Dictionary containing directories for saving outputs
    '''

    # Define directories for saving visual outputs
    viz_dirs = {
        "original": os.path.join(run_paths['path_deep_visualization'], 'original'),
        "grad_cam": os.path.join(run_paths['path_deep_visualization'], 'grad_cam'),
        "guided_backprop": os.path.join(run_paths['path_deep_visualization'], 'guided_backpropagation'),
        "guided_grad_cam": os.path.join(run_paths['path_deep_visualization'], 'guided_grad_cam'),
        "integrated_gradients": os.path.join(run_paths['path_deep_visualization'], 'integrated_gradients'),
    }

    # Save the original image
    original_image = tf.image.convert_image_dtype(images[0], dtype=tf.uint8)
    original_image = cv2.cvtColor(original_image.numpy(), cv2.COLOR_RGB2BGR)
    original_label = labels[0].numpy()
    original_path = os.path.join(viz_dirs['original'], f'original_step_{step}_label_{original_label}.png')
    os.makedirs(viz_dirs['original'], exist_ok=True)
    cv2.imwrite(original_path, original_image)
    logging.info(f'Original image for step {step} saved at {original_path}.')

    # Generate and save Grad-CAM visualization
    grad_image, heatmap, pred, grad_path = grad_cam(model, images, step, viz_dirs['grad_cam'])
    cv2.imwrite(grad_path, grad_image)
    logging.info(f'Grad-CAM visualization for step {step} saved at {grad_path}.')

    # Generate and save Guided Backpropagation visualization
    guided_img, guided_path = guided_backpropagation(model, images, step, pred, viz_dirs['guided_backprop'])
    cv2.imwrite(guided_path, guided_img)
    logging.info(f'Guided Backpropagation visualization for step {step} saved at {guided_path}.')

    # Generate and save Guided Grad-CAM visualization
    guided_cam_img, guided_cam_path = guided_grad_cam(guided_img, heatmap, step, pred, viz_dirs['guided_grad_cam'])
    cv2.imwrite(guided_cam_path, guided_cam_img)
    logging.info(f'Guided Grad-CAM visualization for step {step} saved at {guided_cam_path}.')

    # Generate and save Integrated Gradients visualization
    int_grad_img, int_grad_path = integrated_gradients(model, images, step, pred, viz_dirs['integrated_gradients'])
    cv2.imwrite(int_grad_path, int_grad_img)
    logging.info(f'Integrated Gradients visualization for step {step} saved at {int_grad_path}.')


@gin.configurable
def grad_cam(model, images, step, grad_cam_path):
    '''
    Create Grad-CAM visualizations

    Args:
        model (keras.Model): The neural network model
        images (Tensor): Input batch for visualization
        step (int): Training step for which Grad-CAM is generated
        grad_cam_path (str): Output directory for Grad-CAM images

    Returns:
        tuple: Grad-CAM visualization, CAM heatmap, predicted class index, and image save path
    '''

    os.makedirs(grad_cam_path, exist_ok=True)

    # Build a sub-model for Grad-CAM
    grad_cam_model = tf.keras.Model(inputs=model.input, outputs=[
        model.get_layer('last_conv').output,
        model.get_layer('output').output
    ])

    # Compute gradients and predictions
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_cam_model(images, training=False)
        tape.watch(conv_output)
        predicted_label = tf.argmax(predictions[0]).numpy()
        target_score = predictions[:, predicted_label]

    gradients = tape.gradient(target_score, conv_output)

    # Calculate weights and the CAM
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2)).numpy()
    feature_map = conv_output[0].numpy()
    cam = np.sum(feature_map * weights, axis=-1)

    # Process and normalize CAM
    cam_resized = cv2.resize(cam, (images.shape[1], images.shape[2]))
    cam_resized = np.maximum(cam_resized, 0) / (np.max(cam_resized) + 1e-16)
    cam_colored = (cam_resized * 255).astype(np.uint8)
    cam_colored = cv2.applyColorMap(cam_colored, cv2.COLORMAP_JET)

    # Overlay CAM on the original image
    input_image = tf.image.convert_image_dtype(images[0], dtype=tf.uint8).numpy()
    input_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    grad_cam_result = cv2.addWeighted(input_bgr, 1, cam_colored, 0.3, 0)

    # Define save path
    output_path = os.path.join(grad_cam_path, f'grad_cam_step_{step}_prediction_{predicted_label}.png')

    return grad_cam_result, cam_resized, predicted_label, output_path


@gin.configurable
def guided_backpropagation(model, images, step, prediction, guided_backprop_path):
    '''
    Produce Guided Backpropagation visualizations

    Args:
        model (keras.Model): Model under analysis
        images (Tensor): Input images for visualization
        step (int): Training step when generating the visualization
        prediction (int): Predicted class for the images
        guided_backprop_path (str): Directory for saving the visualizations

    Returns:
        tuple: Guided Backpropagation visualization and its file save path
    '''

    os.makedirs(guided_backprop_path, exist_ok=True)

    # Define custom ReLU function
    @tf.custom_gradient
    def custom_relu(x):
        def grad(dy):
            return tf.where(dy > 0, dy, 0) * tf.where(x > 0, 1, 0)
        return tf.nn.relu(x), grad

    # Clone and modify the model to use custom ReLU
    layer_outputs = model.get_layer('last_conv').output
    guided_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    for layer in guided_model.layers:
        if hasattr(layer, 'activation') and layer.activation == tf.keras.activations.relu:
            layer.activation = custom_relu

    # Compute gradients with respect to the input images
    with tf.GradientTape() as tape:
        tape.watch(images)
        feature_maps = guided_model(images, training=False)
    gradients = tape.gradient(feature_maps, images).numpy()

    # Normalize gradients for visualization
    norm_grads = (gradients - np.mean(gradients)) / (np.std(gradients) + 1e-8)
    norm_grads = np.clip((norm_grads * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)

    # Prepare save path
    output_path = os.path.join(
        guided_backprop_path, f'guided_backprop_step_{step}_prediction_{prediction}.png')

    return norm_grads[0], output_path


@gin.configurable
def guided_grad_cam(guided_backprop_image, cam, step, prediction, guided_grad_cam_path):
    '''
    Generate the guided Grad-CAM images

    Parameters:
        guided_backprop_image (numpy array): guided backpropagation images
        cam (numpy array): CAM mask images
        step (int): number of corresponding model training steps when performing deep visualization
        prediction (int): predicted labels of the corresponding images
        guided_grad_cam_path (string): directory where guided Grad-CAM images are saved

    Returns:
        guided_grad_cam_image (numpy array): guided Grad-CAM images
        save_path (string): the path and file names of the guided Grad-CAM images
    '''

    os.makedirs(guided_grad_cam_path, exist_ok=True)

    # Integrate guided backpropagation and CAM to create the visualization
    cam_reshaped = np.expand_dims(cam, axis=-1)
    guided_grad_cam_image = np.multiply(guided_backprop_image, cam_reshaped).astype('uint8')

    file_name = f"guided_grad_cam_step_{step}_prediction_{prediction}.png"
    save_path = os.path.join(guided_grad_cam_path, file_name)

    return guided_grad_cam_image, save_path


@gin.configurable
def integrated_gradients(model, images, step, prediction, integrated_gradients_path, m_steps=50, batch_size=32):
    '''
    Generate integrated gradients images

    Parameters:
        model (keras.Model): Keras model object
        images (Tensor): Batch of images used for visualization
        step (int): Training step during visualization
        prediction (int): Predicted label for the images
        integrated_gradients_path (string): Directory for saving images
        m_steps (int): Number of steps for integral approximation
        batch_size (int): Batch size for gradient computation

    Returns:
        integrated_gradients_image (numpy array): Integrated gradients visualization
        save_path (string): File path for the saved image
    '''

    os.makedirs(integrated_gradients_path, exist_ok=True)

    # Define baseline and interpolation steps
    baseline = tf.zeros_like(images[0])
    interpolation_steps = tf.linspace(0.0, 1.0, num=m_steps + 1)
    gradient_store = tf.TensorArray(dtype=tf.float32, size=m_steps + 1)

    def interpolate(baseline_img, target_img, alphas):
        expanded_alphas = alphas[:, None, None, None]
        delta = tf.expand_dims(target_img, axis=0) - tf.expand_dims(baseline_img, axis=0)
        return tf.expand_dims(baseline_img, axis=0) + expanded_alphas * delta

    def compute_grads(inputs, target_idx):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            target_predictions = model(inputs)[:, target_idx]
        return tape.gradient(target_predictions, inputs)

    for start_idx in tf.range(0, len(interpolation_steps), batch_size):
        end_idx = tf.minimum(start_idx + batch_size, len(interpolation_steps))
        current_batch = interpolation_steps[start_idx:end_idx]

        interpolated_images = interpolate(baseline, images[0], current_batch)
        gradients = compute_grads(interpolated_images, prediction)

        gradient_store = gradient_store.scatter(tf.range(start_idx, end_idx), gradients)

    stacked_gradients = gradient_store.stack()

    def approximate_integral(grads):
        trapezoidal_grads = (grads[:-1] + grads[1:]) / 2.0
        return tf.reduce_mean(trapezoidal_grads, axis=0)

    def normalize(data):
        return (data - tf.reduce_mean(data)) / (tf.math.reduce_std(data) + 1e-16)

    averaged_gradients = approximate_integral(stacked_gradients)
    normalized_gradients = normalize(averaged_gradients)

    # Compute attribution
    attributions = (images[0] - baseline) * normalized_gradients
    attribution_map = tf.reduce_mean(tf.abs(attributions), axis=-1).numpy()
    scaled_map = ((attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min()) * 255).astype('uint8')

    # Create heatmap and overlay
    heatmap = cv2.applyColorMap(scaled_map, cv2.COLORMAP_INFERNO)
    original_img = (images[0] * 255).numpy().astype('uint8')
    overlay = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), 0.4, heatmap, 0.9, 0)

    file_name = f"integrated_gradients_step_{step}_prediction_{prediction}.png"
    save_path = os.path.join(integrated_gradients_path, file_name)

    return overlay, save_path