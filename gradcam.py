import tensorflow as tf
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    def grad_cam(self, image, class_index):
        img_tensor = tf.keras.preprocessing.image.img_to_array(image)
        img_tensor = np.expand_dims(img_tensor, axis =0)
        img_tensor /= 255.

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)
            loss = predictions[:, class_index]

        output= conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        weights = tf.refuce_mean(grads, axis = (0, 1))
        cam = np.ones(output.shape[0:2], dtype =np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, 1]
        
        cam = cv2.resize(cam.numpy(), (244, 244))
        cam = np.maximum(cam, 0)
        heatmap = cam / np.max(cam)

        return heatmap