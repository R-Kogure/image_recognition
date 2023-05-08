import numpy as np
import tensorflow as tf

def make_heatmap(name_lastconv, model, inputimg):
    with tf.GradientTape() as tape:
        last_convlayer = model.get_layer(name_lastconv) #モデルの最後の畳み込み層を拾う
        iter = tf.keras.models.Model([model.inputs], [model.output, last_convlayer.output])
        model_out, last_convlayer = iter(inputimg)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_convlayer)
        pooled_grads = tf.keras.backend.mean(grads, axis =(0,1, 2))

    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_convlayer), axis =-1)

    heatmap_shape = (grads.shape[1], grads.shape[2])

    # 正規化
    heatmap_emphasis = np.maximum(heatmap, 0)
    heatmap_emphasis /= np.max(heatmap_emphasis)
    heatmap_emphasis = heatmap_emphasis.reshape(heatmap_shape)

    return heatmap_emphasis