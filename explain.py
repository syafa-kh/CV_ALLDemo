import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array, array_to_img
from tensorflow.keras.models import Model

# gradcam implementation taken from https://keras.io/examples/vision/grad_cam/ 
class gradCAM:
    def __init__(self, img_path, model, size=(224,224), last_conv_layer_name='block5_conv4'):
        self.img_array = self.get_img_array(img_path,size)
        self.model = model
        self.model.layers[-1].activation = None
        self.last_conv_layer_name = last_conv_layer_name
        self.heatmap = self.make_gradcam_heatmap() 
        
    def get_img_array(self, img_path, size):
        img = load_img(img_path, target_size=size) # `img` is a PIL image of size 299x299
        self.img = img # class instance variable
        array = img_to_array(img) # `array` is a float32 Numpy array of shape (299, 299, 3)
        array = np.expand_dims(array, axis=0) # We add a dimension to transform our array into a "batch" of size (1, 299, 299, 3)
        return array
    
    def make_gradcam_heatmap(self, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = Model(
            [self.model.inputs], [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )
        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(self.img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def display_gradcam(self, alpha=0.4):
        img = img_to_array(self.img)
        self.heatmap = np.uint8(255 * self.heatmap) # Rescale heatmap to a range 0-255
        jet = cm.get_cmap("jet") # Use jet colormap to colorize heatmap
        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[self.heatmap]
        # Create an image with RGB colorized heatmap
        jet_heatmap = array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = img_to_array(jet_heatmap)
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = array_to_img(superimposed_img)
        return superimposed_img