"""
Encapsulates neural style transfer generator. There is StyleContentModel tf.keras model class that is used for
processing style and content image into generated image. That model class contains vgg19 keras pretrained model
that it uses for neural style transfer. NeuralStyleModel encapsulate whole neural style transfer generating process,
 and there is simple interface that can be sued for neural style transfer.
"""

import os

import tensorflow as tf
import numpy as np

import cv2
from tqdm import tqdm


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = self.vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [self.gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

    def vgg_layers(self, layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        # Load our model. Load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / (num_locations)


class NeuralStyleModel:
    CONTENT_LAYERS = ['block5_conv2']

    STYLE_LAYERS = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    def __init__(self, style_weight=1e-2, content_weight=1e4, content_layers=None, style_layers=None, output_dir=""):
        self.output_dir = output_dir
        if style_layers is None:
            style_layers = NeuralStyleModel.STYLE_LAYERS
        if content_layers is None:
            content_layers = NeuralStyleModel.CONTENT_LAYERS

        self.style_layers = style_layers
        self.content_layers = content_layers

        self.extractor = None
        self.set_extractor(self.style_layers, self.content_layers)

        self.style_weight = style_weight
        self.content_weight = content_weight

        self.style_targets = None
        self.content_targets = None

        self.style_image = None
        self.content_image = None

        self.process_image = None

        self.optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    def set_style_image(self, path_style_image):
        self.style_image = NeuralStyleModel.load_img(path_to_img=path_style_image)
        self.process_style_target(style_image=self.style_image)

    def set_content_image(self, path_context_image):
        self.content_image = NeuralStyleModel.load_img(path_to_img=path_context_image)
        self.process_content_target(content_image=self.content_image)
        self.process_image = tf.Variable(self.content_image)

    def reset_process(self):
        if self.content_image:
            self.process_image = tf.Variable(self.content_image)

    def get_style_targets(self):
        if self.style_targets:
            return self.style_targets
        raise ValueError("Style targets is not initialized.")

    def get_content_targets(self):
        if self.content_targets:
            return self.content_targets
        raise ValueError("Content targets is not initialized.")

    def process_style_target(self, style_image):
        self.style_targets = self.extractor(style_image)['style']

    def process_content_target(self, content_image):
        self.content_targets = self.extractor(content_image)['content']

    def set_extractor(self, style_layers, content_layers):
        self.extractor = StyleContentModel(style_layers, content_layers)

    @staticmethod
    def load_img(path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def imshow(self, image, title="", epoch=""):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)


        image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.output_dir:
            output_path = os.path.join(self.output_dir, title + epoch + ".jpg")
            cv2.imwrite(output_path, image_cv)

        cv2.imshow(title, image_cv)
        cv2.waitKey(1)

    def clip_0_1(self, image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def style_content_loss(self, outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']

        style_targets = self.get_style_targets()
        content_targets = self.get_content_targets()

        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / len(self.style_layers)

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / len(self.content_layers)
        loss = style_loss + content_loss
        return loss

    @staticmethod
    def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return tensor


    @tf.function
    def train_step(self, image):
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs)

        grad = tape.gradient(loss, image)
        self.optimizer.apply_gradients([(grad, image)])
        image.assign(self.clip_0_1(image))
        return loss

    def process_neural_style(self, num_epochs=10, num_steps=100, reset_process=False, title="", save_epoch=False):
        if not tf.is_tensor(self.content_image) or not tf.is_tensor(self.style_image):
            raise ValueError("Content or style image is not set. First set content and style"
                             " images and than you can process them to create neural style transfer.")

        if reset_process:
            self.reset_process()

        if not title:
            title = "Neural style transfer"

        step = 0
        description = "processing neural style transfer [loss: {}]"

        progress_bar = tqdm(total=num_epochs, desc=description, position=0, leave=False)
        for n in range(num_epochs):
            loss = 0
            for m in range(num_steps):
                step += 1
                loss = self.train_step(self.process_image)

            # NeuralStyleModel.imshow(image=NeuralStyleModel.tensor_to_image(self.process_image), title="Neural style transfer")
            image = NeuralStyleModel.tensor_to_image(self.process_image)
            if save_epoch:
                epoch = str(n)
            else:
                epoch = ""
            self.imshow(image=image, title=title, epoch=epoch)
            progress_bar.desc = description.format(loss)
            progress_bar.update(1)
