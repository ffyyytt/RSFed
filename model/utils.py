import tensorflow as tf
import keras_cv_attention_models
import classification_models.keras

from model.Margin import *
from model.TrainingHandler import *

def get_backbone(backbone_name, x):
    if hasattr(tf.keras.applications, backbone_name):
        headModel = tf.keras.layers.Lambda(lambda data: tf.keras.applications.efficientnet_v2.preprocess_input(data))(x)
        backbone = getattr(tf.keras.applications, backbone_name)(weights = "imagenet", include_top = False)
        backbone.trainable = True
        return tf.keras.layers.GlobalAveragePooling2D()(backbone(headModel))
    elif backbone_name in classification_models.keras.Classifiers.models:
        bakbone, preprocess_input = classification_models.keras.Classifiers.get(backbone_name)
        headModel = preprocess_input(x)
        backbone = bakbone(input_shape=(None, None, 3), weights = "imagenet", include_top = False)
        backbone.trainable = True
        return tf.keras.layers.GlobalAveragePooling2D()(backbone(headModel))
    else:
        backbone = getattr(getattr(keras_cv_attention_models, backbone_name.split(".")[0]), backbone_name.split(".")[1])(num_classes=0)
        headModel = tf.keras.layers.Lambda(lambda data: tf.keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"))(x)
        backbone.trainable = True
        if "beit" in backbone_name:
            return backbone(headModel)
        return tf.keras.layers.GlobalAveragePooling2D()(backbone(headModel))
    
def model_factory(backbones, n_classes, embedding_dimention = 1024, margin = 0.3):
    image = tf.keras.layers.Input(shape = (None, None, 3), dtype=tf.uint8, name = 'image')
    label = tf.keras.layers.Input(shape = (n_classes, ), name = 'label', dtype = tf.int32)

    features = [get_backbone(backbone, image) for backbone in backbones]
    if len(backbones) == 1:
        headModel = features[0]
    else:
        headModel = tf.keras.layers.Concatenate()(features)
    headModel = tf.keras.layers.Dense(embedding_dimention)(headModel)
    headModel = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name = "feature")(headModel)
    margin = Margin(num_classes = n_classes, margin = margin)([headModel, label])
    output = tf.keras.layers.Softmax(dtype=tf.float32, name = "output")(margin)

    model = tf.keras.models.Model(inputs = [image, label], outputs = [output])
    return model

def simple_model_factory(backbone, n_classes):
    image = tf.keras.layers.Input(shape = (None, None, 3), dtype=tf.uint8, name = 'image')
    headModel = get_backbone(backbone, image)
    headModel = tf.keras.layers.Dense(n_classes, activation = "softmax", name = "softmax")(headModel)
    model = tf.keras.models.Model(inputs = [image], outputs = [headModel])
    return model