import tensorflow as tf

class Margin(tf.keras.layers.Layer):
    def __init__(self, num_classes, margin = 0.3, scale=20, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[0][-1], self.num_classes), initializer='random_normal', trainable=True, name = "margin_weight")

    def logits_cotangent(self, feature, labels):
        cosine = self.cosine(feature)
        theta = tf.acos(tf.clip_by_value(cosine, -1, 1))
        cot = tf.math.divide_no_nan(1.0, tf.math.tan(theta))
        mr = tf.random.normal(shape = tf.shape(cosine), mean = self.margin, stddev = 0.1*self.margin)
        cot_add =  tf.math.divide_no_nan(1.0, tf.math.tan(theta + mr))
        mask = tf.cast(labels, dtype=cosine.dtype)
        logits = mask*cot_add + (1-mask)*cot
        return logits

    def cosine(self, feature):
        # x = tf.nn.l2_normalize(feature, axis=1)
        w = tf.nn.l2_normalize(self.W, axis=0)
        cos = tf.matmul(feature, w)
        return cos

    def call(self, inputs, training):
        feature, labels = inputs
        if training:
            logits = self.logits_cotangent(feature, labels)
        else:
            logits = self.cosine(feature)
        return logits*self.scale

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scale': self.scale,
            'margin': self.margin,
            'num_classes': self.num_classes,
        })
        return config