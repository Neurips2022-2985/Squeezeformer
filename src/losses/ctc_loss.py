import tensorflow as tf

class CtcLoss(tf.keras.losses.Loss):
    def __init__(self, blank=0, name=None):
        super(CtcLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        self.blank = blank

    def call(self, y_true, y_pred):
        loss = ctc_loss(
            y_pred=y_pred["logits"],
            input_length=y_pred["logits_length"],
            y_true=y_true["labels"],
            label_length=y_true["labels_length"],
            blank=self.blank,
            name=self.name
        )
        return tf.nn.compute_average_loss(loss)


@tf.function
def ctc_loss(y_true, y_pred, input_length, label_length, blank, name=None):
    return tf.nn.ctc_loss(
        labels=tf.cast(y_true, tf.int32),
        logit_length=tf.cast(input_length, tf.int32),
        logits=tf.cast(y_pred, tf.float32),
        label_length=tf.cast(label_length, tf.int32),
        logits_time_major=False,
        blank_index=blank,
        name=name
    )
