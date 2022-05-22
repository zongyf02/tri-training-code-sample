import tensorflow as tf
from tensorflow import keras
from keras.engine import data_adapter
from _ssl_models import SSLModel, Ensemble

@tf.function
def pseudo_label(
    i,
    selected,
    m_infer,
    m_preds,
    m_confidence,
    threshold
):
    """
    Generate pseudo-labels from infered probabilities, predictions, and confidence levels.
    Parameters
    ----------
    i: int
        Index of the layer for which the pseudo-labels are generated.
    m_infer: tensor of shape (3, n_unlabeled_samples, n_classes)
        Infered probabilities of each layer.
    m_preds: tensor of shape (3, n_unlabeled_samples, )
        Inferences of each layer. The argmax of each m_infer.
        Passed instead of calculated from m_infer to save recomputation for each layer.
    m_confidence: tensor of shape (3, n_unlabeled_samples, )
        Probability of the infered class. The max value of rows in each m_infer.
        Passed instead of calculated from m_infer to save recomputation for each layer.
    threshold: float
        Confidence threshold for pseudo-labels.
    Returns
    -------
    selected_i: tensor of shape (n_unlabeled_samples, )
        Mask of the selected pseudo-labeled samples in unlabeled samples.
    pseudo_labels: tensor of shape (n_unlabeled_samples, n_classes)
        Pseudo-labels of the selected samples.
    """
    # Select samples that teachers agree and confidence is above threshold
    selected_i = tf.math.logical_and(selected[i], m_preds[i-1] == m_preds[i-2])
    t_confidence = tf.math.minimum(m_confidence[i-1], m_confidence[i-2])
    selected_i = tf.math.logical_and(selected_i, t_confidence > threshold)
    del t_confidence
    
    # Generate pseudo_labels
    pseudo_labels = (m_infer[i-1][selected_i] + m_infer[i-2][selected_i]) / 2.0
    
    return selected_i, pseudo_labels

@tf.function
def smear(y, std=0.05):
    """
    Smear the one-hot labels to introduce diversity.
    Parameters
    ----------
    y: tensor of shape (n_unlabeled_samples, n_classes)
        Labels to be smeared.
    std: float
        Standard deviation of Gaussian noise using in smearing.
        Controls the level of diversity after smearning.
    """
    y += tf.nn.relu(tf.random.normal(tf.shape(y), stddev=std))
    
    # Divide each label by its sum
    # This implementation is faster than map_fn
    y /= tf.transpose(tf.tile(
        tf.reshape(tf.reduce_sum(y, 1), (1, -1)), (tf.shape(y)[1], 1)
    ))
    return y

class TriTraining(SSLModel):
    """
    Tri-training model with output-smearing and pseudo-label editing.
    Parameters
    ----------
    shared_layer: Layer object
        Shared Keras layer at the bottom of the Model
        whose output is feed into `layer1`, `layer2`, and `layer3`.
    layer1: Layer object
        Keras layer at the top of the Model.
    layer2: Layer object
        Keras layer at the top of the Model.
    layer3: Layer object
        Keras layer at the top of the Model.
    smear_std: float, default = 0.01
        Standard deviation of Gaussian noise using in smearing.
        Controls the level of diversity after smearing.
    threshold: float, default=0.95
        Tri-training labeling threshold.
    editing_level: int, default=2
        Iterations of pseudo-label editing. Higher level
        increases stability but decreases the number of pseudo-label
        generated.
    """
    def __init__(
        self,
        shared_layer,
        layer1,
        layer2,
        layer3,
        smear_std=0.0,
        threshold=0.95,
        editing_level=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.shared_layer = shared_layer
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.smear_std = smear_std
        
        if threshold < 0 or threshold >= 1:
            raise ValueError(
                "`threshold` must be in [0, 1)."
                f"Received {threshold}."
            )
        self.threshold = threshold
        
        self.editing_level = editing_level
    
    def call(self, inputs, training=None):
        """
        Calls the model on new inputs and returns the outputs as tensors.
        Parameters
        ----------
        See `keras.Model.call()`
        Returns
        -------
        Average outputs of the three layers fed by the shared layer.
        Identical behavior to `Ensemble.call()`
        """
        x = self.shared_layer(inputs, training=training)
        m1 = self.layer1(x, training=training)
        m2 = self.layer2(x, training=training)
        m3 = self.layer3(x, training=training)
        return (m1 + m2 + m3) / 3
        
    def get_config(self):
        """
        See `keras.Model.get_config()`.
        """
        config = super().get_config()
        config.update({"shared_layer": self.shared_layer,
                       "layer1": self.layer1,
                       "layer2": self.layer2,
                       "layer3": self.layer3,
                       "smear_std": self.smear_std,
                       "threshold": self.threshold,
                       "editing_level": self.editing_level})
    
    def train_step(self, data):
        """
        The logic for one training step.
        Parameters
        ----------
        data:
            See `keras.Model.train_step`.
            Note sample_weight in data is ignored.
        Returns
        -------
        metrics:
            Average loss and compiled metrics from all three layers on the labeled training data.
        """
        X, y, _ = data_adapter.unpack_x_y_sample_weight(data)
        
        # Seperate labeled and unlabeled samples
        has_label = tf.reduce_sum(y, 1) != 0
        y_labeled = y[has_label]
        del y # Free up memory ASAP
        X_labeled = X[has_label]
        X_unlabeled = X[~has_label]
        del X 
        del has_label
        
        # Generate predictions, and confidence for layers
        shared_infer = self.shared_layer(X_unlabeled, training=False)
        m_infer = tf.stack([
            self.layer1(shared_infer, training=False),
            self.layer2(shared_infer, training=False),
            self.layer3(shared_infer, training=False)
        ])
        del shared_infer
        m_preds = tf.math.argmax(m_infer, -1, output_type=tf.dtypes.int32)
        m_confidence = tf.reduce_max(m_infer, -1)
        
        # Reject unstable labels (labels that change in inference mode)
        selected = tf.ones_like(m_preds, dtype=tf.dtypes.bool)
        for i in range(self.editing_level):
            preds = tf.math.argmax(
                tf.stack([
                    self.layer1(
                        self.shared_layer(X_unlabeled, training=True),
                        training=True
                    ),
                    self.layer2(
                        self.shared_layer(X_unlabeled, training=True),
                        training=True
                    ),
                    self.layer3(
                        self.shared_layer(X_unlabeled, training=True),
                        training=True
                    ),
                ]),
                -1,
                output_type=tf.dtypes.int32
            )
            selected = tf.logical_and(selected, m_preds==preds)
            del preds
        
        # Pseudo-label each tri-training layer
        # NB: can be put into loops, but will likely decrease performance
        selected1, y_pseudo1 = pseudo_label(0,
                                            selected,
                                            m_infer,
                                            m_preds,
                                            m_confidence,
                                            self.threshold)
        selected2, y_pseudo2 = pseudo_label(1,
                                            selected,
                                            m_infer,
                                            m_preds,
                                            m_confidence,
                                            self.threshold)
        selected3, y_pseudo3 = pseudo_label(2,
                                            selected,
                                            m_infer,
                                            m_preds,
                                            m_confidence,
                                            self.threshold)
        del selected
        del m_infer
        del m_preds
        del m_confidence
        
        # Isolate pseudo-labeled samples
        X_pseudo1 = X_unlabeled[selected1]
        X_pseudo2 = X_unlabeled[selected2]
        X_pseudo3 = X_unlabeled[selected3]
        
        # Train the layers
        with tf.GradientTape() as tape:
            # Calculate the loss from layer1
            # NB: can be put into loops but will decrease performance
            y_pred1 = self.layer1(
                self.shared_layer(
                    tf.concat([X_labeled, X_pseudo1], 0),
                    training=True
                ),
                training=True
            )
            del X_pseudo1
            loss1 = self.compiled_loss(
                tf.concat([smear(y_labeled, std=self.smear_std), y_pseudo1], 0),
                y_pred1,
                sample_weight=None,
                regularization_losses=(self.shared_layer.losses +
                                       self.layer1.losses),
            )
            del y_pred1
            del y_pseudo1
            
            # Calculate the loss from layer2
            y_pred2 = self.layer2(
                self.shared_layer(
                    tf.concat([X_labeled, X_pseudo2], 0),
                    training=True
                ),
                training=True
            )
            del X_pseudo2
            loss2 = self.compiled_loss(
                tf.concat([smear(y_labeled, std=self.smear_std), y_pseudo2], 0),
                y_pred2,
                sample_weight=None,
                regularization_losses=(self.shared_layer.losses +
                                       self.layer2.losses),
            )
            del y_pred2
            del y_pseudo2
            
            # Calculate the loss from layer3
            y_pred3 = self.layer3(
                self.shared_layer(
                    tf.concat([X_labeled, X_pseudo3], 0),
                    training=True
                ),
                training=True
            )
            del X_pseudo3
            loss3 = self.compiled_loss(
                tf.concat([smear(y_labeled, std=self.smear_std), y_pseudo3], 0),
                y_pred3,
                sample_weight=None,
                regularization_losses=(self.shared_layer.losses +
                                       self.layer3.losses),
            )
            del y_pred3
            del y_pseudo3
            
            # Calculate total loss
            loss = (loss1 + loss2 + loss3) / 3
        
        del loss1
        del loss2
        del loss3
        
        # Compute predictions on all samples
        y_pred = self(X_labeled, training=True)
        del X_labeled
        
        # Compute gradients against total loss
        gradients = tape.gradient(loss, self.trainable_variables)
        del loss
        
        # Update the metrics configured in `compile`
        self.compiled_metrics.update_state(
            y_labeled,
            y_pred,
            sample_weight=None
        )
        del y_labeled
        del y_pred

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
    @property
    def initializer(self, **kwargs):
        """
        Returns a `Ensemble` with the same layers as the tri-training Model.
        Optionally accepts a smear_std argument for the `Ensemble`.
        """
        model = Ensemble(
            self.shared_layer,
            self.layer1,
            self.layer2,
            self.layer3,
            **kwargs
        )
        return model