import numpy as np
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout
from cxplain.backend.model_builders.base_model_builder import BaseModelBuilder


import collections
import tensorflow as tf
from functools import partial
import tensorflow.keras.backend as K
# from abc import ABCMeta, abstractmethod
from tensorflow.python.keras.models import Model
# from cxplain.backend.validation import Validation
from cxplain.backend.causal_loss import causal_loss_precalculated
from cxplain.backend.masking.masking_util import MaskingUtil
# from tensorflow.python.keras.backend import resize_images, resize_volumes
# from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Input, Dense, Flatten, Lambda, Reshape
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping


class CustomMLPModelBuilder(BaseModelBuilder):
    def __init__(self, num_layers=2, num_units=64, activation="relu", with_bn=False, p_dropout=0.0,
                 callbacks=list([]), early_stopping_patience=12,
                 batch_size=64, num_epochs=100, validation_fraction=0.1, shuffle=True,
                 learning_rate=0.0001, optimizer=None, verbose=0, n_feature_groups=-1):
        super(CustomMLPModelBuilder, self).__init__(callbacks, early_stopping_patience, batch_size, num_epochs,
                                              validation_fraction, shuffle, learning_rate, optimizer, verbose)
        self.num_layers = num_layers
        self.num_units = num_units
        self.activation = activation
        self.with_bn = with_bn
        self.p_dropout = p_dropout
        self.n_feature_groups = n_feature_groups

    def build(self, input_layer):
        last_layer = input_layer
        for _ in range(self.num_layers):
            last_layer = Dense(self.num_units, activation=self.activation)(last_layer)
            if self.with_bn:
                last_layer = BatchNormalization()(last_layer)
            if not np.isclose(self.p_dropout, 0):
                last_layer = Dropout(self.p_dropout)(last_layer)
        return last_layer


    def build_explanation_model(self, input_dim, output_dim=None, loss=None, downsample_factors=(1,)):
        num_indices, num_channels, steps, downsampling_factor =\
            MaskingUtil.get_input_constants(input_dim, downsample_factors)

        if self.n_feature_groups > 0:
            num_indices = self.n_feature_groups

        if downsampling_factor != 1 and num_indices is None:
            raise ValueError("Attribution downsampling is not supported for variable length inputs. "
                             "Please pad your data samples to the same size to use downsampling.")

        input_shape = (input_dim,) if not isinstance(input_dim, collections.Sequence) else input_dim
        input_layer = Input(shape=input_shape)
        last_layer = self.build(input_layer)

        if num_indices is None:
            last_layer = Dense(1, activation="linear")(last_layer)
            last_layer = Flatten()(last_layer)  # None * None outputs
            last_layer = Lambda(K.softmax, output_shape=K.int_shape(last_layer))(last_layer)
        else:
            last_layer = Flatten()(last_layer)
            last_layer = Dense(num_indices, activation="softmax")(last_layer)

        # Prepare extra inputs for causal loss.
        # all_auxiliary_outputs = Input(shape=(output_dim,), name="all")
        # all_but_one_auxiliary_outputs_input = Input(shape=(num_indices, output_dim), name="all_but_one")
        # omega = Input(shape=(num_indices,), name="omega")

        # if num_indices is not None:
            # all_but_one_auxiliary_outputs = Lambda(lambda x: tf.unstack(x, axis=1))(all_but_one_auxiliary_outputs_input)
            # if K.int_shape(all_but_one_auxiliary_outputs_input)[1] == 1:
                # all_but_one_auxiliary_outputs = [all_but_one_auxiliary_outputs]
        # else:
            # all_but_one_auxiliary_outputs = all_but_one_auxiliary_outputs_input

        # causal_loss_fun = partial(causal_loss_precalculated,
        #                           attention_weights=last_layer,
        #                           omega=omega)
        #                           # auxiliary_outputs=all_auxiliary_outputs,
        #                           # all_but_one_auxiliary_outputs=all_but_one_auxiliary_outputs,
        #                           # loss_function=loss)
        # causal_loss_fun.__name__ = "causal_loss_precalculated"

        # We must connect all inputs to the output to bypass a bug in model saving in tf < 1.15.0rc0.
        # For easier handling when calling .fit(), we transform all outputs to be the same shape.
        # See https://github.com/tensorflow/tensorflow/pull/30244
        # all_but_one_same_shape_output_layer = Lambda(lambda x: x[:, 0, :])(all_but_one_auxiliary_outputs_input)

        model = Model(inputs=[input_layer],
                      outputs=[last_layer])
        model = self.compile_model(model, main_losses=[causal_loss_precalculated], loss_weights=[1.0],
                                   learning_rate=self.learning_rate, optimizer=self.optimizer)

        prediction_model = Model(input_layer, last_layer)
        return model, prediction_model

    def fit(self, model, X, y, model_filepath, omega):
        callbacks = [
            ModelCheckpoint(filepath=model_filepath,
                            save_best_only=True,
                            save_weights_only=True),
            EarlyStopping(patience=self.early_stopping_patience)
        ] + self.callbacks

        # Perform an initial model save so that one version of the model is always saved
        # even if model fitting or check-pointing fails.
        model.save_weights(model_filepath)

        history = model.fit(x=[X], 
                            y=[omega],
                            batch_size=self.batch_size,
                            shuffle=self.shuffle,
                            validation_split=self.validation_fraction,
                            epochs=self.num_epochs,
                            verbose=self.verbose,
                            callbacks=callbacks)

        # Restore to best encountered model.
        model.load_weights(model_filepath)
        return history