"""
@author: Tim Janke, Energy Information Networks & Systems Lab @ TU Darmstadt, Germany

Module for IGEP class.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


################################################################################################################
# define energy score
def energy_score(y_true, S):
    """
    Computes energy score:

    Parameters
    ----------
    y_true : tf tensor of shape (BATCH_SIZE, D, 1)
        True values.
    S : tf tensor of shape (BATCH_SIZE, D, N_SAMPLES)
        Predictive samples.

    Returns
    -------
    tf tensor of shape (BATCH_SIZE,)
        Scores.

    """
    beta=1
    n_samples = S.shape[-1]
    def expected_dist(diff, beta):
        return K.sum(K.pow(K.sqrt(K.sum(K.square(diff), axis=-2)+K.epsilon()), beta),axis=-1)
    es_1 = expected_dist(y_true - S, beta)
    es_2 = 0
    for i in range(n_samples):
        es_2 = es_2 + expected_dist(K.expand_dims(S[:,:,i]) - S, beta)
    return es_1/n_samples - es_2/(2*n_samples**2)


# subclass tensorflow.keras.losses.Loss
class EnergyScore(Loss):
    def call(self, y_true, S):
        return energy_score(y_true, S)
################################################################################################################




class igep(object):
    """
    Class for Implicit Generative Ensemble Postprocessing (IGEP) models.
    IGEP models can be used to generate samples from an implicit multivariate predictive distribution.
    See Janke&Steinke (2020): "Probabilistic multivariate electricity price forecasting using implicit generative ensemble post-processing"
    https://arxiv.org/abs/2005.13417
    
    Passing model_type = 1 will create original model.
    Passing model_type = 2 will create new advanced model.

    Parameters
    ----------
    dim_out : int
        Number of output dimensions.
    dim_in_mean : int
        Number of features used for predictive mean.
    dim_in_noise : int
        Number of features used for uncertainty. Will be ignored if model_type=1.
    dim_latent : int
        Number of latent variables.
    n_samples_train : int
        Number of predictive samples to be drawn in training.
        More samples should results in improved accuracy but takes lonbger to train.
    model_type : int, 1 or 2
            1 will create original model, 2 will create an improved and more flexible model.
    latent_dist : sting, optional
        Family of the latent distributions. Options are uniform and normal. The default is "uniform".
    latent_dist_params : tuple, optional
        Parameters for latent distributions. (min,max) for uniform, (mean,stddev) for normal. 
        If None is passed params are set to (-1,1) and (0,1) respectively.

    Returns
    -------
    None.

    """
    
    def __init__(self, dim_out, dim_in_mean, dim_in_noise, dim_latent, n_samples_train, model_type, latent_dist="uniform", latent_dist_params=None):
        
        self.dim_out = dim_out
        self.dim_in_mean = dim_in_mean
        self.dim_in_noise = dim_in_noise
        self.dim_latent = dim_latent
        self.n_samples_train = n_samples_train
        self.latent_dist = latent_dist
        if latent_dist_params is None:
            if self.latent_dist == "uniform":
                self.latent_dist_params = (-1.0, 1.0)
            elif self.latent_dist == "normal":
                self.latent_dist_params = (0.0, 1.0)
        else:
            self.latent_dist_params = latent_dist_params
        
        self._n_samples = n_samples_train
        if model_type == 2:
            self._build_model = self._build_model_2
        elif model_type == 1:
            self._build_model = self._build_model_1
        self.model = self._build_model()

    def _build_model_1(self):
        """
        Defines original IGEP model:
        Variance of dim_out of the latent distributions depend on the ensemble spread.
        See Janke&Steinke (2020): "Probabilistic multivariate electricity price forecasting using implicit generative ensemble post-processing"
        https://arxiv.org/abs/2005.13417

        Returns
        -------
        object
            Keras model.

        """
        
        ### Inputs ###
        x_mean = keras.Input(shape=(self.dim_out, self.dim_in_mean), name = "x_mean")
        delta = keras.Input(shape=(self.dim_out,1), name = "delta")
        bs = K.shape(delta)[0]

        
        ### mean model ###
        mu =  layers.LocallyConnected1D(filters=1, 
                                        kernel_size=1, 
                                        strides=1,
                                        padding='valid',
                                        data_format='channels_last',
                                        use_bias=True,
                                        activation='linear')(x_mean) # [n_dim_out x 1] 
        mu = layers.Lambda(lambda arg: K.repeat_elements(arg, self._n_samples, axis=-1))(mu) # [n_dim_out x n_samples]     
        
        #### noise model ###
        # generate noise
        if self.latent_dist == "uniform":
            u = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                            minval=-1.0, 
                                                            maxval=1.0))([bs, self.dim_out, self._n_samples])
            v = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                            minval=self.latent_dist_params[0], 
                                                            maxval=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])

        elif self.latent_dist == "normal":
            u = layers.Lambda(lambda args: K.random_uniform(shape=(args[0], args[1], args[2]), 
                                                            mean=0.0, 
                                                            stddev=1.0))([bs, self.dim_out, self._n_samples])
            v = layers.Lambda(lambda args: K.random_normal(shape=(args[0], args[1], args[2]), 
                                                            mean=self.latent_dist_params[0], 
                                                            stddev=self.latent_dist_params[1]))([bs, self.dim_latent, self._n_samples])
        
        u = layers.Multiply()([delta, u]) # adapt u samples by ensemble spread
        
        # decode samples from adaptive latent variables
        # ("channels_first" produces an error, therefore we use channels_last + 2 x permute_dims)
        u = layers.Lambda(lambda arg: K.permute_dimensions(arg, (0,2,1)))(u)
        eps_u = layers.Conv1D(filters=self.dim_out, 
                              kernel_size=1,
                              strides=1,
                              padding="valid",
                              data_format="channels_last",
                              activation="linear", 
                              use_bias=False)(u)
        eps_u = layers.Lambda(lambda arg: K.permute_dimensions(arg, (0,2,1)))(eps_u)
        
        # decode samples from independent latent variables
        v = layers.Lambda(lambda arg: K.permute_dimensions(arg, (0,2,1)))(v)
        eps_v = layers.Conv1D(filters=self.dim_out, 
                              kernel_size=1,
                              strides=1,
                              padding="valid",
                              data_format="channels_last",
                              activation="linear", 
                            use_bias=False)(v)
        eps_v = layers.Lambda(lambda arg: K.permute_dimensions(arg, (0,2,1)))(eps_v)
        
        #### add noise to mean ###
        y = layers.Add()([mu, eps_u, eps_v])
        
        return Model(inputs=[x_mean, delta], outputs=y)
    

    def fit(self, x, y, batch_size=32, epochs=10, verbose=0, callbacks=None, validation_split=0.0, validation_data=None, sample_weight=None, optimizer="Adam", plot_learning_curve=False):
        """
        Fits the model to traning data.

        Parameters
        ----------
        x : list of two arrays.
            x contains two arrays as inputs for the model.
            First array contains the inputs for mean model with shape (n_examples, dim_out, dim_in_mean).
            Second array contains the inputs for noise model with shape (n_examples, dim_out, dim_in_noise).
        y : array of shape (n_examples, dim_out, 1)
            Target values.
        batch_size : int, optional
            Number of samples per gradient update. The default is 32.
        epochs : int, optional
            Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
            The default is 10.
        verbose : int, optional
            0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            If not 0 will plot model summary and arachitecture as well as show the learning curve.
            The default is 0.
        callbacks : list of keras.callbacks.Callback instances, optional
            List of callbacks to apply during training. The default is None.
        validation_split : float between 0 and 1, optional
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data, 
            will not train on it, and will evaluate the loss and any model 
            metrics on this data at the end of each epoch. The default is 0.0.
        validation_data : tuple of arrays like (x,y), optional
            Data on which to evaluate the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. The default is None.
        sample_weight : array, optional
            Weights for training examples. The default is None.
        optimizer : string or keras optimizer instance, optional
            Sets options for model optimization. The default is "Adam".
        plot_learning_curve : boolean, optional
            If True, plots the training and validation loss. The default is False.

        Returns
        -------
        None.

        """    

################################################################################################################
        self.model.compile(loss=EnergyScore(), optimizer=optimizer)
        self.history = self.model.fit(x=x, 
                                      y=y,
                                      batch_size=batch_size, 
                                      epochs=epochs, 
                                      verbose=verbose, 
                                      callbacks=callbacks, 
                                      validation_split=validation_split, 
                                      validation_data=validation_data,
                                      shuffle=True,
                                      sample_weight=sample_weight)
################################################################################################################
        if verbose > 0:
            keras.utils.plot_model(self.model, show_shapes=True)
            self.model.summary()
        
        if plot_learning_curve:
            learning_curve_plot(self.history.history)
            
        # create new model with same architecture and weights but 100 samples per call
        self._n_samples = 100
        weights = self.model.get_weights()
        self.model = self._build_model()
        # self.model.compile(loss=EnergyScore(), optimizer=optimizer) # not necessary if only used for prediction
        self.model.set_weights(weights)
        return self
    

    def predict(self, x_test, n_samples=1):
        """
        Draw predictive samples from model.

        Parameters
        ----------
        x_test : list of two arrays.
            x_test contains two arrays as inputs for the model.
            First array contains the inputs for mean model with shape (n_examples, dim_out, dim_in_mean).
            Second array contains the inputs for noise model with shape (n_examples, dim_out, dim_in_noise).
        n_samples : int, optional
            Number of samples to draw. The default is 1.

        Returns
        -------
        array of shape (n_examples, dim_out, n_samples)
            Predictive samples.

        """
        S = []
        for i in range(np.int(np.ceil(n_samples/self._n_samples))):
            S.append(self.model.predict(x_test))
        return np.concatenate(S, axis=2)[:, :, 0:n_samples]


    def get_model(self):
        """
        Just returns the model.

        Returns
        -------
        object
            IGEP model.

        """
        return self.model