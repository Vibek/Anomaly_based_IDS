from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.regularizers import Regularizer
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import keras.backend as K

class Sparse(Regularizer):
    def __init__(self, rho=0.005, alpha=10):
        self.rho = rho
        self.alpha = alpha

    def __call__(self, x):
        rho_hat = K.mean(x)
        regularization = self.rho * K.log(self.rho / rho_hat) + (1 - self.rho) * K.log((1 - self.rho) / (1 - rho_hat))
        return self.alpha * regularization

    def get_config(self):
        return {'rho:': float(self.rho),
                'alpha': float(self.alpha)
                }

def build_unsw_AE(rho=0.04):

    # first step is to define a AE and pre-training
    
    # Layer 1
    input_traffic = Input(shape=(11, ))
    encoded_1 = Dense(10, activation='relu')(input_traffic)
    encoded_1_bn = BatchNormalization()(encoded_1)
    decoded_1 = Dense(11, activation='relu')(encoded_1_bn)

    autoendoder_1 = Model(inputs=input_traffic, outputs=decoded_1)
    encoder_1 = Model(inputs=input_traffic, outputs=encoded_1_bn)

    # Layer 2
    encoded1_input = Input(shape=(10, ))
    encoded_2 = Dense(8, activation='relu')(encoded1_input)
    encoded_2_bn = BatchNormalization()(encoded_2)
    decoded_2 = Dense(10, activation='relu')(encoded_2_bn)

    autoendoder_2 = Model(inputs=encoded1_input, outputs=decoded_2)
    encoder_2 = Model(inputs=encoded1_input, outputs=encoded_2_bn)

    # Layer 3
    encoded2_input = Input(shape=(8, ))
    encoded_3 = Dense(6, activation='relu')(encoded2_input)
    encoded_3_bn = BatchNormalization()(encoded_3)
    decoded_3 = Dense(8, activation='relu')(encoded_3_bn)

    autoendoder_3 = Model(inputs=encoded2_input, outputs=decoded_3)
    encoder_3 = Model(inputs=encoded2_input, outputs=encoded_3_bn)


	# optimizer
    autoendoder_1.compile(loss='mse', optimizer='adam')
    encoder_1.compile(loss='mse', optimizer='adam')

    autoendoder_2.compile(loss='mse', optimizer='adam')
    encoder_2.compile(loss='mse', optimizer='adam')

    autoendoder_3.compile(loss='mse', optimizer='adam')
    encoder_3.compile(loss='mse', optimizer='adam')

	# model
    model_input = Input(shape=(11,))
    model_encoded_1 = Dense(10, activation='relu')(model_input)
    model_encoded1_bn = BatchNormalization()(model_encoded_1)
    model_encoded_2 = Dense(8, activation='relu')(model_encoded1_bn)
    model_encoded2__bn = BatchNormalization()(model_encoded_2)
    model_encoded_3 = Dense(6, activation='relu')(model_encoded2__bn)
    model_encoded3__bn = BatchNormalization()(model_encoded_3)

    model_decoded_3 = Dense(8, activation='relu')(model_encoded3__bn)
    model_decoded_3_bn = BatchNormalization()(model_decoded_3)
    model_decoded_2 = Dense(10, activation='relu')(model_decoded_3_bn)
    model_decoded_2_bn = BatchNormalization()(model_decoded_2)
    model_decoded_1 = Dense(11, activation='relu')(model_decoded_2_bn)

    ae_model = Model(inputs=model_input, outputs=model_decoded_1)
    ae_encoder = Model(inputs=model_input, outputs=model_encoded3__bn)
    ae_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # second step is to define a classifier and fine-tuning

    return autoendoder_1, encoder_1, autoendoder_2, encoder_2, autoendoder_3, encoder_3, ae_model, ae_encoder

