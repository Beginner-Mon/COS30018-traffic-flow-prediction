from keras import Sequential, Model, Input
from keras.api.layers import Dense, Dropout, Activation, LSTM, GRU

def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """
    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))
    
    return model

def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """
    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))
    
    return model

def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """
    input_layer = Input(shape=(inputs,))
    hidden_layer = Dense(hidden, activation='sigmoid', name ='hidden')(input_layer)
    hidden_layer = Activation('sigmoid')(hidden_layer)
    dropout_layer = Dropout(0.2)(hidden_layer)
    output_layer = Dense(output, activation='sigmoid')(hidden_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    # model = Sequential()
    # model.add(Dense(hidden, input_dim=inputs, name='hidden'))
    # model.add(Activation('sigmoid'))
    # model.add(Dropout(0.2))
    # model.add(Dense(output, activation='sigmoid'))

    return model

def get_saes(layers):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.

    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        models: List(Model), List of SAE and SAEs.
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    # saes = Sequential()
    # saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    # saes.add(Activation('sigmoid'))
    # saes.add(Dense(layers[2], name='hidden2'))
    # saes.add(Activation('sigmoid'))
    # saes.add(Dense(layers[3], name='hidden3'))
    # saes.add(Activation('sigmoid'))
    # saes.add(Dropout(0.2))
    # saes.add(Dense(layers[4], activation='sigmoid'))

    # input_layer = Input(shape=(layers[0],))  
    # # Reuse encoders from SAEs
    # encoded1 = sae1.layers[1](input_layer)  # Encoder of sae1 (Dense layer)
    # encoded2 = sae2.layers[1](encoded1)     # Encoder of sae2 (Dense layer)
    # encoded3 = sae3.layers[1](encoded2)     # Encoder of sae3 (Dense layer)  
    # # Final layers
    # dropout = Dropout(0.2)(encoded3)
    # output_layer = Dense(layers[4], activation='sigmoid')(dropout)  
    # saes = Model(inputs=input_layer, outputs=output_layer)

    # Create the SAES model using functional API
    input_layer = Input(shape=(layers[0],))
    hidden1 = Dense(layers[1], name='hidden1')(input_layer)
    hidden1 = Activation('sigmoid')(hidden1)
    hidden2 = Dense(layers[2], name='hidden2')(hidden1)
    hidden2 = Activation('sigmoid')(hidden2)
    hidden3 = Dense(layers[3], name='hidden3')(hidden2)
    hidden3 = Activation('sigmoid')(hidden3)
    dropout = Dropout(0.2)(hidden3)
    output_layer = Dense(layers[4], activation='sigmoid')(dropout)

    saes = Model(inputs=input_layer, outputs=output_layer)

    models = [sae1, sae2, sae3, saes]
    return models