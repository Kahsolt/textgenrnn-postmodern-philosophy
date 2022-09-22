from tensorflow.config import get_visible_devices
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, SpatialDropout1D, concatenate
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.models import Model

NUM_GPUS = len(get_visible_devices('GPU'))
USE_CUDNNLSTM = K.backend() == 'tensorflow' and NUM_GPUS > 0


def textgenrnn_model(num_classes, config, optimizer, weights_path=None) -> Model:
    '''
    Builds the model architecture for textgenrnn and loads the specified weights for the model.
    '''

    input = Input(shape=(config['max_length'],), name='input')
    embedded = Embedding(num_classes, config['dim_embeddings'], input_length=config['max_length'], name='embedding')(input)

    if config['dropout']:
        embedded = SpatialDropout1D(config['dropout'], name='dropout')(embedded)

    rnn_layer_list = []
    for i in range(config['rnn_layers']):
        prev_layer = embedded if i == 0 else rnn_layer_list[-1]
        rnn_layer_list.append(new_rnn(config, i+1)(prev_layer))

    seq_concat = concatenate([embedded] + rnn_layer_list, name='rnn_concat')
    attention = AttentionWeightedAverage(name='attention')(seq_concat)
    output = Dense(num_classes, name='output', activation='softmax')(attention)

    model = Model(inputs=[input], outputs=[output])
    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


def new_rnn(config, layer_idx):
    '''
    Create a new LSTM layer per parameters. Unfortunately,
    each combination of parameters must be hardcoded.

    The normal LSTMs use sigmoid recurrent activations
    for parity with CuDNNLSTM:
    https://github.com/keras-team/keras/issues/8860

    FIXME:
    From TensorFlow 2 you do not need to specify CuDNNLSTM.
    You can just use LSTM with no activation function and it will
    automatically use the CuDNN version.
    This part can probably be cleaned up.
    '''

    if USE_CUDNNLSTM:
        rnn = LSTM(config['rnn_size'], return_sequences=True, name=f'rnn_{layer_idx}')
    else:
        rnn = LSTM(config['rnn_size'], return_sequences=True, recurrent_activation='sigmoid', name=f'rnn_{layer_idx}')
    
    return Bidirectional(rnn, name=f'birnn_{layer_idx}') if config['rnn_bidirectional'] else rnn


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for
    a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name=f'{self.name}_W',
                                 trainable=True,
                                 initializer=self.init)
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):        # no reference?
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):     # no reference?
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
