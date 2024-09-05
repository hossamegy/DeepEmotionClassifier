import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPool1D, Flatten, LSTM, Bidirectional, Concatenate, Dropout, Input, concatenate
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.regularizers import l2

class Attention(tf.keras.Model):
    """
    Attention mechanism for weighting the importance of different parts of the input sequence.

    Attributes:
    - W1: Dense layer for transforming the input features.
    - W2: Dense layer for transforming the hidden state.
    - V: Dense layer for calculating the attention scores.
    """
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class CnnBlock(Model):
    """
    Convolutional block consisting of two Conv1D layers followed by max pooling and a Dense layer.

    Attributes:
    - kernel_initializer: Kernel initializer for Conv1D and Dense layers.
    - conv1: First Conv1D layer.
    - conv2: Second Conv1D layer.
    - pool: MaxPooling1D layer.
    - flatten: Flatten layer.
    - dense: Dense layer for output.
    """
    def __init__(self, filters, kernel_size):
        super(CnnBlock, self).__init__()
        self.kernel_initializer = HeNormal(seed=42)
        self.conv1 = Conv1D(filters=filters, kernel_size=kernel_size, activation='tanh',
                            kernel_constraint=MaxNorm(max_value=3, axis=[0, 1]),
                            kernel_initializer=self.kernel_initializer,
                            kernel_regularizer=l2(0.01))
        self.conv2 = Conv1D(filters=filters, kernel_size=kernel_size, activation='tanh',
                            kernel_initializer=self.kernel_initializer,
                            kernel_constraint=MaxNorm(max_value=3, axis=[0, 1]))
        self.pool = MaxPool1D(pool_size=2, strides=2)
        self.flatten = Flatten()
        self.dense = Dense(512, activation='tanh', kernel_initializer=self.kernel_initializer)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def get_config(self):
        config = super(CnnBlock, self).get_config()
        config.update({
            "filters": self.conv1.filters,
            "kernel_size": self.conv1.kernel_size,
        })
        return config

class LSTMBlock(Model):
    """
    LSTM block with bidirectional LSTM layers and an attention mechanism.

    Attributes:
    - kernel_initializer: Kernel initializer for LSTM and Dense layers.
    - bi_lstm1: First Bidirectional LSTM layer.
    - bi_lstm2: Second Bidirectional LSTM layer with return state.
    - attention: Attention mechanism layer.
    - dense: Dense layer for output.
    """
    def __init__(self, lstm_unit=64):
        super(LSTMBlock, self).__init__()
        self.kernel_initializer = HeNormal(seed=42)
        self.bi_lstm1 = Bidirectional(LSTM(lstm_unit, return_sequences=True, kernel_initializer=self.kernel_initializer))
        self.bi_lstm2 = Bidirectional(LSTM(lstm_unit, return_sequences=True, return_state=True, kernel_initializer=self.kernel_initializer))
        self.attention = Attention(32)
        self.dense = Dense(512, activation='tanh', kernel_initializer=self.kernel_initializer)

    def call(self, inputs):
        x = self.bi_lstm1(inputs)
        (lstm, forward_h, _, backward_h, _) = self.bi_lstm2(x)
        state_h = Concatenate()([forward_h, backward_h])
        context_vector, _ = self.attention(lstm, state_h)
        x = self.dense(context_vector)
        return x

    def get_config(self):
        config = super(LSTMBlock, self).get_config()
        config.update({
            "lstm_unit": self.bi_lstm1.forward_layer.units,
        })
        return config

class Parallel_Cnn_Lstm_Model(Model):
    """
    A model that combines CNN and LSTM blocks in parallel for feature extraction and classification.

    Attributes:
    - embedding: Embedding layer for input sequences.
    - cnn_block: Instance of CnnBlock for CNN-based feature extraction.
    - lstm_block: Instance of LSTMBlock for LSTM-based feature extraction.
    - dense: Dense layer for merging CNN and LSTM features.
    - dropout: Dropout layer for regularization.
    - output_layer: Final Dense layer for classification.
    """
    def __init__(self, target, max_len, embedding_config, filters, kernel_size, lstm_unit):
        super(Parallel_Cnn_Lstm_Model, self).__init__()
        self.kernel_initializer = HeNormal(seed=42)
        self.num_words, self.embedding_dim = embedding_config
        self.max_len = max_len
        self.embedding = Embedding(self.num_words, self.embedding_dim, input_length=max_len)
        self.cnn_block = CnnBlock(filters, kernel_size)
        self.lstm_block = LSTMBlock(lstm_unit)
        self.dense = Dense(512, activation='tanh', kernel_initializer=self.kernel_initializer)
        self.dropout = Dropout(0.25)
        self.output_layer = Dense(target, activation='softmax', kernel_initializer=self.kernel_initializer)

    def call(self, inputs):
        x = self.embedding(inputs)
        cnn_output = self.cnn_block(x)
        lstm_output = self.lstm_block(x)
        merged = concatenate([cnn_output, lstm_output])
        x = self.dense(merged)
        x = self.dropout(x)
        return self.output_layer(x)

    def get_config(self):
        config = super(Parallel_Cnn_Lstm_Model, self).get_config()
        config.update({
            "target": self.output_layer.units,
            "max_len": self.max_len,
            "embedding_config": (self.num_words, self.embedding_dim),
            "filters": self.cnn_block.conv1.filters,
            "kernel_size": self.cnn_block.conv1.kernel_size,
            "lstm_unit": self.lstm_block.bi_lstm1.forward_layer.units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build_graph(self):
        x = Input(shape=(self.max_len,))
        return Model(inputs=[x], outputs=self.call(x))
