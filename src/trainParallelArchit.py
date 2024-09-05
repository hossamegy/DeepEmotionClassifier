from tensorflow.keras.utils import plot_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from parallelCnnLstmModel import Parallel_Cnn_Lstm_Model
from helper_functions.data_loader import data_importer
from helper_functions.data_cleaner import data_cleaner
from helper_functions.data_preprocessing import data_preprocessor
from helper_functions.model_trainer import model_trainer
from helper_functions.model_evaluator import model_evaluation

# Configuration and Parameters
df_path = r"dataset\train.csv"  # Path to the dataset
input_column_name = "Text"  # Column name for input text data
num_words = 100000  # Maximum number of words to consider in the tokenizer
oov_token = '<OOV>'  # Token to represent out-of-vocabulary words
type_padding = 'post'  # Padding type for sequences
type_truncating = 'post'  # Truncating type for sequences
maxlen = 40  # Maximum length of input sequences

# Step 1: Import Dataset
dataset = data_importer(df_path)

# Step 2: Clean the Dataset
dataset = data_cleaner(input_column_name, "Emotion", dataset)

# Step 3: Preprocess Data
all_input_sequence_data, all_label, le, tokenizer = data_preprocessor(
    input_column_name, "Emotion", dataset, num_words, oov_token, type_padding, type_truncating, maxlen
)

# Define Model Parameters
target = 6  # Number of output classes
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size
embedding_dim = 64  # Dimension of the embedding vector
embedding_config = (vocab_size, embedding_dim)  # Embedding configuration
filters = 64  # Number of convolutional filters
kernel_size = 3  # Size of the convolutional kernel
lstm_unit = 128  # Number of units in the LSTM layer

# Initialize and Build Model
model = Parallel_Cnn_Lstm_Model(target, maxlen, embedding_config, filters, kernel_size, lstm_unit)
model.build_graph().summary()  # Print model summary

# Step 4: Train the Model
his, model = model_trainer(all_input_sequence_data, all_label, model)

# Step 5: Evaluate the Model
model_evaluation(his, model, all_input_sequence_data, all_label)
