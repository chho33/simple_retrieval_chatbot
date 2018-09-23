import tensorflow as tf
import re
import pandas as pd
import os
from datetime import datetime
dirname = os.path.dirname(os.path.abspath(__file__))

# Data path
tf.flags.DEFINE_string("input_dir", os.path.join(dirname,"data"), "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_string("training_data_path", os.path.join(tf.flags.FLAGS.input_dir,'train.tfrecords'), "path of training data")
tf.flags.DEFINE_string("valid_data_path", os.path.join(tf.flags.FLAGS.input_dir,'validation.tfrecords'), "path of validation data")
tf.flags.DEFINE_string("vocab_path", os.path.join(tf.flags.FLAGS.input_dir,'vocabulary.txt'), "Path to vocabulary.txt file")

with open(tf.flags.FLAGS.vocab_path,'r') as f:
    vocab_size = len([row for row in f.readlines()]) 
# Model Parameters
tf.flags.DEFINE_integer(
  "vocab_size",
  vocab_size,
  "The size of the vocabulary. Only change this if you changed the preprocessing")
## glove = 100, fasttext = 300
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of the embeddings")
tf.flags.DEFINE_string("rnn_dim", '200,100,100,200', "Dimensionality of the RNN cell")
last_rnn_dim = int(tf.flags.FLAGS.rnn_dim.split(',')[-1])
tf.flags.DEFINE_integer("last_rnn_dim", last_rnn_dim, "Dimensionality of the RNN cell")
#tf.flags.DEFINE_integer("max_context_len", 160, "Truncate contexts to this length")
#tf.flags.DEFINE_integer("max_utterance_len", 80, "Truncate utterance to this length")
tf.flags.DEFINE_integer("max_context_len", 80, "Truncate contexts to this length")
tf.flags.DEFINE_integer("max_utterance_len", 55, "Truncate utterance to this length")
# for rnn dropout
tf.app.flags.DEFINE_float('input_keep_prob', '1.0', 'step input dropout of saving model')
tf.app.flags.DEFINE_float('output_keep_prob', '1.0', 'step output dropout of saving model')
tf.app.flags.DEFINE_float('state_keep_prob', '1.0', 'step state dropout of saving model')

# Pre-trained embeddings
pretrain_path = os.path.join(tf.flags.FLAGS.input_dir,'fasttext.%sd.txt'%(vocab_size-2))
if not os.path.exists(pretrain_path):
    from data.etl_utils import shrink_pretrain_vectors
    shrink_pretrain_vectors(os.path.join(tf.flags.FLAGS.input_dir,'fasttext.250057d.txt'),os.path.join(tf.flags.FLAGS.input_dir,'vocabulary.txt'))
tf.flags.DEFINE_string("pretrain_path", pretrain_path, "Path to pre-trained pretrain vectors")
#tf.flags.DEFINE_string("pretrain_path", None, "Path to pre-trained pretrain vectors")

# Training Parameters
tf.flags.DEFINE_integer("train_steps",300000, "Training steps.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 16, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")

# Training Config
tf.flags.DEFINE_string("model_dir",os.path.join(dirname,"models/%s"%datetime.now().strftime('%s')) , "Directory to store model checkpoints (defaults to ./runs)")
#tf.flags.DEFINE_string("model_dir",os.path.join(dirname,"models/%s"%'1537709884') , "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 2000, "Evaluate after this many train steps")
tf.flags.DEFINE_integer("save_summary_steps", 100, "save_summary_steps")
tf.flags.DEFINE_integer("log_step_count_steps", 100, "log_step_count_steps")
tf.flags.DEFINE_integer("save_checkpoints_steps", 500, "save_checkpoints_steps")

# Prediction
tf.flags.DEFINE_string("vocab_processor_file", os.path.join(tf.flags.FLAGS.input_dir,"vocab_processor.bin"), "Saved vocabulary processor file")
df = pd.read_csv(os.path.join(tf.flags.FLAGS.input_dir,"train.csv"))
df = df.query('Label==1')
train_size = df.shape[0]
del df
with open(os.path.join(tf.flags.FLAGS.input_dir,"utterances.csv"),"r") as f:
    utt_size = len([row for row in f.readlines()]) - 1
with open(os.path.join(tf.flags.FLAGS.input_dir,"valid.csv"),"r") as f:
    val_size = len([row for row in f.readlines()]) - 1
with open(os.path.join(tf.flags.FLAGS.input_dir,"test.csv"),"r") as f:
    predict_size = len([row for row in f.readlines()]) - 1
tf.flags.DEFINE_integer("utt_size", train_size, "size of all utterances")
tf.flags.DEFINE_integer("train_size", val_size, "size of train corpus")
tf.flags.DEFINE_integer("val_size", val_size, "size of validation corpus")
tf.flags.DEFINE_integer("predict_size", predict_size, "size of prediction corpus")

# Etl
tf.flags.DEFINE_integer(
  "min_word_frequency", 5, "Minimum frequency of words in the vocabulary")
tf.flags.DEFINE_integer("max_sentence_len", tf.flags.FLAGS.max_context_len, "Maximum Sentence Length")
tf.flags.DEFINE_string("output_dir", os.path.join(dirname,"data"), "Output directory for TFrEcord files (default = './data')")

FLAGS = tf.flags.FLAGS
