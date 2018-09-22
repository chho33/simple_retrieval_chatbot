import tensorflow as tf
from flags import FLAGS
import array
import numpy as np
import sys
import os
import re
import csv
from collections import namedtuple, defaultdict

PREDICT = tf.estimator.ModeKeys.PREDICT
EVAL = tf.estimator.ModeKeys.EVAL
TRAIN = tf.estimator.ModeKeys.TRAIN
TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "train.tfrecords"))
VALIDATION_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "validation.tfrecords"))
PREDICTION_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "test.tfrecords"))
UTT_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "utterances.tfrecords"))

TRAIN_CSV = os.path.abspath(os.path.join(FLAGS.input_dir, "train.csv"))
VALIDATION_CSV = os.path.abspath(os.path.join(FLAGS.input_dir, "validation.csv"))
PREDICTION_CSV = os.path.abspath(os.path.join(FLAGS.input_dir, "test.csv"))
UTT_CSV = os.path.abspath(os.path.join(FLAGS.input_dir, "utterances.csv"))

def load_vocab(filename):
    vocab = None
    with open(filename) as f:
        vocab = f.read().splitlines()
    dct = defaultdict(int)
    for idx, word in enumerate(vocab):
        dct[word] = idx
    return [vocab, dct]

def load_pretrain_vectors(filename, vocab):
    """
    Load pretrain evectors from a .txt file.
    Optionally limit the vocabulary to save memory. `vocab` should be a set.
    """
    dct = {}
    vectors = array.array('d')
    current_idx = 0
    with open(filename, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            tokens = line.split(" ")
            word = tokens[0]
            entries = tokens[1:]
            if not vocab or word in vocab:
                dct[word] = current_idx
                vectors.extend(float(x) for x in entries)
                current_idx += 1
        word_dim = len(entries)
        num_vectors = len(dct)
        tf.logging.info("Found {} out of {} vectors in pretrain vec".format(num_vectors, len(vocab)))
        return [np.array(vectors).reshape(num_vectors, word_dim), dct]

def build_initial_embedding_matrix(vocab_dict, pretrain_dict, pretrain_vectors, embedding_dim):
      initial_embeddings = np.random.uniform(-0.25, 0.25, (len(vocab_dict), embedding_dim)).astype("float32")
      for word, pretrain_word_idx in pretrain_dict.items():
          word_idx = vocab_dict.get(word)
          initial_embeddings[word_idx, :] = pretrain_vectors[pretrain_word_idx]
      return initial_embeddings

def get_params():
    HParams = namedtuple(
      "HParams",
      [
        "batch_size",
        "embedding_dim",
        "eval_batch_size",
        "learning_rate",
        "max_context_len",
        "max_utterance_len",
        "optimizer",
        "rnn_dim",
        "vocab_size",
        "pretrain_path",
        "vocab_path"
      ])
    return HParams(
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        vocab_size=FLAGS.vocab_size,
        optimizer=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate,
        embedding_dim=FLAGS.embedding_dim,
        max_context_len=FLAGS.max_context_len,
        max_utterance_len=FLAGS.max_utterance_len,
        pretrain_path=FLAGS.pretrain_path,
        vocab_path=FLAGS.vocab_path,
        rnn_dim=FLAGS.rnn_dim)

def get_feature_columns(mode):
    feature_columns = []

    feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="context", dimension=FLAGS.max_context_len, dtype=tf.int64))
    feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="context_len", dimension=1, dtype=tf.int64))

    feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="utterance", dimension=FLAGS.max_context_len, dtype=tf.int64))
    feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="utterance_len", dimension=1, dtype=tf.int64))

    if mode == TRAIN:
        # During training we have a label feature
        feature_columns.append(tf.contrib.layers.real_valued_column(
          column_name="label", dimension=1, dtype=tf.int64))

    if mode == EVAL:
        # During evaluation we have distractors
        for i in range(9):
            feature_columns.append(tf.contrib.layers.real_valued_column(
              column_name="distractor_{}".format(i), dimension=FLAGS.max_context_len, dtype=tf.int64))
            feature_columns.append(tf.contrib.layers.real_valued_column(
              column_name="distractor_{}_len".format(i), dimension=1, dtype=tf.int64))
    return set(feature_columns)

def get_id_feature(features, key, len_key, max_len):
    # key = context, utterance
    # len_key = context_len, utterance_len
    ids = features[key]
    #print('features: ',features)
    #ids = tf.Print(ids,[ids], '############ features, %s ##########: '%key)
    ids_len = tf.squeeze(features[len_key], [1])
    #ids_len = tf.Print(ids_len,[ids_len], '############ features, %s len ##########: '%key)
    ids_len = tf.minimum(ids_len, tf.constant(max_len, dtype=tf.int64))
    #ids_len = tf.Print(ids_len,[ids_len], '############ features, %s len2 ##########: '%key)
    return ids, ids_len

def get_embeddings(params=FLAGS):
    if params.pretrain_path and params.vocab_path:
        tf.logging.info("Loading pretrain embeddings...")
        vocab_array, vocab_dict = load_vocab(params.vocab_path)
        pretrain_vectors, pretrain_dict = load_pretrain_vectors(params.pretrain_path, vocab=set(vocab_array))
        initializer = build_initial_embedding_matrix(vocab_dict, pretrain_dict, pretrain_vectors, params.embedding_dim)
    else:
      tf.logging.info("No pretrain_vec path specificed, starting with random embeddings.")
      initializer = tf.random_uniform_initializer(-0.25, 0.25)

    return tf.get_variable(
      "word_embeddings",
      initializer=initializer)


# save and load matrix
def sync(f):
	"""
	This makes sure data is written to disk, so that buffering doesn't influence the timings.
	"""
	f.flush()
	os.fsync(f.fileno())

def np_save(arr, pth):
    with open(pth, 'wb+') as f:
        np.save(f, arr, allow_pickle=False)
        sync(f)

# load data
def get_txt(typ='utterance'):
    if typ == 'train': CSV = TRAIN_CSV
    elif typ == 'val': CSV = VALIDATION_CSV
    elif typ == 'predict': CSV = PREDICTION_CSV
    elif typ == 'utterance': CSV = UTT_CSV 
    else: sys.exit('wrong type')

    with open(CSV,'r') as f:
        reader = csv.reader(f)
        next(reader)
        txt = (re.sub(' __eou__','',row[0].strip()) for row in reader)
        txt = list(map(lambda x: ''.join(x.split(" ")),txt))
    return txt
