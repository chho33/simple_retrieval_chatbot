import tensorflow as tf
from tensorflow.python.estimator.canned import metric_keys
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

SPECIAL_TAGS_COUNT = 2

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
        "last_rnn_dim",
        "vocab_size",
        "pretrain_path",
        "pretrain_trainable",
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
        pretrain_trainable=FLAGS.pretrain_trainable,
        vocab_path=FLAGS.vocab_path,
        last_rnn_dim=FLAGS.last_rnn_dim,
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
    vocab_array, vocab_dict = load_vocab(params.vocab_path)
    pretrain_vectors, pretrain_dict = load_pretrain_vectors(params.pretrain_path, vocab=set(vocab_array))
    if params.pretrain_path and params.vocab_path:
        tf.logging.info("Loading pretrain embeddings...")
        special_tags = np.random.uniform(-0.25, 0.25, (SPECIAL_TAGS_COUNT, params.embedding_dim)).astype("float32")
        special_tags = tf.get_variable("special_tag",
                                  initializer=special_tags,
                                  trainable=True)
        pretrain_vectors = tf.get_variable("word_embeddings",
                                       initializer=pretrain_vectors,
                                       trainable=params.pretrain_trainable)
        print('pretrain_vec: ',pretrain_vectors.shape)
        #initializer = build_initial_embedding_matrix(vocab_dict, pretrain_dict, pretrain_vectors, params.embedding_dim)
    else:
        tf.logging.info("No pretrain_vec path specificed, starting with random embeddings.")
        special_tags = np.random.uniform(-0.25, 0.25, (SPECIAL_TAGS_COUNT, params.embedding_dim)).astype("float32")
        special_tags = tf.get_variable("special_tag",
                                  initializer=special_tags,
                                  trainable=True)
        pretrain_vectors = np.random.uniform(-0.25, 0.25, (len(pretrain_dict), params.embedding_dim)).astype("float32")
        pretrain_vectors = tf.get_variable("word_embeddings",
                                       initializer=pretrain_vectors,
                                       trainable=params.pretrain_trainable)
    pretrain_vectors = tf.concat([special_tags,pretrain_vectors],0)
    return pretrain_vectors

def compare_fn(best_eval_result, current_eval_result, default_key = metric_keys.MetricKeys.LOSS):
    '''
    default_key can be: [recall_at_1 | recall_at_2 | recall_at_5 | metric_keys.MetricKeys.LOSS]
    '''
    print('********* best_eval_result: %s **********'%best_eval_result[default_key])
    print('********* current_eval_result: %s **********'%current_eval_result[default_key])
    print('### metric:%s ###'%default_key)
    if not best_eval_result or default_key not in best_eval_result:
      raise ValueError(
          'best_eval_result cannot be empty or no loss is found in it.')

    if not current_eval_result or default_key not in current_eval_result:
      raise ValueError(
          'current_eval_result cannot be empty or no loss is found in it.')

    if 'loss' in default_key.lower():
        return best_eval_result[default_key] > current_eval_result[default_key]
    else:
        return best_eval_result[default_key] < current_eval_result[default_key]


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
