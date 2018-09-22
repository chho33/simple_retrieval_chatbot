#!/usr/bin/env python
import os
import functools
import numpy as np
import tensorflow as tf
from run_estimator import model_fn, get_inputs_fn, get_train_inputs
from flags import FLAGS
from utils import get_params,np_save,PREDICT,UTT_FILE

vocab_mapping = {}
with open(FLAGS.vocab_path,'r') as f:
    for i,w in enumerate(f.readlines()):
        vocab_mapping[w.strip()] = i 

INPUT_CONTEXT = "電腦 都 撐 幾年 ? ? ? ? ?"
POTENTIAL_RESPONSES = ["還好吧 我 自組 到 現在 4年 了 還 頭好壯壯", "甲鐵城 的 卡巴斯基 . . . . . ."]
#vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore('./data/vocab_processor.bin')

def infer_manual(argv=None):
    """Run the inference and print the results to stdout."""
    # Initialize the estimator and run the prediction
    model_estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        params=get_params()
    )
    
    for res in POTENTIAL_RESPONSES:
        result = model_estimator.predict(input_fn=lambda: test_inputs(INPUT_CONTEXT,res))

        for r in result:
            print('response: ',res, 'dot: ', r[0])

def infer(mode='predict_context'):
    """Run the inference and print the results to stdout."""
    # Initialize the estimator and run the prediction
    model_estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        params=get_params()
    )
    if mode == 'predict_context':
        result = model_estimator.predict(input_fn=get_prediction_inputs)
        size_limit = FLAGS.predict_size
    elif mode == 'all_utterances':
        result = model_estimator.predict(input_fn=lambda: get_prediction_inputs(path=UTT_FILE))
        size_limit = FLAGS.utt_size
    results = []
    for i, r in enumerate(result):
        if i == size_limit: break
        results.append(r)
    return results

def infer_utterances(argv=None):
    return infer(mode='all_utterances')

def test_inputs(context,utterance):
    """Returns training set as Operations.
    Returns:
        (features, ) Operations that iterate over the test set.
    """
    with tf.name_scope('Test_data'):
        features,_ = get_features(context,utterance)
        #print('features: ', features)
        #dataset = tf.data.Dataset.from_tensor_slices((features,))
        dataset = tf.data.Dataset.from_tensor_slices(features)
        #print('dataset: ', dataset)
        # Return as iteration in batches of 1
        return dataset.batch(1).make_one_shot_iterator().get_next()

def get_prediction_inputs(path=None):
    return get_inputs_fn(PREDICT,path)

def map_vocab(vocab):
    vocab = vocab.split(" ")
    vocab = list(map(lambda x: vocab_mapping[x] ,vocab)) 
    return np.array([vocab])

def get_features(context,utterance):
    #print('context: ',context)
    #print('utterance: ',utterance)
    #context_matrix = np.array(list(vp.transform([context])))
    #utterance_matrix = np.array(list(vp.transform([utterance])))
    context_matrix = map_vocab(context) 
    utterance_matrix = map_vocab(utterance) 
    context_len = len(context.split(" "))
    utterance_len = len(utterance.split(" "))
    features = {
      "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
      "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
      "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
      "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
    }
    return features, None

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    pred_npy_path = os.path.join(FLAGS.input_dir,'predictions.npy')
    utt_npy_path = os.path.join(FLAGS.input_dir,'utterances.npy')
    #tf.app.run(main=infer_predict)
    #tf.app.run(main=infer_train)
    predictions = infer()
    np_save(predictions,pred_npy_path)
    utterances = infer_utterances()
    np_save(utterances,utt_npy_path)
