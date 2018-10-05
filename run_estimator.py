#!/usr/bin/env python
"""Script to illustrate usage of tf.estimator.Estimator in TF v1.8"""
import tensorflow as tf
#from tensorflow.contrib.learn import MetricSpec
from flags import FLAGS
import functools
from utils import get_embeddings,get_params,get_feature_columns,get_id_feature,compare_fn,\
                  PREDICT,\
                  EVAL,\
                  TRAIN,\
                  TRAIN_FILE,\
                  VALIDATION_FILE,\
                  PREDICTION_FILE

def main(argv=None):
    """Run the training experiment."""
    # Read parameters and input data
    #params = parser.parse_args(argv[1:])
    params = get_params()
    config = tf.estimator.RunConfig(
        #model_dir=params.job_dir,
        model_dir=FLAGS.model_dir,
        save_summary_steps=FLAGS.save_summary_steps,
        log_step_count_steps=FLAGS.log_step_count_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    )
    # Setup the Estimator
    model_estimator = build_estimator(config, params)
    # Setup and start training and validation
    train_spec = tf.estimator.TrainSpec(
        input_fn=get_train_inputs,
        max_steps=FLAGS.train_steps)

    serving_feature_spec = tf.feature_column.make_parse_example_spec(get_feature_columns(EVAL))
    serving_input_receiver_fn =\
      tf.estimator.export.build_parsing_serving_input_receiver_fn(serving_feature_spec)

    compare = functools.partial(compare_fn,default_key='recall_at_5')
    exporter = tf.estimator.BestExporter(
          name="best_exporter",
          serving_input_receiver_fn=serving_input_receiver_fn,
          compare_fn=compare,
          exports_to_keep=3)


    eval_spec = tf.estimator.EvalSpec(
        input_fn=get_eval_inputs,
        steps=300,
        exporters=exporter,
        start_delay_secs=30,  # Start evaluating after 10 sec.
        throttle_secs=300  # Evaluate only every 30 sec
    )
    tf.estimator.train_and_evaluate(model_estimator, train_spec, eval_spec)

def build_estimator(config, params):
    """
    Build the estimator based on the given config and params.

    Args:
        config (RunConfig): RunConfig object that defines how to run the Estimator.
        params (object): hyper-parameters (can be argparse object).
    """
    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=params,
    )

def model_fn(features, labels, mode, params=FLAGS):
    """Model function used in the estimator.

    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
        params (object): hyper-parameters (can be argparse object).

    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """
    context, context_len = get_id_feature(
        features, "context", "context_len", FLAGS.max_context_len)
    utterance, utterance_len = get_id_feature(
        features, "utterance", "utterance_len", FLAGS.max_utterance_len)

    # Define model's architecture
    if mode == TRAIN:
        logits = architecture(
            params,
            mode,
            context,
            context_len,
            utterance,
            utterance_len
        )
    elif mode == PREDICT:
        logits = architecture(
            params,
            mode,
            context,
            context_len,
            utterance,
            utterance_len
        )
    elif mode == EVAL:
        batch_size = labels.get_shape().as_list()[0]

        # We have 10 exampels per record, so we accumulate them
        all_contexts = [context]
        all_context_lens = [context_len]
        all_utterances = [utterance]
        all_utterance_lens = [utterance_len]
        all_targets = [tf.ones([batch_size, 1], dtype=tf.int64)]

        for i in range(9):
            distractor, distractor_len = get_id_feature(features,
                "distractor_{}".format(i),
                "distractor_{}_len".format(i),
                FLAGS.max_utterance_len)
            #print('========context=========: ',context)
            all_contexts.append(context)
            all_context_lens.append(context_len)
            all_utterances.append(distractor)
            all_utterance_lens.append(distractor_len)
            all_targets.append(
              tf.zeros([batch_size, 1], dtype=tf.int64)
            )
        logits = architecture(
            params,
            mode,
            tf.concat(all_contexts,0),
            tf.concat(all_context_lens,0),
            tf.concat(all_utterances,0),
            tf.concat(all_utterance_lens,0)
        )
        labels = tf.concat(all_targets,0)

    # Apply sigmoid to convert logits to probabilities
    if mode == EVAL:
        #print('EVAL')
        #print('-------------')
        #print('logits: ',logits)
        probs = tf.sigmoid(logits)
        #print('probs: ',probs)
        split_probs = tf.split(probs, 10, 0)
        #print('split_probs: ',split_probs)
        probs = tf.concat(split_probs,1)
        #print('probs: ',probs)
        #print('==============')
    elif mode == TRAIN:
        #print('TRAIN')
        #print('-------------')
        #print('logits: ',logits)
        probs = tf.sigmoid(logits)
        #print('probs: ',probs)
        #print('==============')
    elif mode == PREDICT:
        #probs = logits
        probs = tf.sigmoid(logits)

    # Setup the estimator according to the phase (Train, eval, predict)
    loss = None 
    train_op = None
    export_outputs = None
    eval_metric_ops = {}
    # Loss will only be tracked during training or evaluation.
    if mode in (TRAIN, EVAL):
        # Calculate the binary cross-entropy loss
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(labels))

        # Mean loss across the batch of examples
        loss = tf.reduce_mean(losses, name="mean_loss")
        # Training operator only needed during training.
    if mode == TRAIN:
        train_op = get_train_op_fn(loss, params)
    # Evaluation operator only needed during evaluation
    elif mode == EVAL:
        split_labels = tf.split(labels, 10, 0)
        labels = tf.concat(split_labels,1)
        for k in [1, 2, 5, 10]:
            #print('EVAL RECALL@%s::::::::> '%k)
            #print('probs: ',probs)
            #print('labels: ',labels)
            eval_metric_ops["recall_at_%d" % k] = tf.contrib.metrics.streaming_sparse_recall_at_k(predictions=probs,labels=labels,k=k)
            #eval_metric_ops["recall_at_%d" % k] = MetricSpec(metric_fn=functools.partial(
            #tf.contrib.metrics.streaming_sparse_recall_at_k,
            #k=k))
    elif mode == PREDICT:
        export_outputs = {
          'prediction': tf.estimator.export.PredictOutput({'scores': probs})}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=probs,
        export_outputs=export_outputs, 
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )

def architecture(
    params, 
    mode, 
    context,
    context_len,
    utterance,
    utterance_len,
    ):
    """Return the output operation following the network architecture.
    Returns:
         Logits output Op for the network.
    """
    #if mode == TRAIN:
    #    print('TRAIN:::::')
    #    print('context: ',context)
    #    print('context_len: ',context_len)
    #    print('utterance: ',utterance)
    #    print('utterance_len: ',utterance_len)
    #elif mode == EVAL:
    #    print('EVAL:::::')
    #    print('context: ',context)
    #    print('context_len: ',context_len)
    #    print('utterance: ',utterance)
    #    print('utterance_len: ',utterance_len)

    # Initialize embedidngs randomly or with pre-trained vectors if available
    embeddings_W = get_embeddings(params)
  
    #context = tf.Print(context,[context], '############ context ##########: ')
    # Embed the context and the utterance
    context_embedded = tf.nn.embedding_lookup(
        embeddings_W, context, name="embed_context")
  
    if mode != PREDICT:
        utterance_embedded = tf.nn.embedding_lookup(
            embeddings_W, utterance, name="embed_utterance")
    #print('context_embedded: ',context_embedded)
    #print('utterance_embedded: ',utterance_embedded)
  
    with tf.variable_scope("rnn") as vs:
        # We use an LSTM Cell
        #cell = tf.nn.rnn_cell.LSTMCell(
        #             params.rnn_dim,
        #             forget_bias=2.0,
        #             use_peepholes=True)

        rnn_dims = params.rnn_dim.split(',')
        cell = [ tf.nn.rnn_cell.LSTMCell(
                     int(rnn_dim),
                     forget_bias=2.0,
                     use_peepholes=True) for rnn_dim in rnn_dims]
        cell = tf.nn.rnn_cell.MultiRNNCell(cell) 
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=FLAGS.input_keep_prob,
            output_keep_prob=FLAGS.output_keep_prob,
            state_keep_prob=FLAGS.state_keep_prob)
        # Run the utterance and context through the RNN
        #context_len = tf.Print(context_len,[context_len],"context_len: ")
        #utterance_len = tf.Print(utterance_len, [utterance_len], 'utterance_len: ')
        if mode != PREDICT:
            tmp_concat = tf.concat([context_embedded, utterance_embedded],0)
            tmp_concat_len = tf.concat([context_len, utterance_len],0)
        else:
            tmp_concat = context_embedded
            tmp_concat_len = context_len
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell,
            tmp_concat,
            tmp_concat_len,
            dtype=tf.float32)
        if isinstance(rnn_states,list) or isinstance(rnn_states,tuple):
            rnn_states = rnn_states[0]
        #context_embedded:  Tensor("embed_context:0", shape=(64, 160, 300), dtype=float32)
        #utterance_embedded:  Tensor("embed_utterance:0", shape=(64, 160, 300), dtype=float32)
        #tf.concat([context_embedded, utterance_embedded]:  Tensor("concat:0", shape=(128, 160, 300), dtype=float32)
        #tf.concat([context_len, utterance_len]:  Tensor("concat_1:0", shape=(128,), dtype=int64)
        if mode != PREDICT:
            encoding_context, encoding_utterance = tf.split(rnn_states.h,2,0)
        else:
            encoding_context = rnn_states.h
  
    with tf.variable_scope("prediction") as vs:
        M = tf.get_variable("M",
          shape=[FLAGS.last_rnn_dim, FLAGS.last_rnn_dim],
          initializer=tf.truncated_normal_initializer())
    
        # "Predict" a  response: c * M
        generated_response = tf.matmul(encoding_context, M)
        generated_response = tf.expand_dims(generated_response, 2)
        if mode == PREDICT:
            return generated_response
        encoding_utterance = tf.expand_dims(encoding_utterance, 2)
    
        # Dot product between generated response and actual response
        # (c * M) * r
        #logits = tf.batch_matmul(generated_response, encoding_utterance, True)
        logits = tf.matmul(generated_response, encoding_utterance, True)
        logits = tf.squeeze(logits, [2])
        return logits

def get_train_op_fn(loss, params):
    """Get the training Op.

    Args:
         loss (Tensor): Scalar Tensor that represents the loss function.
         params (object): Hyper-parameters (needs to have `learning_rate`)

    Returns:
        Training Op
    """
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

    """
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params.learning_rate,
        clip_gradients=10.0,
        optimizer=params.optimizer)
    """
    return train_op

def get_inputs_fn(mode,path=None):
    """Return the input function to get the training data.

    Returns:
        DataSet: A tensorflow DataSet object to represent the training input
                 pipeline.
    """
    if mode == TRAIN:
        input_files = [TRAIN_FILE]
        batch_size = FLAGS.batch_size
        randomize_input = True
    elif mode == EVAL:
        input_files = [VALIDATION_FILE]
        batch_size = FLAGS.eval_batch_size
        randomize_input = True
    elif mode == PREDICT:
        if path: 
            input_files = [path]
        else:
            input_files = [PREDICTION_FILE]
        batch_size = 1
        randomize_input = False

    features = tf.contrib.layers.create_feature_spec_for_parsing(
        get_feature_columns(mode))

    feature_map = tf.contrib.learn.read_batch_features(
        file_pattern=input_files,
        batch_size=batch_size,
        features=features,
        reader=tf.TFRecordReader,
        randomize_input=randomize_input,
        num_epochs=FLAGS.num_epochs,
        queue_capacity=400000 + batch_size * 10,
        name="read_batch_features_{}".format(mode))

    if mode == TRAIN:
      #print('features: ',features)
      #print('feature_map: ',feature_map)
      target = feature_map.pop("label")
    else:
      # In evaluation we have 10 classes (utterances).
      # The first one (index 0) is always the correct one
      #print('feature_map: ',feature_map)
      target = tf.zeros([batch_size, 1], dtype=tf.int64)
    return feature_map, target

def get_train_inputs():
    #dataset = tf.data.TFRecordDataset(FLAGS.training_data_path)
    #dataset = dataset.shuffle(
    #    buffer_size=100000, reshuffle_each_iteration=True
    #).repeat(count=None).batch(FLAGS.batch_size)
    #"""
    #dataset = <BatchDataset shapes: (?,), types: tf.string>
    #"""
    #""" 0 (64,) () <dtype: 'string'> tf.Tensor(b'\n\xb4\x03\n\x0e\n\x05label\x12\x05\x1a\x03\n\x01\x00\n\xb6\x01\n\x07context\x12\xaa\x01\x1a\xa7\x01\n\xa4\x01\xb21\xa5\x10\xdc\x18\x00\x0f\x81\x04\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x16\n\rutterance_len\x12\x05\x1a\x03\n\x01\x0c\n\x14\n\x0bcontext_len\x12\x05\x1a\x03\n\x01\x07\n\xba\x01\n\tutterance\x12\xac\x01\x1a\xa9\x01\n\xa6\x01} \xf48\xad\x0bF\xb8\x03\x00\x95\x01\x1d\x94\x07\xfd\x02\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', shape=(), dtype=string)"""
    #return dataset
    return get_inputs_fn(TRAIN)

def get_eval_inputs():
    """Return the input function to get the validation data.

    Returns:
        DataSet: A tensorflow DataSet object to represent the validation input
                 pipeline.
    """
    #dataset = tf.data.TFRecordDataset(FLAGS.valid_data_path)
    #dataset = dataset.batch(FLAGS.batch_size)
    return get_inputs_fn(EVAL)
    


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()

