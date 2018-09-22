import pandas as pd
import functools
from etl_utils import * 
ALL_PATH = os.path.join(FLAGS.output_dir,'./all.csv')

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VALIDATION_PATH)
test = pd.read_csv(TEST_PATH)

input_iter = create_csv_iter(ALL_PATH)
input_iter = (x[0] + " " + x[1] for x in input_iter)
vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency)

# Create validation.tfrecords
create_tfrecords_file(
    input_filename=VALIDATION_PATH,
    output_filename=os.path.join(FLAGS.output_dir, "validation.tfrecords"),
    example_fn=functools.partial(create_example_test, vocab=vocab))

# Create test.tfrecords
create_tfrecords_file(
    input_filename=TEST_PATH,
    output_filename=os.path.join(FLAGS.output_dir, "test.tfrecords"),
    example_fn=functools.partial(create_example_test, vocab=vocab))

# Create train.tfrecords
create_tfrecords_file(
    input_filename=TRAIN_PATH,
    output_filename=os.path.join(FLAGS.output_dir, "train.tfrecords"),
    example_fn=functools.partial(create_example_train, vocab=vocab))
