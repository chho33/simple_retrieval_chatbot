import pandas as pd
import functools
from etl_utils import * 
ALL_PATH = os.path.join(FLAGS.output_dir,'all.csv')
UTT_PATH = os.path.join(FLAGS.output_dir,'utterances.csv') 
UTT_TFR_PATH = os.path.join(FLAGS.output_dir,'utterances.tfrecords') 

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VALIDATION_PATH)
test = pd.read_csv(TEST_PATH)

train = train.query('Label==1')
train_u = train['Utterance']
val_u = val['Ground Truth Utterance']
test_u = test['Ground Truth Utterance']
train_u.name = 'Context'
val_u.name = 'Context'
test_u.name = 'Context'

df = pd.concat([train_u,val_u,test_u],axis=0)
df = pd.DataFrame(df)
#df = df.assign(Utterance=df.Context)
#df = df.assign(Label=0)
df.to_csv(UTT_PATH,index=False)

input_iter = create_csv_iter(ALL_PATH)
input_iter = (x[0] + " " + x[1] for x in input_iter)
vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency)
create_tfrecords_file(
    input_filename=UTT_PATH,
    output_filename=UTT_TFR_PATH,
    example_fn=functools.partial(create_example_train_utt,vocab=vocab))
