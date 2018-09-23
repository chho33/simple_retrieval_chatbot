import numpy as np
from utils import get_txt
from flags import FLAGS
from datetime import datetime

predictions_txt = get_txt('predict')
pred = np.load('data/predictions.npy')
utterances_txt = get_txt('utterance')
utt = np.load('data/utterances.npy')

with open('result/%s.txt'%(datetime.now().strftime('%s')),'a') as f:
    for i,p in enumerate(pred):
        result = np.matmul(p.reshape(1,1,FLAGS.rnn_dim),utt)
        ix = np.argmax(result)
        print(predictions_txt[i],' ======> ',utterances_txt[ix])
        f.write('%s\t%s\n'%(predictions_txt[i],utterances_txt[ix]))
