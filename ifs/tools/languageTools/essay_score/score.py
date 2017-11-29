#!/bin/python

import sys
import os

from optparse import OptionParser

from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import tensorflow as tf


WORD2VEC_FILE="models/word2vec.model"

parser = OptionParser()
parser.add_option('-f', '--file', dest='filename', 
                  help='Input file', metavar='FILE')
parser.add_option('-j', '--json', dest='json', action='store_true',
                  help='Use IFS-compatible output')
parser.add_option('-m', '--model', dest='model', metavar="MODEL",
                  default="models/current_rnn_model",
                  help='Specify Neural Network Model')
parser.add_option('-w', '--word2vec', dest='word_model', metavar="MODEL",
                  default=WORD2VEC_FILE,
                  help='Specify Word2Vec Model')

(options, args) = parser.parse_args()

model      = options.model

## start train.py code (RNN + model )
# params from FLAGS (but formated to be variables)
g_n_out = 1
g_n_in = 50
g_max_nonimprove = 50
g_l2_beta = 0.01
g_rand_range = 1.0
g_forget_bias = 1.0
g_hidden_layers = [ 100, 200 ]

class RNN(object):

    def __init__(self, max_n_steps):
        self.hidden_layers   = g_hidden_layers
        self.n_out           = g_n_out
        self.n_in            = g_n_in
        self.l2_beta         = g_l2_beta
        self.forget_bias     = g_forget_bias
        self.rand_range      = g_rand_range
        self.max_n_steps     = max_n_steps
        params               = self.def_param()

        self.W_o             = params['W_o']
        self.b_o             = params['b_o']
        self.W_h             = params['W_h']
        self.b_h             = params['b_h']
        self.x               = params['in_x']
        self.y               = params['lbl_y']
        self.data_size       = params['data_size']
        self.n_steps         = params['n_steps']
        

        learned_param        = [self.W_o, self.b_o, self.W_h, self.b_h]
        self.activation      = tf.tanh


        
        # TF cannot perform 3d mult 2d atm so below is a hack
        x_orig_shape = tf.shape(self.x)
        x2d = tf.reshape(self.x,[tf.shape(self.x)[0]*tf.shape(self.x)[1], tf.shape(self.x)[2]]) 
        h_in_rs = tf.matmul(x2d,self.W_h) + self.b_h
        h_in = tf.reshape(h_in_rs,[x_orig_shape[0], x_orig_shape[1], self.hidden_layers[0]])

        # just need h_os_raw
        def lstm_cell(hidden):
            return tf.nn.rnn_cell.BasicLSTMCell(hidden,forget_bias=self.forget_bias)

        #lstm_layer = lstm_cell(self.hidden_layers[0])
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(n) for n in self.hidden_layers])
        h_os_raw, _ = tf.nn.dynamic_rnn(stacked_lstm, h_in, dtype="float32")
       
        # NOTE: because recurrent weights are randomly initialized, the zero rows are no longer zero.

        # Gathering only the last outputs of the sequence.
        # TF does not support advanced indexing right now. Below is a hacky way to get around it.
        # we make use of the current batche's longest seq and the reshaping of 3d to 2d to create
        # an index list so tf.gather can be applied to it.
        h_os_raw2d = tf.reshape(h_os_raw, [tf.shape(h_os_raw)[0]*tf.shape(h_os_raw)[1], tf.shape(h_os_raw)[2]])
        starts = tf.range(0, tf.shape(self.n_steps)[0])*self.max_n_steps 
        n_step_idx = tf.range(0, tf.shape(self.n_steps)[0])*self.max_n_steps + (tf.subtract(self.n_steps,1))
        ranges = tf.stack([starts, n_step_idx], axis=1) # ranges of values in a tensor like [[start, stop]]
        h_os = tf.gather(h_os_raw2d,n_step_idx, name='Gather_last_out')


        # take a range, take raw values from that range, perform sigmoid, then take the mean from all timesteps
        def take_means(lst):
            range = tf.range(tf.cast(lst[0], tf.int32), tf.cast(lst[1], tf.int32))
            vals = tf.gather(h_os_raw2d, range)
            s = tf.sigmoid(tf.matmul(vals, self.W_o) + self.b_o)
            return tf.reduce_mean(s)

        means = tf.map_fn(take_means, ranges, dtype=tf.float32)

        # takes the mean score from all the time steps
        self.y_pred_mean = tf.reshape(means, [tf.constant(1), tf.shape(h_os_raw)[0], tf.constant(1)])

        # predicate at the _last_ timestep rather than the mean predicate 
        self.y_pred = [tf.sigmoid(tf.matmul(h_os, self.W_o) + self.b_o)]

        # simple mean-squared loss on labels
        self.loss = tf.reduce_mean((self.y_pred - self.y)**2) + self.l2()
    
    def l2(self):
        reg = tf.nn.l2_loss(self.W_h)
        return reg*self.l2_beta

    def def_param(self):
        """ define parameters required for TF graph.
        """
        rand_range=self.rand_range # 0.01
        #n initialization from Glorot and Bengio 2010.
        W_hid = tf.Variable(np.random.uniform(low=-1*rand_range, high=1*rand_range,\
                size=(self.n_in, self.hidden_layers[0])), dtype='float32', trainable = True, name='W_h')

        b_hid = tf.Variable(np.zeros([self.hidden_layers[0]], dtype='float32'), name='b_h')

        W_out = tf.Variable(np.random.uniform(low=-1*rand_range, high=1*rand_range,\
                size=(self.hidden_layers[-1], self.n_out)), dtype='float32',trainable= True,  name='W_o')

        b_out =  tf.Variable(tf.zeros([self.n_out]), name='b_o')

        input_x = tf.placeholder(tf.float32, [None, self.max_n_steps, self.n_in])
        # N_OUT == 1
        label_y = tf.placeholder(tf.float32, [None, self.n_out], name='label_y')

        data_size = tf.placeholder(tf.float32, [None], name= 'data_size')

        n_steps = tf.placeholder('int32', [None], name='n_steps')

        params = {
            'W_o'       : W_out,
            'b_o'       : b_out,
            'W_h'       : W_hid,
            'b_h'       : b_hid,
            'in_x'      : input_x,
            'lbl_y'     : label_y,
            'data_size' : data_size,
            'n_steps'   : n_steps
            }

        return params
## end of train.py code (RNN)

## start of preprocess.py code 
def make_list(raw_documents):
    vectorizer = CountVectorizer(stop_words=None)
    analyze = vectorizer.build_analyzer()
    x = []
    for doc in raw_documents:
        x.append(analyze(doc))
    return x

def get_feature_from(model):
    m = model.wv.syn0.mean(axis=0)  # mean vector
    def get_feature(x):
        try:
            return model[x]
        except KeyError:
            # no matching vector, replace with mean vector
            return m
    return get_feature

## end of preprocess.py code

def run_rnn(str):
    return 0.75

def build_json(filename, score):
    return """{
    "feedback": [],
    "feedbackStats": [
      {
        "type": "stat",
        "toolName": "essay_score",
        "name": "essayScoreStat",
        "level": "basic",
        "category": "score",
        "filename": "%s",
        "statName": "Estimated Essay Score",
        "statValue": %d
      }
    ]
}""" % (filename, int(score*100))

def build_output(filename, score):
    return "Score: %d" % (int(score*100))

def load_word2vec(file):
    return word2vec.Word2Vec.load(file)

def run_rnn(data_X):
    steps = len(data_X)
    max_steps = 1031 # based on training + validation data

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    model = RNN(steps)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, options.model)

        pred_y = sess.run([model.y_pred], \
            feed_dict={model.x: [data_X], 
                       model.y: [[0]], 
                       model.n_steps: [steps]})

        pred_y = pred_y[0][0][0]
        return pred_y

    return -0.01

def main():
    file = sys.stdin
    if options.filename is not None:
        file = open(options.filename, "r")
    else:
        print "Reading from stdin. Press ^D to stop."

    if file is None:
        print "Could not open file %s" % options.filename

    str = file.readlines()
    file.close()
    str = ' '.join(str)
    str = make_list([str])[0]

    word_model = load_word2vec(options.word_model);
    essay_vectors = map(get_feature_from(word_model), str)

    score = run_rnn(essay_vectors)

    outfunc = build_json if options.json else build_output
    print(outfunc(options.filename, score))

    return 0

if __name__ == "__main__":
    main()
