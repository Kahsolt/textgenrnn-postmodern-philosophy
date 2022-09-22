#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/18 

import os
from argparse import ArgumentParser

from textgenrnn import textgenrnn
from util import *


def train(args):
  ''' Data '''
  corpus_fp = os.path.join(CORPUS_PATH, f'{args.corpus}.txt')
  texts = load_txt(corpus_fp)

  ''' Model '''
  textgen = textgenrnn(name=args.corpus)

  cfg = {
    'rnn_layers':        args.rnn_layers,
    'rnn_size':          args.rnn_size,
    'rnn_bidirectional': args.rnn_bidirectional,
    'max_length':        args.max_length,
    'dim_embeddings':    args.dim_embeddings,
    'dropout':           args.dropout,
    'max_vocab':         args.max_vocab,
    'min_freq':          args.min_freq,
    'epochs':            args.epochs,
    'batch_size':        args.batch_size,
    'learning_rate':     args.learning_rate,
  }
  textgen.train(texts, cfg)


if __name__ == '__main__':
  __ = textgenrnn.default_config
  s2b = lambda x: True if x.lower in ['true', '1'] else False

  parser = ArgumentParser()
  parser.add_argument('--corpus',            default='corpus',                          help='corpus name under `util.CORPUS_PATH`')
  parser.add_argument('--rnn_layers',        default=__['rnn_layers'],        type=int, help='LSTM layers')
  parser.add_argument('--rnn_size',          default=__['rnn_size'],          type=int, help='LSTM hidden dim')
  parser.add_argument('--rnn_bidirectional', default=__['rnn_bidirectional'], type=s2b, help='probably useful for texts with certain paradim')
  parser.add_argument('--max_length',        default=__['max_length'],        type=int, help='N for the N-gram model')
  parser.add_argument('--dim_embeddings',    default=__['dim_embeddings'],    type=int, help='Embed dim')
  parser.add_argument('--dropout',           default=__['dropout'],           type=int, help='Dropout rate')
  parser.add_argument('--max_vocab',         default=__['max_vocab'],         type=int)
  parser.add_argument('--min_freq',          default=__['min_freq'],          type=int)
  parser.add_argument('--epochs',            default=__['epochs'],            type=int)
  parser.add_argument('--batch_size',        default=__['batch_size'],        type=int)
  parser.add_argument('--learning_rate',     default=__['learning_rate'],     type=float)
  args = parser.parse_args()
  
  train(args)