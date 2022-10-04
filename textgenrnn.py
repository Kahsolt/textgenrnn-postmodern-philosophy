import os
from traceback import print_exc
from typing import Counter

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

from model import *
from util import *

tf_cfg = tf.compat.v1.ConfigProto()
tf_cfg.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=tf_cfg))


class textgenrnn:

    META_TOKEN = '<s>'
    default_config = {
        'name': None,
        'rnn_layers': 2,
        'rnn_size': 128,
        'rnn_bidirectional': False,
        'max_length': 40,
        'dim_embeddings': 100,
        'dropout': 0.0,
        'max_vocab': 10000,
        'min_freq': 3,
        'epochs': 100,
        'batch_size': 128,
        'learning_rate': 4e-3,
    }

    def __init__(self, name):
        self.name = name
        self.default_config.update({'name': name})
        
        self.workspace_dp = os.path.join(LOG_PATH, name)
        if os.path.exists(self.workspace_dp):
            try:
                self.load()
                print(f"[textgenrnn] open existing workspace {name!r}")
            except:
                print(f"[textgenrnn] create new workspace {name!r}")
        else:
            print(f"[textgenrnn] create new workspace {name!r}")
            os.makedirs(self.workspace_dp, exist_ok=True)   # include parents
    
    def save(self, suffix=''):
        try:
            save_json(self.config, os.path.join(self.workspace_dp, 'config.json'))
            save_txt(self.vocab, os.path.join(self.workspace_dp, 'vocab.txt'))
            self.model.save_weights(os.path.join(self.workspace_dp, f'weights{suffix}.hdf5'))
        except Exception as e:
            print_exc()
        
    def load(self, ckpt_fn='weights.hdf5'):
        try:
            self.config = self.default_config.copy()
            self.config.update(load_json(os.path.join(self.workspace_dp, 'config.json')))

            self.vocab = load_txt(os.path.join(self.workspace_dp, 'vocab.txt'))
            self.idx_to_word = {i: w for i, w in enumerate(self.vocab, start=1)}
            self.word_to_idx = {w: i for i, w in enumerate(self.vocab, start=1)}

            self.optimizer = Adam(learning_rate=self.config['learning_rate'])
            self.model = textgenrnn_model(num_classes=len(self.vocab) + 1, config=self.config, 
                                          optimizer=self.optimizer,
                                          weights_path=os.path.join(self.workspace_dp, ckpt_fn))
        except Exception as e:
            print_exc()
    
    def train(self, texts:List[str], config:dict):
        if not hasattr(self, 'model'):
            print('>> fresh training on a new model')

            self.config = self.default_config.copy()
            if config: self.config.update(config)

            words = [word for sent in texts for word in sent.split(' ')]
            cnt_word = [(cnt, word) for word, cnt in Counter(words).items() if cnt > self.config['min_freq']]
            cnt_word.sort(reverse=True)
            vocab = [w for c, w in cnt_word[:self.config['max_vocab']]]         # should it be unique
            self.vocab = sorted(vocab)
            self.idx_to_word = {i: w for i, w in enumerate(self.vocab, start=1)}
            self.word_to_idx = {w: i for i, w in enumerate(self.vocab, start=1)}

            self.optimizer = Adam(learning_rate=self.config['learning_rate'])
            self.model = textgenrnn_model(num_classes=len(self.vocab) + 1, config=self.config, 
                                          optimizer=self.optimizer)

        else:
            self.config.update(config)
            print('>> retrain model on a pretrained weights, configs about model definition will be ignored')

        self.do_train(texts, **self.config)
        self.save()

    def do_train(self, texts, epochs=50, batch_size=128, learning_rate=4e-3, split_ratio=0.95, max_gen_length=300, **kwargs):
        self.model.summary()

        # list of list of words
        texts = [text.split(' ') for text in texts]

        # calculate all combinations of text indices + token indices, (sent_id, end_idx)
        index_list = [np.meshgrid(np.array(i), np.arange(len(text) + 1)) for i, text in enumerate(texts)]
        # indices_list = np.block(indices_list)   # this hangs when indices_list is large enough
        # FIX BEGIN ------
        tmp = np.block(index_list[0])
        for i in range(len(index_list)-1):
            tmp = np.concatenate([tmp, np.block(index_list[i+1])])
        index_list = tmp

        # FIX END ------
        n_examples = len(index_list)          # [N, 2]
        mask = np.random.rand(n_examples) < split_ratio
        index_list_train = index_list[ mask, :]
        index_list_valid = index_list[~mask, :]
        n_examples_train = len(index_list_train)
        n_examples_valid = len(index_list_valid)
        train_loader = make_dataloader(texts, index_list_train, self)
        valid_loader = make_dataloader(texts, index_list_valid, self)
        train_steps = max(n_examples // batch_size, 1)
        valid_steps = max(n_examples // batch_size, 1)

        assert n_examples >= batch_size, "Fewer tokens than batch_size."
        print(f"Training on {n_examples_train:,} word sequences, validating on {n_examples_valid:,} examples")

        # scheduler function must be defined inline.
        def learning_rate_linear_decay(epoch, start_decay=10):
            return learning_rate if epoch < start_decay else (learning_rate * (1 - ((epoch - start_decay) / epochs)))

        self.model.fit(train_loader, 
                       steps_per_epoch=train_steps,
                       epochs=epochs,
                       callbacks=[
                           LearningRateScheduler(learning_rate_linear_decay),
                           GenerateAfterEpoch(self, max_gen_length),
                           SaveModelWeights(self)],
                       verbose=True,
                       max_queue_size=10,
                       validation_data=valid_loader,
                       validation_steps=valid_steps)

    def generate(self, temperature=[1.0, 0.5, 0.2, 0.2], max_gen_length=300, interactive=False, top_n=3) -> str:
        
        def textgenrnn_sample(preds, temperature:List[float], interactive=False, top_n=3):
            '''
            Samples predicted probabilities of the next character to allow
            for the network to show "creativity."
            '''

            preds = np.asarray(preds).astype('float64')

            if temperature in [None, 0.0]:
                return np.argmax(preds)

            preds = np.log(preds + K.epsilon()) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)

            # prevent from being able to choose 0 (placeholder)
            preds[0] = 0

            if not interactive:
                probas = np.random.multinomial(1, preds, 1)
                index = np.argmax(probas)
            else:
                # return list of top N chars/words descending order, based on probability
                index = (-preds).argsort()[:top_n]

            return index

        model = self.model
        idx_to_word = self.idx_to_word
        word_idx_to = self.word_to_idx
        maxlen = self.config['max_length']
        meta_token = self.META_TOKEN

        is_eos = False
        tokens = [meta_token]

        while not is_eos and len(tokens) < max_gen_length:
            encoded_text = tokens_to_index(tokens[-maxlen:], word_idx_to, maxlen)[None, :]  # [B=1, T=40]
            next_temperature = temperature[(len(tokens) - 1) % len(temperature)]
            preds = model.predict(encoded_text, batch_size=1, verbose=False)[0]

            if not interactive:     # auto-generate text without user intervention
                next_index = textgenrnn_sample(preds, next_temperature)
                next_char = idx_to_word[next_index]

                tokens.append(next_char)
                if next_char == meta_token:
                    is_eos = True
            
            else:                   # ask user what the next char/word should be
                options_index = textgenrnn_sample(preds, next_temperature, interactive=True, top_n=top_n)
                options = [idx_to_word[idx] for idx in options_index]

                print('[控制]  s: 结束   x: 删除')
                print('[选项]')
                for i, option in enumerate(options, 1):
                    print(f'   {i}: {option}')
                print(f'[当前句子]: {"".join(tokens[1:])}\n')

                while True:
                    user_input = input('你的选择 或 输入新词 > ').strip()

                    if str.isdigit(user_input):
                        try:
                            next_char = options[int(user_input) - 1]
                            tokens.append(next_char)
                            break
                        except IndexError as e:
                            print(e)
                    else:
                        if user_input == 's':
                            is_eos = True
                        elif user_input == 'x':
                            if len(tokens) > 1: del tokens[-1]
                        else:
                            tokens.append(user_input)
                        break

        # remove the <s> meta_tokens
        return ''.join(tokens[1:-1])


class GenerateAfterEpoch(Callback):
    def __init__(self, textgenrnn:textgenrnn, max_gen_length:int):
        super().__init__()

        self.textgenrnn = textgenrnn
        self.max_gen_length = max_gen_length

    def on_epoch_end(self, epoch, logs=None):
        for temperature in [0.2, 0.5, 1.0]:
            print('#'*20 + '\nTemperature: {}\n'.format(temperature) + '#'*20)
            for _ in range(3):
                print(self.textgenrnn.generate(temperature=temperature))


class SaveModelWeights(Callback):
    def __init__(self, textgenrnn: textgenrnn):
        super().__init__()

        self.textgenrnn = textgenrnn
        self.weights_name = textgenrnn.config['name']

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            self.textgenrnn.save(suffix=f'_{epoch + 1}')


def tokens_to_index(text:List[str], word_to_idx:Dict[str, int], maxlen:int) -> np.array:
    seq = np.array([word_to_idx.get(x, 0) for x in text])
    return pad_sequences([seq], maxlen=maxlen).squeeze()      # left zero pad


def token_to_onehot(char:str, word_to_idx:Dict[str, int]) -> np.array:
    oh = np.float32(np.zeros(len(word_to_idx) + 1))
    oh[word_to_idx.get(char, 0)] = 1
    return oh


def make_dataloader(texts:List[list], idx_list:np.ndarray, textgenrnn:textgenrnn):
    max_length = textgenrnn.config['max_length']
    batch_size = textgenrnn.config['batch_size']
    meta_token = textgenrnn.META_TOKEN
    word_to_idx = textgenrnn.word_to_idx

    while True:
        X, Y = [], []
        cnt = 0

        np.random.shuffle(idx_list)     # [392458, 2]
        for i in range(len(idx_list)):
            sent_idx, end_idx = idx_list[i, 0], idx_list[i, 1]
            text = [meta_token] + list(texts[sent_idx]) + [meta_token]

            # n-gram model: `x` is a charseq, `y` is a char
            start_idx = max(0, end_idx - max_length)
            x = text[start_idx: end_idx + 1]
            y = text[end_idx + 1]

            if y in word_to_idx:
                x = tokens_to_index(x, word_to_idx, max_length)     # [T=max_len]
                y = token_to_onehot(y, word_to_idx)                 # [K=vocab_size]
                X.append(x)
                Y.append(y)

                cnt += 1
                if cnt % batch_size == 0:
                    X = np.asarray(X, dtype=np.int32)   # [B, T]
                    Y = np.asarray(Y, dtype=np.int32)   # [B, K]
                    yield (X, Y)

                    X, Y = [], []
                    cnt = 0
