#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/18 

import os
import json
from re import compile as Regex
from typing import List
from argparse import ArgumentParser

import docx
import jieba
import jieba.posseg as pseg
jieba.initialize()
try: jieba.enable_paddle()
except: print('>> [warn] so far paddlepaddle-tiny only supports up to python3.7')

from util import CORPUS_PATH, CORPORA_PATH, load_user_vocab, user_vocab_fp

REGEX_BLANK = Regex('\s+')
REGEX_BRACKETS = Regex('(\(.*\))')


def _process_file(fp:str, filter:str='') -> List[str]:
  _, ext = os.path.splitext(fp)
  if   ext == '.txt':  return _process_txt(fp, filter)
  elif ext == '.json': return _process_json(fp, filter)
  elif ext == '.docx': return _process_docx(fp, filter)
  else: raise ValueError(f'unsupported file type {ext}')

def _process_txt(fp:str, filter:str=''):
  with open(fp, 'r', encoding='utf-8') as fh:
    lines = fh.read().strip().split('\n')

  if filter:
    fulltext = ''.join(lines)
    if filter not in fulltext:
      return []

  return [_clean_text(sent) for sent in lines]

def _process_json(fp:str, filter:str=''):
  with open(fp, 'r', encoding='utf-8') as fh:
    articles = json.load(fh)

  return [_clean_text(x['title'] + x['content']) for x in articles 
          if filter in x['title'] or filter in x['author'] or filter in x['organization'] or filter in x['content']]

def _process_docx(fp:str, filter:str=''):
  doc = docx.Document(fp)
  
  sents = [ ]
  for p in doc.paragraphs:
    p = p.text.strip()
    p = p.replace('.', '，')        # 扫描错误
    p = REGEX_BRACKETS.sub('', p)   # 忽略注释
    if not p: continue

    sents.append(_clean_text(p))

  return sents

def _clean_text(text:str) -> str:
  # remove blank chars
  text = text.replace('\u200b', '')
  text = REGEX_BLANK.sub(' ', text)
  text = text.strip()

  # tokenize
  words = jieba.lcut(text, HMM=False)
  words = [w.strip() for w in words if w.strip()]

  # concatenate
  text = ' '.join(words)
  return text


def preprocess(args):
  # load user_dict.txt for tokenizer
  jieba.load_userdict(user_vocab_fp)
  user_vocab = load_user_vocab()
  for word in user_vocab:
    jieba.suggest_freq(word, tune=True)
  print(f'[UserDict] found {len(user_vocab)} user-defined words')

  # gather copora files
  if args.input == '*':
    in_fps = [os.path.join(CORPORA_PATH, fn) for fn in os.listdir(CORPORA_PATH) if fn != 'README.md']
  else:
    in_fps = [os.path.join(CORPORA_PATH, fn) for fn in args.input.split(',') if fn != 'README.md']
  print(f'[Corpora] gathered {len(in_fps)} documents')

  # cleanify to sentences
  sentences = []    # List[str]
  for fp in in_fps:
    print(f'  >> processing {fp}')
    sentences.extend(_process_file(fp, args.filter))

  # write corpus.txt
  corpus_fp = os.path.join(CORPUS_PATH, args.output)
  with open(corpus_fp, 'w', encoding='utf-8') as fh:
    for sent in sentences:
      fh.write(sent)
      fh.write('\n')
  print(f'[Corpus] wrote {len(sentences)} sentences')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--input', default='*', help='comma seperated filenames relative to `util.CORPORA_PATH`; `*` for all files')
  parser.add_argument('--output', default='corpus.txt', help='output filename relative to `util.CORPUS_PATH`')
  parser.add_argument('--filter', default='', help='filter for document level')
  args = parser.parse_args()
  
  preprocess(args)