#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/18 

import os
import json
from re import compile as Regex
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Union

CORPUS_PATH         = 'corpus'
CORPORA_PATH        = 'corpora'
LOG_PATH            = 'log'
user_vocab_fp       = os.path.join(CORPUS_PATH,  'user_vocab.txt')
crawler_database_fp = os.path.join(CORPORA_PATH, 'crawler_db.json')

INVALID_PATH_CHAR_REGEX = Regex('[\\\/\:\*\?\"\<\>\|]')


@dataclass
class Article:

  url: str = ''
  title: str = ''
  author: str = ''
  organization: str = ''
  content: str = ''


def load_json(fp:str, default={}) -> Dict:
  ret = default
  if os.path.exists(fp):
    with open(fp, 'r', encoding='utf-8') as fh:
      ret = json.load(fh)
  return ret


def save_json(data:Union[List, Dict], fp:str):
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)


def load_txt(fp:str, default=[]) -> List:
  ret = default
  if os.path.exists(fp):
    with open(fp, 'r', encoding='utf-8') as fh:
      ret = [line.strip() for line in fh.readlines() if line.strip()]
  return ret


def save_txt(data:List, fp:str):
  with open(fp, 'w', encoding='utf-8') as fh:
    for line in data:
      if line.strip():
        fh.write(line.strip())
        fh.write('\n')


def load_user_vocab() -> Set[str]:
  return set(load_txt(user_vocab_fp, default=[]))


def load_crawler_database() -> List[dict]:
  return load_json(crawler_database_fp, default=[])


def save_crawler_database(data: List[dict]):
  save_json(data, crawler_database_fp)


def safe_filename(fn: str) -> str:
  return INVALID_PATH_CHAR_REGEX.sub('-', fn)


def parse_url(url:str) -> Tuple[str, Dict[str, str]]:
  url = url.strip()
  if '#' in url:
    url = url[:url.index('#')]
  
  api, params = url, {}
  if '?' in url:
    api, param_str = url.split('?')
    if param_str:
      for kv in param_str.split('&'):
        i = kv.index('=')         # value中可能出现'='，不能直接 `kv.split('=')`
        k, v = kv[:i], kv[i+1:]
        params[k] = v
  
  return api, params


def make_url(api: str, params: Dict[str, str]) -> str:
  if not params: return api

  param_str = '&'.join([f'{k}={v}' for k, v in params.items()])
  return f'{api}?{param_str}'


def cleanify_url(url: str, filter_keys:List[str]) -> str:
  api, params = parse_url(url)
  filtered_params = dict(filter(lambda kv: kv[0] in filter_keys, params.items()))
  return make_url(api, filtered_params)
