#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/18 

from argparse import ArgumentParser
from textgenrnn import textgenrnn


def train(args):
  textgen = textgenrnn(name=args.corpus)
  if args.ckpt_fn: textgen.load(args.ckpt_fn)

  if args.interactive:
    text = textgen.generate(interactive=True, top_n=7, max_gen_length=args.max_gen_length)
    print(text)
  else:
    for _ in range(args.n):
      text = textgen.generate(max_gen_length=args.max_gen_length)
      print(text)
      print()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--corpus', default='corpus', help='experiment name under `util.LOG_PATH`')
  parser.add_argument('--ckpt_fn', default=None, help='path to model weight file')
  parser.add_argument('--n', default=3, type=int, help='number of generated sentences')
  parser.add_argument('-i', '--interactive', action='store_true', help='interactive mode')
  parser.add_argument('-L', '--max_gen_length', default=140, type=int, help='max generated length')
  args = parser.parse_args()
  
  train(args)