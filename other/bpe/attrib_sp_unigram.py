"""
attrib_sp_unigram.py

Learn a Unigram encoding (see https://github.com/google/sentencepiece)
and Kudo (https://arxiv.org/abs/1804.10959)

This file uses the pytorch training environment (see README).
"""
import sentencepiece as sp
import sys
import random
import os
sys.path.append("../common/")
from common import data_io_utils

datasets_dir = '../../../data'

sp.SentencePieceTrainer.Train(f'--num_threads=64 --add_dummy_prefix=false --normalization_rule_name=identity --input={datasets_dir}/bpe/train_seqs_4bpe.txt --model_prefix=attrib_sp_100 --vocab_size=100 --max_sentence_length=20000 --character_coverage=1.0 --eos_id=-1 --bos_id=-1 --hard_vocab_limit=false')

print(f"Finished the first encoding")

sp.SentencePieceTrainer.Train(f'--num_threads=64 --add_dummy_prefix=false --normalization_rule_name=identity --input={datasets_dir}/bpe/train_seqs_4bpe.txt --model_prefix=attrib_sp_1000 --vocab_size=1000 --max_sentence_length=20000 --character_coverage=1.0 --eos_id=-1 --bos_id=-1 --hard_vocab_limit=false')

print(f"Finished the second encoding")
sp.SentencePieceTrainer.Train(f'--num_threads=64 --add_dummy_prefix=false --normalization_rule_name=identity --input={datasets_dir}/bpe/train_seqs_4bpe.txt --model_prefix=attrib_sp_5000 --vocab_size=5000 --max_sentence_length=20000 --character_coverage=1.0 --eos_id=-1 --bos_id=-1 --hard_vocab_limit=false')

print(f"Finished the third encoding")
sp.SentencePieceTrainer.Train(f'--num_threads=64 --add_dummy_prefix=false --normalization_rule_name=identity --input={datasets_dir}/bpe/train_seqs_4bpe.txt --model_prefix=attrib_sp_10000 --vocab_size=10000 --max_sentence_length=20000 --character_coverage=1.0 --eos_id=-1 --bos_id=-1 --hard_vocab_limit=false')
print(f"Finished the last encoding")
