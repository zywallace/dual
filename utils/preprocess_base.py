# -*- coding: utf-8 -*-
import argparse
import codecs
import torch
import onmt
import onmt.IO
import utils.opts as opts
import sys
from utils.config import *


def preprocess(data, des, lang):
    train_src = data + "train.en" + SUFFIX
    train_trg = data + "train.fr" + SUFFIX
    val_src = data + "valid.en" + SUFFIX
    val_trg = data + "valid.fr" + SUFFIX

    if lang == 'fr':
        train_src, train_trg = train_trg, train_src
        val_src, val_trg = val_trg, val_src
    sys.argv = ['', '-train_src', train_src, '-train_tgt', train_trg, '-valid_src', val_src, '-valid_tgt', val_trg,
                '-save_data', des + lang + '/', '-lower']

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.preprocess_opts(parser)
    opts.add_md_help_argument(parser)
    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    print("Source language is {}".format("English" if lang == 'en' else "French"))
    print('Preparing training ...')
    with codecs.open(opt.train_src, "r", "utf-8") as src_file:
        src_line = src_file.readline().strip().split()
        _, _, n_src_features = onmt.IO.extract_features(src_line)
    with codecs.open(opt.train_tgt, "r", "utf-8") as tgt_file:
        tgt_line = tgt_file.readline().strip().split()
        _, _, n_tgt_features = onmt.IO.extract_features(tgt_line)

    fields = onmt.IO.get_fields(n_src_features, n_tgt_features)
    print("Building Training...")
    train = onmt.IO.ONMTDataset(
        opt.train_src, opt.train_tgt, fields,
        opt.src_seq_length, opt.tgt_seq_length,
        src_seq_length_trunc=opt.src_seq_length_trunc,
        tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict)
    print("Building Vocab...")
    onmt.IO.build_vocab(train, opt)

    print("Building Valid...")
    valid = onmt.IO.ONMTDataset(
        opt.valid_src, opt.valid_tgt, fields,
        opt.src_seq_length, opt.tgt_seq_length,
        src_seq_length_trunc=opt.src_seq_length_trunc,
        tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict)
    print("Saving train/valid/fields to {}".format(opt.save_data))

    # Can't save fields, so remove/reconstruct at training time.
    torch.save(onmt.IO.save_vocab(fields),
               open(opt.save_data + 'vocab', 'wb'))
    print(' * vocabulary size. source = %d; target = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    train.fields = []
    valid.fields = []
    torch.save(train, open(opt.save_data + 'train', 'wb'))
    print(' * number of training sentences: %d' % len(train))
    torch.save(valid, open(opt.save_data + 'valid', 'wb'))
    print(' * number of validation sentences: %d' % len(valid))
    print("Done!")
