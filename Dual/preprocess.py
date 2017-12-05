# -*- coding: utf-8 -*-
import argparse
import codecs
import torch
import onmt
import onmt.IO
import Utils.Opts as opts


def preprocess(opt):
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
    print("Saving train/valid/fields")

    # Can't save fields, so remove/reconstruct at training time.
    torch.save(onmt.IO.save_vocab(fields),
               open(opt.save_data + '.vocab.' + opt.src, 'wb'))
    train.fields = []
    valid.fields = []
    torch.save(train, open(opt.save_data + '.train.' + opt.src, 'wb'))
    torch.save(valid, open(opt.save_data + '.valid.' + opt.src, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.preprocess_opts(parser)
    opts.add_md_help_argument(parser)
    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    preprocess(opt)

