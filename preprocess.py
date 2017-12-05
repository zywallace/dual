from utils.preprocess_base import preprocess
from argparse import ArgumentParser
from os.path import isfile
from random import sample

WARM_UP = "dual_model/data/warmup/"
BASELINE = "baseline_model/data/"


def shuffle(p):
    # use entire dev set so we only shuffle training set
    prefix = BASELINE + "train"
    with open(prefix + '.en', 'r') as f:
        train_src = list(f)
    with open(prefix + '.fr', 'r') as f:
        train_trg = list(f)
    assert len(train_src) == len(train_trg), "train_src and train_trg have different length"
    size = len(train_src)
    ind = sample(range(1, size + 1), int(size * p))
    train_src = [train_src[i] for i in ind]
    train_trg = [train_trg[i] for i in ind]
    prefix = WARM_UP + "train"
    with open(prefix + '.en', 'w') as f:
        f.writelines(train_src)
    with open(prefix + '.fr', 'w') as f:
        f.writelines(train_trg)


def check_warm_up(lang):
    prefix = WARM_UP + lang + '/'
    vocab = prefix + "vocab"
    train = prefix + "train"
    valid = prefix + "valid"
    return isfile(vocab) and isfile(train) and isfile(valid)


def main():
    parser = ArgumentParser()
    parser.add_argument("-warmup", action='store_true', help='if warmup, preprocess warmup data for dual model')
    parser.add_argument("-shuffle", action='store_true',
                        help='if shuffle, generate small portion[p] of data from all to be used for warmup')
    parser.add_argument("-p", default=.1, help='portion of data used for warmup')
    parser.add_argument("-src", required=True, choices=['en', 'fr'], help="source language")
    opt = parser.parse_args()
    if opt.warmup:
        if not check_warm_up(opt.src) or opt.shuffle:
            shuffle(opt.p)
        preprocess(WARM_UP, opt.src)
    else:
        preprocess(BASELINE, opt.src)


if __name__ == "__main__":
    main()
