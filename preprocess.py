from utils.preprocess_base import preprocess
from argparse import ArgumentParser
from os.path import isfile
from random import sample
from utils.config import *
from os import system


def warm_up(data, p, start):
    with open(data + FILE_NAMES[start] + SUFFIX, 'r') as f:
        train_src = list(f)
    with open(data + FILE_NAMES[start + 1] + SUFFIX, 'r') as f:
        train_trg = list(f)
    assert len(train_src) == len(train_trg), "train_src and train_trg have different length"
    size = len(train_src)
    ind = sample(range(size), int(size * p))
    train_src = [train_src[i] for i in ind]
    train_trg = [train_trg[i] for i in ind]
    with open(data + "warmup/" + FILE_NAMES[start] + SUFFIX, 'w') as f:
        f.writelines(train_src)
    with open(data + "warmup/" + FILE_NAMES[start + 1] + SUFFIX, 'w') as f:
        f.writelines(train_trg)


def check_data(data):
    for file in FILE_NAMES:
        if not isfile(data + file + SUFFIX):
            return False
    return True


def main():
    parser = ArgumentParser()
    parser.add_argument("-data", required=True, help='(raw) data files location')
    parser.add_argument("-tok", action="store_true", help="prepare data: run moses scrips on raw data")
    parser.add_argument("-warmup", action="store_true", help="generate small portion of data for warm up")
    parser.add_argument("-p", default=.1, help='portion of data used for warmup')
    opt = parser.parse_args()
    if opt.tok or not check_data(opt.data):
        cmd = "for l in en fr; do for f in {}*.$l; do perl scripts/tokenizer.perl -threads 12 -a -no-escape -l $l -q  < $f > $f{}; done; done".format(
            opt.data, SUFFIX)
        system(cmd)
    print("Prepare data done!")

    # for baseline model
    preprocess(opt.data, BASELINE, 'en')
    preprocess(opt.data, BASELINE, 'fr')

    if opt.warmup or not check_data(opt.data + "warmup/"):
        # only shuffle training set
        warm_up(opt.data, opt.p, 0)
        warm_up(opt.data, 1, 2)
    # for project model warm up
    preprocess(opt.data + "warmup/", WARM_UP, 'en')
    preprocess(opt.data + "warmup/", WARM_UP, 'fr')


if __name__ == "__main__":
    main()
