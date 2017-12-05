import sys
from utils.preprocess_base import preprocess as main

if __name__ == "__main__":
    train_src = "data/warmup/train.en"
    train_trg = "data/warmup/train.fr"
    val_src = "data/warmup/val.en"
    val_trg = "data/warmup/val.fr"

    if sys.argv[1] == 'fr':
        train_src, train_trg = train_trg, train_src
        val_src, val_trg = val_trg, val_src
    sys.argv = ['-train_src', train_src, '-train_tgt', train_trg, '-valid_src', val_src, '-valid_tgt', val_trg, '-src',
                sys.argv[1]]
    main()
