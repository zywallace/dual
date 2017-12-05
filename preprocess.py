from utils.preprocess_base import preprocess
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("-warmup", action='store_true', help='if warmup, preprocess warmup data for dual model')
    parser.add_argument("-src", required=True, choices=['en', 'fr'], help="source language")
    opt = parser.parse_args()
    prefix = "/dual_model/data/warmup/" if opt.warmup else "/baseline_model/data/"
    preprocess(prefix, opt.src)


if __name__ == "__main__":
    main()
