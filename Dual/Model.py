from LM.main import LM
import torch.nn as nn
from onmt.ModelConstructor import make_base_model
from argparse import ArgumentParser
from Utils.Opts import LM_opts
from TM.main import init_tm

class Dual(nn.Module):
    def __init__(self, opt):
        super(Dual, self).__init__()
        self.A = 'en'
        self.B = 'fr'
        self.LM_A = LM(opt, self.A)
        self.LM_B = LM(opt, self.B)
        self.TM_A = init_tm(self.A)
        self.TM_B = init_tm(self.B)



    def forward(self, *input):



def make_base_model():
    parser = ArgumentParser()
    LM_opts(parser)
    opt = parser.parse_args()
    model = Dual(opt)


if __name__ == "__main__":
    make_base_model()
