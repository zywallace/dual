from subprocess import check_output
from os import system
from math import log
import torch

class LM:
    def __init__(self, opt, lang):
        if opt.training_mode:
            cmd = "./faster-rnnlm/faster-rnnlm/rnnlm -rnnlm {} -train {} -valid {} -hidden {} -hidden-type {} -nce {} -alpha {}" \
                .format(opt.model, opt.train + lang, opt.valid + lang, opt.hidden, opt.hidden_type, opt.nce, opt.rl)
            system(cmd)
        self.model = opt.model

    def apply(self, batch):
        # line should end with '\n'
        with open('tmp.txt', 'w') as f:
            for line in batch:
                f.write(line)

        cmd = "./faster-rnnlm/faster-rnnlm/rnnlm -rnnlm {} -test tmp.txt".format(self.model)
        result = check_output(cmd, shell=True)
        score = []
        for i in result:
            if i == "OOV":
                print("here is an OOV")
                score.append(float('-inf'))
            else:
                score.append(float(i) * log(10))
        return torch.stack(score)

    def score(self, pred):
        """
        :param pred: batch * k predicted sentences - type: list of list
        :return: score: batch * k, pred's log prob, natural base
        """
        num = len(pred)
        flat = [sent for n_best in pred for sent in n_best]
        score = self.apply(flat)
        return score.view(num, -1)

