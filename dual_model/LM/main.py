from subprocess import check_output
from os import system


class LM:
    def __init__(self, opt, lang):
        if (opt.training_mode):
            cmd = "./faster-rnnlm/faster-rnnlm/rnnlm -rnnlm {} -train {} -valid {} -hidden {} -hidden-type {} -nce {} -alpha {}" \
                .format(opt.model, opt.train + lang, opt.valid + lang, opt.hidden, opt.hidden_type, opt.nce, opt.rl)
            system(cmd)
        self.model = opt.model

    def apply(self, input):
        # line should end with '\n'
        with open('tmp.txt', 'w') as f:
            for line in input:
                f.write(line)

        cmd = "./faster-rnnlm/faster-rnnlm/rnnlm -rnnlm {} -test tmp.txt".format(self.model)
        result = check_output(cmd, shell=True)
        return result
