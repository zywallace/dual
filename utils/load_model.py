import torch
import onmt
from onmt.Utils import use_gpu


def load(opt):
    checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)
    fields = onmt.IO.load_fields(checkpoint['vocab'])
    model_opt = checkpoint['opt']

    model = onmt.ModelConstructor.make_base_model(
                                model_opt, fields, use_gpu(opt), checkpoint)
    return model, fields
