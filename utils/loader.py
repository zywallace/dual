import torch
import onmt
from onmt.Utils import use_gpu

SRC_TMP = ""
TRG_TMP = ""

def load_model(opt):
    checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)
    fields = onmt.IO.load_fields(checkpoint['vocab'])
    model_opt = checkpoint['opt']

    model = onmt.ModelConstructor.make_base_model(
                                model_opt, fields, use_gpu(opt), checkpoint)
    return model, fields


def load_batch(fields, opt):
    data = onmt.IO.ONMTDataset(
        TRG_TMP, SRC_TMP, fields,
        use_filter_pred=False)

    data_iter = onmt.IO.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size * opt.n_best, shuffle=False, repeat=False)

    for batch in data_iter:
        return batch
