import torch
from itertools import zip_longest
import onmt
from torch.autograd import Variable
from utils.beam import Beam
from onmt.Utils import use_gpu

class TranslationModel:
    def __init__(self, model, fields, opt):


        self.opt = opt
        self.fields = fields
        self.model = model

    def train(self):
        self.model.train()
        self.model.generator.train()

    def eval(self):
        self.model.eval()
        self.model.generator.eval()

    def apply_lm(self, batch, data, direction):
        assert direction in ["in", "out"]
        self.eval()

        pred_batch, gold_batch, pred_scores, gold_scores, attn, src \
            = self.translate(batch, data)
        if direction == "in":
            return None, gold_scores

        z_batch = zip_longest(
            pred_batch, gold_batch,
            pred_scores, gold_scores,
            (sent.squeeze(1) for sent in src.split(1, dim=1)))
        pred = []
        score = []
        for pred_sents, gold_sent, pred_score, gold_score, src_sent in z_batch:
            n_best_preds = [" ".join(pred) for pred in pred_sents[:self.opt.n_best]]
            pred.append(n_best_preds)
            score.append(pred_score[:self.opt.n_best])
        self.train()
        return pred, score

    def build_trg_tok(self, pred, src, attn, copy_vocab):
        vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(copy_vocab.itos[tok - len(vocab)])
            if tokens[-1] == onmt.IO.EOS_WORD:
                tokens = tokens[:-1]
                break

        if self.opt.replace_unk and attn is not None:
            for i in range(len(tokens)):
                if tokens[i] == vocab.itos[onmt.IO.UNK]:
                    _, max_index = attn[i].max(0)
                    tokens[i] = self.fields["src"].vocab.itos[src[max_index[0]]]
        return tokens

    def _run_trg(self, batch, data):

        _, src_lengths = batch.src
        src = onmt.IO.make_features(batch, 'src')
        tgt_in = onmt.IO.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, context = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
            src, context, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.opt.cuda else torch
        gold_scores = Variable(tt.FloatTensor(batch.batch_size).fill_(0))
        dec_out, dec_states, attn = self.model.decoder(
            tgt_in, context, dec_states)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.IO.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:]):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores

    def translate_batch(self, batch, data_set):
        beam_size = self.opt.beam_size
        batch_size = batch.batch_size

        # (1) Run the encoder on the src.
        _, src_lengths = batch.src
        src = onmt.IO.make_features(batch, 'src')
        enc_states, context = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
            src, context, enc_states)

        #  (1b) Initialize for the decoder.
        def var(a):
            if isinstance(a, Variable):
                return a
            return Variable(a, volatile=True)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        # Repeat everything beam_size times.
        context = rvar(context)
        dec_states.repeat_beam_size_times(beam_size)

        beam = [Beam(beam_size, n_best=self.opt.n_best,
                     cuda=use_gpu(self.opt),
                     vocab=self.fields["tgt"].vocab,
                     global_scorer=None)
                for __ in range(batch_size)]

        # (2) run the decoder to generate sentences, using beam search.
        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        for i in range(self.opt.max_sent_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(1, -1))

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn = \
                self.model.decoder(inp, context, dec_states)
            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            out = self.model.generator.forward(dec_out)
            out = unbottle(out)
            # beam x tgt_vocab

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j], unbottle(attn["std"])[:, j])
                dec_states.beam_update(j, b.getCurrentOrigin().data, beam_size)

        if "tgt" in batch.__dict__:
            all_gold = self._run_trg(batch, data_set)
        else:
            all_gold = [0] * batch_size

        # (3) Package everything up.
        all_hyps, all_scores, all_attn = [], [], []
        for b in beam:
            n_best = self.opt.n_best
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            all_hyps.append(hyps)
            all_scores.append(scores)
            all_attn.append(attn)

        return all_hyps, all_scores, all_attn, all_gold

    def translate(self, batch, data):
        batch_size = batch.batch_size

        #  translate
        pred, pred_score, attn, gold_score = self.translate_batch(batch, data)
        assert (len(gold_score) == len(pred))
        pred, pred_score, attn, gold_score, i = list(zip(
            *sorted(zip(pred, pred_score, attn, gold_score,
                        batch.indices),
                    key=lambda x: x[-1])))
        inds, perm = torch.sort(batch.indices)

        # to words
        pred_batch, gold_batch = [], []
        src = batch.src[0].index_select(1, perm)
        if self.opt.tgt:
            tgt = batch.tgt.index_select(1, perm)
        for b in range(batch_size):
            src_vocab = data.src_vocabs[inds[b]]
            pred_batch.append(
                [self.build_trg_tok(pred[b][n], src[:, b],
                                    attn[b][n], src_vocab)
                 for n in range(self.opt.n_best)])
            if self.opt.tgt:
                gold_batch.append(
                    self.build_trg_tok(tgt[1:, b], src[:, b],
                                       None, None))
        return pred_batch, gold_batch, pred_score, gold_score, attn, src
