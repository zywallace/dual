from onmt.Optim import Optim


class Optim:
    def __init__(self, model_a, model_b, opt=None):
        assert opt is None, "not support yet"

        self.optim_a = Optim('sgd', 0.0002)
        self.optim_a.set_parameters(model_a.parameters())

        self.optim_b = Optim('sgd', 0.02)
        self.optim_b.set_parameters(model_b.parameters())

    def step(self):
        self.optim_a.step()
        self.optim_b.step()


class Trainer:
    def __init__(self, train_data, model_a, model_b, lm_a, lm_b, optim_a, optim_b, alpha=0.005):
        """
        :param train_data: list of dict of monolingual data, train_data[0]["a"] is batch 0 of sentences in language A
        :param model_a: translation model to language A
        :param model_b: translation model to language B
        :param lm_a: language model of language of A
        :param lm_b: language model of language of B
        :param optim_a: optimizers for a - b - a round
        :param optim_b: optimizers for b - a - b round
        """
        self.train_data = train_data
        self.model_a = model_a
        self.model_b = model_b

        self.lm_a = lm_a
        self.lm_b = lm_b

        self.alpha = alpha
        self.optim_a = optim_a
        self.optim_b = optim_b

        # training round
        self.start = ("a", "b")

    def train(self):
        for s_a, s_b in self.train_data:
            # init grad at first!
            # a -> b -> a
            self.zero_grad()
            self._round(s_a, 'a')
            self.optim_a.step()

            # b -> a -> b
            self.zero_grad()
            self._round(s_b, 'b')
            self.optim_b.step()

    def zero_grad(self):
        self.model_a.zero_grad()
        self.model_b.zero_grad()

    def _round(self, s, start):
        """
        for detail, look at the algorithm in report
        :param s: one batch of sentences in start language
        :param start: indicates what is the start of this round
        :return: None
        """
        assert start in ['a', 'b']

        model_a, model_b, lm = self.model_a, self.model_b, self.lm_b
        if start == "b":
            model_a, model_b = model_b, model_a
            lm = self.lm_a

        pred, pred_score, _ = model_a.predict(s, None)
        r_1 = lm.score(pred)  # batch * k r_1 is not Variable
        _, _, r_2 = model_b.predict(pred, s)
        r = self.alpha * r_1 + (1 - self.alpha) * r_2.data

        # batch * k
        grad_a = (pred_score * r).mean()
        grad_b = (r_2 * (1 - self.alpha)).mean()
        grad_a.backward()
        grad_b.backward()
