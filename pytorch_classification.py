
import torch
import torch.nn as nn


class GamblerLoss(nn.Module):
    def __init__(self, inference_will=1.1):
        super().__init__()
        """ Returns a CrossEntropyLoss-like object that has been modified
            according to the DeepGambler paper [1].
        Parameters
        ----------
        inference_will: floating point
            The willingness to infer or abstain.  Greater values make the
            inference more confident, while lower values make abstention more
            likely.  Given m different classes (not inclusive of the abstention
            class), meaningful values are in the range (1, m).
        Notes
        -----
        [1] DeepGamblers: Learning to Abstain with Portfolio Theory. L. Ziyin,
            Z. T. Wang, P. Pu Liang, R. Salakhutdinov, L.P. Morency, M. Ueda.
            NeurIPS, 2019
        """

        self.o = torch.tensor(inference_will)
        self.nllloss = nn.NLLLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, v, labels):
        """ Returns the Gambler loss value
        Parameters
        ----------
        v: a tensor of inputs with dimension BxI
        labels: a tensor with dimension Bx1 of indexes in [0, I-2] of the
                correct label (the I-1-th class is the abstinence one and
                should not be used in training).
        Notes
        -----
        This loss is a Negative Log Likelihood of weighted and scaled
        probabilities (from the Softmax response).  We refer to the log of
        these values as log_gambler_values.
        """

        x = self.log_gambler_values(v)
        return self.nllloss(x, labels)

    def log_gambler_values(self, v):
        """ Returns the log values to be passed to the Negative Log Likelihood
            Loss.
        Parameters
        ----------
        v: a tensor of inputs with dimension BxI
        Notes
        -----
        In order to avoid overflow issues due to exp of a large number, we
        execute it only on arguments lower or equal than 0. In order to avoid
        instability issues due to log of a small number, we execute it only on
        arguments greater or equal than 1.
        Our target values are in the form:
            xi = log (fi*o + fm)
        where fi = exp(vi)/sum(exp(vj)).
        To avoid computing exponential with argument greater than 0 we rescale
        the values wrt. the maximum per the first dimension, indicated as V.
        => fi = exp(vi-V)/sum(exp(vj-V)).
        xi = log[o*exp(vi-V)/sum(exp(vj-V)) + exp(vm-V)/sum(exp(vj-V))]
        vm is the value of the m+1-th class, the abstinence class.
        xi = log[(o*exp(vi-V) + exp(vm-V))/sum(exp(vj-V))]
           = log[o*exp(vi-V)(1 + exp(vm-V)/(exp(vi-V)*o)] - log(sum(exp(vj-V))
           = vi - V + log(o + exp(vm-V)/exp(vi-V)) - log(sum(exp(vj-V))
           = log(o + exp(vm-V)/exp(vi-V)) + log_softmax(v)
        The two exponential must be computed separately as there is no
        guarantee that vm-vi<=0. Being the exponentials greater than 0 and o
        >= 1, it follows the stability constraints are respected.
        """
        maxs = v.max(dim=1)[0]
        lasts = v[:,-1]

        upper = torch.exp(lasts-maxs)
        upside = torch.exp(v.transpose(0, 1).add(-maxs)).div(upper).transpose(0, 1)
        x = 1./upside + self.o
        return self.log_softmax(v) + torch.log(x)


def log_gambler(v, o):
    """
    dummy implementation for testing
    """
    softmax = nn.Softmax(dim=1)
    x = softmax(v)
    z = x[:, -1]
    y = torch.mul(x, o)
    x = y+z.unsqueeze(1)

    x = torch.log(x)  ## could be numerically unstable if probabilities approach zero
    return x


def output2class(model_output, coverage, abstain_class):
    probs = torch.softmax(model_output, dim=1)[:, abstain_class]
    _, predicted = torch.max(model_output[:, :abstain_class], dim=1)
    predicted = (probs < coverage).int().mul(predicted)
    predicted += (probs >= coverage).int().mul(abstain_class)
    return predicted


if __name__ == "__main__":
    v = torch.tensor([[1, 2, 3], [6, 4, 5]], dtype=float)
    o = 1.5
    gl = GamblerLoss(o)

    y = gl.log_gambler_values(v)
    z = log_gambler(v, o)

    print(f"Expected {z}")
    print(f"Computed {y}")