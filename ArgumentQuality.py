import pandas as pd
from mynn.layers.dense import dense
from mynn.initializers.glorot_normal import glorot_normal
from mynn.optimizers.adam import Adam
from mygrad.nnet.losses import softmax_crossentropy
import mygrad as mg
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

ghost = KeyedVectors.load_word2vec_format("ghost.6B.50d.txt.w2v", binary=False)


train = pd.read_csv("train.csv")

stance1 = np.array(train["evidence_1_stance"])
stance2 = np.array(train["evidence_2_stance"])
evidence_1 = [
    train["evidence_1"][i] for i in range(len(stance1)) if stance1[i] == stance2[i]
]
evidence_2 = [
    train["evidence_2"][i] for i in range(len(stance1)) if stance1[i] == stance2[i]
]
y_train = [
    train["evidence_1_detection_score"][i]
    for i in range(len(stance1))
    if stance1[i] == stance2[i]
]
y_test = [
    train["evidence_2_detection_score"][i]
    for i in range(len(stance1))
    if stance1[i] == stance2[i]
]
np.equal(stance1, stance2)


class RNN:
    def __init__(self, dim_input, dim_recurrent, dim_output):
        """ Initializes all layers needed for RNN

        Parameters
        ----------
        dim_input: int
            Dimensionality of data passed to RNN (C)

        dim_recurrent: int
            Dimensionality of hidden state in RNN (D)

        dim_output: int
            Dimensionality of output of RNN (K)
        """

        self.fc_x2h = dense(dim_input, dim_recurrent, weight_initializer=glorot_normal)
        self.fc_h2h = dense(
            dim_recurrent, dim_recurrent, weight_initializer=glorot_normal, bias=False
        )
        self.fc_h2y = dense(dim_recurrent, dim_output, weight_initializer=glorot_normal)
        self.Uz = mg.Tensor(
            np.random.randn(dim_input * dim_recurrent).reshape(dim_input, dim_recurrent)
        )
        self.Wz = mg.Tensor(
            np.random.randn(dim_recurrent * dim_recurrent).reshape(
                dim_recurrent, dim_recurrent
            )
        )
        self.bz = mg.Tensor(np.random.randn(dim_recurrent))
        self.Ur = mg.Tensor(
            np.random.randn(dim_input * dim_recurrent).reshape(dim_input, dim_recurrent)
        )
        self.Wr = mg.Tensor(
            np.random.randn(dim_recurrent * dim_recurrent).reshape(
                dim_recurrent, dim_recurrent
            )
        )
        self.br = mg.Tensor(np.random.randn(dim_recurrent))
        self.Uh = mg.Tensor(
            np.random.randn(dim_input * dim_recurrent).reshape(dim_input, dim_recurrent)
        )
        self.Wh = mg.Tensor(
            np.random.randn(dim_recurrent * dim_recurrent).reshape(
                dim_recurrent, dim_recurrent
            )
        )
        self.bh = mg.Tensor(np.random.randn(dim_recurrent))

    def __call__(self, x):
        """ Performs the full forward pass for the RNN.

        Note that we only care about the last y - the final classification scores for the full sequence

        Parameters
        ----------
        x: Union[numpy.ndarray, mygrad.Tensor], shape=(T, C)
            The one-hot encodings for the sequence

        Returns
        -------
        mygrad.Tensor, shape=(1, K)
            The final classification of the sequence
        """

        h = mg.nnet.gru(
            x,
            self.Uz,
            self.Wz,
            self.bz,
            self.Ur,
            self.Wr,
            self.br,
            self.Uh,
            self.Wh,
            self.bh,
        )
        return self.fc_h2y(h[-1])

    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.

        This can be accessed as an attribute, via `model.parameters`

        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model
        """
        return self.fc_x2h.parameters + self.fc_h2h.parameters + self.fc_h2y.parameters


from collections import Counter

counters = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

panda = pd.read_csv("train.csv")
import re

stance1 = np.array(panda["evidence_1_stance"])
stance2 = np.array(panda["evidence_2_stance"])
evidence_1 = [
    panda["evidence_1"][i] for i in range(len(stance1)) if stance1[i] == stance2[i]
]
evidence_2 = [
    panda["evidence_2"][i] for i in range(len(stance1)) if stance1[i] == stance2[i]
]
y_train1 = [
    panda["evidence_1_detection_score"][i]
    for i in range(len(stance1))
    if stance1[i] == stance2[i]
]

y_train2 = [
    panda["evidence_2_detection_score"][i]
    for i in range(len(stance1))
    if stance1[i] == stance2[i]
]

x_train1 = []
indices_1 = []
for index, i in enumerate(evidence_1):

    i = i.lower().replace("[ref]", "")
    i = "".join(c for c in i if c.isdigit() or c.isalpha() or c == " ")
    i = re.sub(r" \W+", " ", i)
    i = i.split()
    row = []
    for word in i:
        try:

            row.append(ghost[word])
        except:
            continue
    x_train1.append(row)
for i in x_train1:
    for j in range(len(i), 78):
        i.append(np.zeros(50))

x_train2 = []
indices_2 = []
for index, i in enumerate(evidence_2):

    i = i.lower().replace("[ref]", "")
    i = "".join(c for c in i if c.isdigit() or c.isalpha() or c == " ")
    i = re.sub(r" \W+", " ", i)
    i = i.split()
    row = []
    for word in i:
        try:
            row.append(ghost[word])
        except:
            continue
    x_train2.append(row)
for i in x_train2:
    for j in range(len(i), 78):
        i.append(np.zeros(50))

xtrain = np.array(x_train1 + x_train2)

ytrain = np.array(y_train1 + y_train2)

print(Counter(ytrain).most_common())

dim_input = 50
dim_recurrent = 150
dim_output = 1
rnn = RNN(dim_input, dim_recurrent, dim_output)
optimizer = Adam(rnn.parameters)

from noggin import create_plot

plotter, fig, ax = create_plot(metrics=["loss"])


def coolKidsLoss(pred, actual):
    return mg.mean(mg.square(pred - actual))


batch_size = 1

for epoch_cnt in range(100):
    idxs = np.arange(len(xtrain))
    np.random.shuffle(idxs)

    for batch_cnt in range(0, len(xtrain) // batch_size):
        batch_indices = idxs[batch_cnt * batch_size : (batch_cnt + 1) * batch_size]

        old = xtrain[batch_indices]
        batch = np.ascontiguousarray(np.swapaxes(old, 0, 1))
        prediction = rnn(batch)
        truth = ytrain[batch_indices]

        loss = coolKidsLoss(prediction, truth)

        loss.backward()

        optimizer.step()
        loss.null_gradients()

        plotter.set_train_batch({"loss": loss.item()}, batch_size=batch_size)
    plotter.set_train_epoch()


y_test = [
    panda["evidence_2_detection_score"][i]
    for i in range(len(stance2))
    if stance1[i] == stance2[i]
]
diff = 0
sum = 0
print(len(ytrain), len(xtrain))
for i in range(len(ytrain)):
    old = xtrain[i]
    w = np.ascontiguousarray(np.swapaxes(np.array(old).reshape(1, 78, 50), 0, 1))
    pred = rnn(w)
    true = ytrain[i]
    diff += mg.abs(pred - true)
    sum += true
print("diff: ", diff / len(ytrain))
print("mean: ", sum / len(ytrain))
print("std: ", np.std(ytrain))

i = 1
old = xtrain[i]
w = np.ascontiguousarray(np.swapaxes(np.array(old).reshape(1, 78, 50), 0, 1))
pred = rnn(w)
true = ytrain[i]
print(pred, true)

params = rnn.parameters
npparams = np.asarray(params)
np.save("ArgumentQualityModel", npparams)
