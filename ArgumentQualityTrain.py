import pandas as pd
import mygrad as mg
import numpy as np
import re
from mynn.layers.dense import dense
from mynn.initializers.glorot_normal import glorot_normal
from mynn.optimizers.adam import Adam
from mygrad.nnet.losses import softmax_crossentropy
from gensim.models.keyedvectors import KeyedVectors
from noggin import create_plot

"""
Running this file trains an argument quality model and saves it to a file ArgumentQualityModel.npy.
Do not run this every time you want to test the model on your data.
Only run this once at the beginning to create the ArgumentQuality.npy file
    or to retrain the model.
"""

# Loads glove, which contains english words and their embeddings into 50-dimensional vectors
glove = KeyedVectors.load_word2vec_format("glove.6B.50d.txt.w2v", binary=False)


class RNN:  # The RNN class, which passes the data through a gated recurrent unit to convert each sentence into an array
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


counters = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Reads the csv file containing the train data
panda = pd.read_csv("train.csv")

# Creates arrays containing the evidence given that both stances are the same.
stance1 = np.array(panda["evidence_1_stance"])
stance2 = np.array(panda["evidence_2_stance"])
evidence_1 = [
    panda["evidence_1"][i] for i in range(len(stance1)) if stance1[i] == stance2[i]
]
evidence_2 = [
    panda["evidence_2"][i] for i in range(len(stance1)) if stance1[i] == stance2[i]
]

# Creates arrays containing the scores of the evidence given that both stances are the same.
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

# Takes in the text and converts it into an array of sentences.
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
            row.append(glove[word])
        except:
            continue
    x_train1.append(row)

# Makes all sentence arrays the same shape by adding arrays of zeros to the end.
for i in x_train1:
    for j in range(len(i), 78):
        i.append(np.zeros(50))

# Repeats the process above for the 2nd half of the train data
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
            row.append(glove[word])
        except:
            continue
    x_train2.append(row)
for i in x_train2:
    for j in range(len(i), 78):
        i.append(np.zeros(50))

# Concatenates the 2 sets of train data
xtrain = np.array(x_train1 + x_train2)

ytrain = np.array(y_train1 + y_train2)

# Initializes the optimizer and the model with parameters.
dim_input = 50
dim_recurrent = 150
dim_output = 1
rnn = RNN(dim_input, dim_recurrent, dim_output)
optimizer = Adam(rnn.parameters)


plotter, fig, ax = create_plot(metrics=["loss"])


def l2loss(pred, actual):  # L2 loss function (mean square distance)
    """

    Parameters
    ----------
    pred: Union[mygrad.Tensor, numpy.ndarray]
        A tensor or numpy array containing the model's predicted values
    actual: Union[mygrad.Tensor, numpy.ndarray]
        A tensor or numpy array containing the actual values

    Returns
    -------
    mg.Tensor
        A tensor containing the mean square distance between the prediction and actual values.
    """
    return mg.mean(mg.square(pred - actual))


batch_size = 1

# Trains the model over 10 epochs.
for epoch_cnt in range(10):
    idxs = np.arange(len(xtrain))
    np.random.shuffle(idxs)
    print("epoch", epoch_cnt)

    for batch_cnt in range(0, len(xtrain) // batch_size):
        batch_indices = idxs[batch_cnt * batch_size : (batch_cnt + 1) * batch_size]

        old = xtrain[batch_indices]
        batch = np.ascontiguousarray(np.swapaxes(old, 0, 1))
        prediction = rnn(batch)
        truth = ytrain[batch_indices]

        loss = l2loss(prediction, truth)

        loss.backward()

        optimizer.step()
        loss.null_gradients()

        plotter.set_train_batch({"loss": loss.item()}, batch_size=batch_size)
    plotter.set_train_epoch()


diff = 0
sum = 0

# Tests the model
for i in range(len(ytrain)):
    old = xtrain[i]
    w = np.ascontiguousarray(np.swapaxes(np.array(old).reshape(1, 78, 50), 0, 1))
    pred = rnn(w)
    true = ytrain[i]
    diff += mg.abs(pred - true)
    sum += true


i = 1
old = xtrain[i]
w = np.ascontiguousarray(np.swapaxes(np.array(old).reshape(1, 78, 50), 0, 1))
pred = rnn(w)
true = ytrain[i]

# Saves the model as "ArgumentQualityModel.npy"
params = rnn.parameters
npparams = np.asarray(params)
np.save("ArgumentQualityModel", npparams)
