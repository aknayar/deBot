import pickle
import numpy as np
from noggin import create_plot
import factuality_model as fm
from mynn.optimizers.adam import Adam
from mygrad.nnet.losses import softmax_crossentropy
import mygrad as mg
import matplotlib.pyplot as plt


def load_pickle():
    pickle_in = open("factual.pickle", "rb")
    fact_dict = pickle.load(pickle_in)
    return fact_dict


def split(dictionary):
    keys = [key for key in dictionary]
    values = np.squeeze(np.array(list(dictionary[key] for key in keys)))
    return keys, values


def train():
    fact_dict = load_pickle()

    sentences, ratings = split(fact_dict)

    plotter, fig, ax = create_plot(["loss", "accuracy"])

    model = fm.Model(dim_input=50, dim_recurrent=100, dim_output=1)
    optimizer = Adam(model.parameters)

    plot_every = 500

    for k in range(100000):
        output = model(sentences)

        loss = softmax_crossentropy(output, ratings)

        acc = float(output.data.squeeze() == ratings.item())

        plotter.set_train_batch(
            {"loss": loss.item(), "accuracy": acc}, batch_size=1, plot=False
        )

        if k % plot_every == 0 and k > 0:
            plotter.set_train_epoch()

        loss.backward()
        optimizer.step()
        loss.null_gradients()
