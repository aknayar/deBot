from mynn.layers.dense import dense
from mynn.initializers.glorot_normal import glorot_normal
from mynn.activations.relu import relu
import numpy as np


class Model:
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

    def __call__(self, x):
        """ Performs the full forward pass for the RNN.

        Note that we only care about the last y - the final classification scores for the full sequence

        Parameters
        ----------
        x: Union[numpy.ndarray, mygrad.Tensor], shape=(M, N, 50)
            The word embeddings for the sequence

        Returns
        -------
        mygrad.Tensor, shape=(M,)
            The final classification of the sequence
        """
        h = np.zeros((1, self.fc_h2h.weight.shape[0]), dtype=np.float32)
        for x_t in x:
            h = relu(self.fc_x2h(x_t[np.newaxis]) + self.fc_h2h(h))

        return self.fc_h2y(h)

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
