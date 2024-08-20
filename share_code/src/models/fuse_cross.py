from torch import nn
import torch
from src.models.layers import *
from src.models.att import  MultiHeadedAttention

#speech processing
class Fuse(nn.Module):
    def __init__(
        self,
        K=64,
        d_a=int(512/64),
        input_shape = (32, None, 40),
        layer_sizes = [256,256,256,256,256,100],
        neuron_type="ourRLIFSA",
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        use_readout_layer=True,
    ):
        super().__init__()

        # Fixed parameters
        self.K = K
        self.d_a = d_a
        self.reshape = True if len(input_shape) > 3 else False
        self.input_size = float(torch.prod(torch.tensor(input_shape[2:])))
        self.batch_size = input_shape[0]
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.num_outputs = layer_sizes[-1]
        self.neuron_type = neuron_type
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.use_readout_layer = use_readout_layer
        self.is_snn = True

        if neuron_type not in ["LIF", "adLIF", "RLIF", "RadLIF", "ourLIFSA", "ourRLIFSA"]:
            raise ValueError(f"Invalid neuron type {neuron_type}")

        # Init trainable parameters
        self.snn = self._init_layers()


    def _init_layers(self):

        snn = nn.ModuleList([])
        input_size = self.input_size
        snn_class = self.neuron_type + "Layer"
        print (snn_class)

        if self.use_readout_layer:
            num_hidden_layers = self.num_layers - 1
        else:
            num_hidden_layers = self.num_layers

        # frontend
        snn.append(globals()['ourRLIFLayer'](
                    input_size=input_size,
                    hidden_size=self.layer_sizes[0],
                    batch_size=self.batch_size,
                    threshold=self.threshold,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,))
        input_size = self.layer_sizes[0]
        snn.append(globals()['ourRLIFLayer'](
                    input_size=input_size,
                    hidden_size=self.layer_sizes[1],
                    batch_size=self.batch_size,
                    threshold=self.threshold,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,))
        input_size = self.layer_sizes[1]
        # attention speech block
        snn.append(globals()['ourLIFSALayer'](
                    thres = 1.0,
                    input_size=input_size,
                    hidden_size=self.layer_sizes[2],
                    batch_size=self.batch_size,
                    threshold=self.threshold,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,))
        input_size = self.layer_sizes[2]
         # attention speech block
        snn.append(globals()['ourLIFSALayer'](
                    thres = 1.0,
                    input_size=input_size,
                    hidden_size=self.layer_sizes[3],
                    batch_size=self.batch_size,
                    threshold=self.threshold,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,))
        input_size = self.layer_sizes[3]
         # attention speech block
        snn.append(globals()['ourLIFSALayer'](
                    thres = 1.0,
                    input_size=input_size,
                    hidden_size=self.layer_sizes[4],
                    batch_size=self.batch_size,
                    threshold=self.threshold,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,))
        input_size = self.layer_sizes[4]
        if self.use_readout_layer:
            snn.append(
                ReadoutLayer(
                    input_size=input_size,
                    hidden_size=self.layer_sizes[-1],
                    batch_size=self.batch_size,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,
                )
            )

        return snn

    def forward(self, v, x):

        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            else:
                raise NotImplementedError

        # Process all layers
        for i, snn_lay in enumerate(self.snn):
            x = snn_lay(v,x)

        return x

class ourLIFSALayer(nn.Module):
    def __init__(
        self,
        thres,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.1,
        normalization="batchnorm",
        use_bias=False,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.batch_size = self.batch_size
        self.tau = 0.5
        self.gama = 1.0
        self.act = ZIF.apply
        self.thres = thres
        #VCA2M
        self.ca = MultiHeadedAttention(8, 100, self.thres, self.hidden_size)
        # Trainable parameters
        self.W = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, v, a):

        if self.batch_size != a.shape[0]:
            self.batch_size = a.shape[0]

        x = self.ca(v, a, a)
        x = x + a
        Wx = self.W(x)

        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        s = self._ourlif_cell(Wx)
        s = self.drop(s)

        return s

    def _ourlif_cell(self, Wx):

        device = Wx.device
        ut = 0
        st = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Loop over time axis
        for t in range(Wx.shape[1]):

            ut = self.tau * ut + Wx[:, t, :]

            # Compute spikes with surrogate gradient
            st = self.act(ut - self.threshold, self.gama)
            ut = (1 - st) * ut
            s.append(st)

        return torch.stack(s, dim=1)


class ourRLIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons with layer-wise
    recurrent connections (RLIF).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.batch_size = self.batch_size
        self.tau = 0.5
        self.gama = 1.0
        self.act = ZIF.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        nn.init.orthogonal_(self.V.weight)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, v, x):

        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        Wx = self.W(x)

        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        s = self._ourrlif_cell(Wx)
        s = self.drop(s)

        return s

    def _ourrlif_cell(self, Wx):

        device = Wx.device
        ut = 0
        st = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Set diagonal elements of recurrent matrix to zero
        V = self.V.weight.clone().fill_diagonal_(0)

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute membrane potential (RLIF)
            ut = self.tau * ut + (Wx[:, t, :] + torch.matmul(st, V))

            # Compute spikes with surrogate gradient
            st = self.act(ut - self.threshold, self.gama)
            ut = (1 - st) * ut
            s.append(st)

        return torch.stack(s, dim=1)
    
class ReadoutLayer(nn.Module):
    """
    This function implements a single layer of non-spiking Leaky Integrate and
    Fire (LIF) neurons, where the output consists of a cumulative sum of the
    membrane potential using a softmax function, instead of spikes.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.use_bias = use_bias

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)

    def forward(self, v, a):

        Wx = self.W(a)

        return Wx

