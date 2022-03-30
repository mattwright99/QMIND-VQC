import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit,Aer
from qiskit.opflow import Gradient, NaturalGradient, QFI, Hessian
from qiskit.opflow import Z, I, X, Y
from qiskit.opflow import StateFn
from qiskit.opflow import PauliExpectation, CircuitSampler
from qiskit.utils import QuantumInstance


from circuits import randomLayer, featureMap, quanvolutionESU2


class QuanvCircuit:
    def __init__(
            self,
            kernel_size=2,
            backend=Aer.get_backend('qasm_simulator'),
            shots=128,
            ansatz=None,
            feature_map=None
            ) -> None:
        """Parameterized quanvolution circuit wrapper.
        
        Parameters
        ----------
        kernel_size : int
            Width of square filter used for convolution.
        backend : qiskit.providers.Backend 
            Qiskit quantum backend to execute circuits on.
        shots : int
            Number of shots used for circuit execution.
        """
        
        # Instantiate quantum circuit
        self.n_qubits = kernel_size**2
        self.qc = QuantumCircuit(self.n_qubits)
        
        fMap, input_vars = feature_map  # feature map to encode input data
        self.qc.compose(fMap, inplace=True)
        
        ansatz, weights_vars = ansatz
        self.qc.compose(ansatz, inplace=True)
        
        # Save useful parameter sizes for variational weights and input data
        self.weight_vars = weights_vars
        self.n_weights = len(weights_vars)

        self.input_vars = input_vars
        self.n_inputs = self.n_qubits
        
        # Configure quantum instance
        self.backend = backend
        self.shots = shots
        
        self.q_instance = QuantumInstance(self.backend, shots=self.shots, seed_simulator=2718, seed_transpiler=2718)
        self.sampler = CircuitSampler(self.q_instance)
        self.shifter = Gradient()  # parameter-shift rule is the default
        self.hamiltonian = Z ^ Z ^ Z ^ Z


    def execute(self, input_data, weights):
        if isinstance(input_data, torch.Tensor):
            input_data = np.array(input_data.tolist())
        if isinstance(weights, torch.Tensor):
            weights = np.array(weights.tolist())
        
        input_data = np.pi * input_data  # scale data from [0,1] to [0, pi]
        # Set measurement expectation
        expectation = StateFn(self.hamiltonian, is_measurement=True) @ StateFn(self.qc)
        in_pauli_basis = PauliExpectation().convert(expectation)        

        # Dind values to circuit and get expectation value
        value_dict = dict(zip(self.weight_vars, weights))
        value_dict.update(dict(zip(self.input_vars, input_data)))
        result = self.sampler.convert(in_pauli_basis, params=value_dict).eval()
        
        return np.real(np.array([result]))


    def grad_weights(self, input_data, weights):
        if isinstance(input_data, torch.Tensor):
            input_data = np.array(input_data.tolist())
        if isinstance(weights, torch.Tensor):
            weights = np.array(weights.tolist())

        input_data = 2*np.pi * input_data  # scale data from [0,1] to [0, pi]

        expectation = StateFn(self.hamiltonian, is_measurement=True) @ StateFn(self.qc)
        expectation = expectation.bind_parameters(dict(zip(self.input_vars, input_data)))
        
        grad = self.shifter.convert(expectation)
        gradient_in_pauli_basis = PauliExpectation().convert(grad)
        value_dict = dict(zip(self.weight_vars, weights))
        
        result = np.array(self.sampler.convert(gradient_in_pauli_basis, params=value_dict).eval())
    
        return np.real(result)

    def grad_input(self, input_data, weights):
        if isinstance(input_data, torch.Tensor):
            input_data = np.array(input_data.tolist())
        if isinstance(weights, torch.Tensor):
            weights = np.array(weights.tolist())
    
        input_data = 2*np.pi * input_data  # scale data from [0,1] to [0, pi]
                        
        expectation = StateFn(self.hamiltonian, is_measurement=True) @ StateFn(self.qc)
        expectation = expectation.bind_parameters(dict(zip(self.weight_vars, weights)))
                
        grad = self.shifter.convert(expectation)
        gradient_in_pauli_basis = PauliExpectation().convert(grad)
        value_dict = dict(zip(self.input_vars, input_data))
        
        result = np.array(self.sampler.convert(gradient_in_pauli_basis, params=value_dict).eval())
    
        return np.real(result)


class QuanvFunction(Function):
    """Variational Quanvolution function definition."""
    
    @staticmethod
    def forward(ctx, input_data, weights, quantum_circuit):
        # forward pass of the quanvolutional function
        
        ctx.weights = weights
        ctx.input_data = input_data
        ctx.qc = quantum_circuit
        

        expectations = quantum_circuit.execute(input_data, weights)
        result = torch.tensor(np.expand_dims(expectations, axis=0))
    
        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        # backwards pass of the quanvolutional function
        
        # access saved objects
        params = ctx.weights
        input_data = ctx.input_data
        qc = ctx.qc

        # Gradients w.r.t each inputs to the function
        grad_input = grad_params = grad_qc = None
        
        # Compute gradients
        if ctx.needs_input_grad[0]:
            gradients = qc.grad_input(input_data, params).T
            grad_input = torch.tensor([gradients.tolist()]).float() * grad_output.float()
        if ctx.needs_input_grad[1]:
            gradients = qc.grad_weights(input_data, params).T
            grad_params = torch.tensor([gradients.tolist()]).float() * grad_output.float()

        return grad_input, grad_params, grad_qc



class QuanvLayer(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=4,
            kernel_size=2,
            stride=1,
            shots=128,
            backend=Aer.get_backend('qasm_simulator')):
        """Parameterized quanvolution layer.
        
        Parameters
        ----------
        in_channels : int
            Number of inputs channels.
        out_channels : int 
            Number of output channels. Equivalent to number of quanvolution kernels to apply.
        kernel_size : int
            Width of square filter used for convolution.
        stride : int
            Step size used for convolution.
        backend : qiskit.providers.Backend 
            Qiskit quantum backend to execute circuits on.
        shots : int
            Number of shots used for circuit execution.
        """

        super(QuanvLayer, self).__init__()
        
        if in_channels != 1:
            raise Exception(f'Only support 1 input channel but got {in_channels}')

        self.quantum_circuits = [
            QuanvCircuit(kernel_size=kernel_size, 
                         backend=backend, 
                         shots=shots, 
                         ansatz=quanvolutionESU2(  # parameterized ansatz
                            kernel_size**2,
                            entanglement='full', 
                            gates=['rx','ry'], 
                            reps=4),
                         feature_map=featureMap(kernel_size**2))
            for c in range(out_channels)
        ]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weights = nn.Parameter(torch.empty(self._get_parameter_shape()))
        nn.init.uniform_(self.weights, -0.1, 0.1)

    def _get_parameter_shape(self):
        """Computes the number of trainable parameters required by the quantum circuit functions"""

        n_filters = len(self.quantum_circuits)
        n_params_per_circ = self.quantum_circuits[0].n_weights
        return (n_filters, n_params_per_circ)

    def _get_out_dim(self, imgs):
        """Get dimensions of output tensor after applying convolution"""

        bs, ch, h, w = imgs.shape
        h_out = (h - self.kernel_size) // self.stride + 1
        w_out = (w - self.kernel_size) // self.stride + 1
        return bs, self.out_channels, h_out, w_out

    def convolve(self, imgs):
        """Get input to circuit following a convolution pattern"""

        _, nchannels, height, width = imgs.shape
        if nchannels != 1:
            raise Exception(f'Only support images with one channel but got {nchannels}')

        # Iterate over all images in batch
        for batch_idx, img in enumerate(imgs):
            for r in range(0, height - self.kernel_size, self.stride):
                for c in range(0, width - self.kernel_size, self.stride):
                    # Grab section of image under filter
                    data = img[0, r : r + self.kernel_size, c : c + self.kernel_size]
                    yield data.flatten(), batch_idx, r, c

    def forward(self, imgs):
        """Apply variational quanvolution layer to image

        Parameters
        ----------
        imgs : np.ndarray or torch.Tensor
            A vector of input images. Should have shape [batch_size, n_channels, height, width].

        Returns
        -------
        torch.Tensor
            Processed results.
        """

        output = torch.empty(self._get_out_dim(imgs), dtype=torch.float32)

        # Convolve over given images
        for data, batch_idx, row, col in self.convolve(imgs):
            # Process data with each quanvolutional circuit
            for ch in range(self.out_channels):
                qc = self.quantum_circuits[ch]
                weights = self.weights[ch]

                res = QuanvFunction.apply(data, weights, qc)
                output[batch_idx, ch, row//self.stride, col//self.stride] = res

        return output


class QuanvNet(nn.Module):
    """Overall model architecture that applies the quanvolutional layer"""
    def __init__(self, input_size=8, shots=128):
        super(QuanvNet, self).__init__()

        self.fc_size = (input_size - 3)**2 * 16  # output size of convloving layers
        self.quanv = QuanvLayer(in_channels=1, out_channels=2, kernel_size=2, shots=shots)
        self.conv = nn.Conv2d(2, 16, kernel_size=3)
        # self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(self.fc_size, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # this is where we build our entire network
        # whatever layers of quanvolution, pooling,
        # convolution, dropout, flattening,
        # fully connectecd layers, go here
        x = F.relu(self.quanv(x))
        x = F.relu(self.conv(x))
        x = x.view(-1, self.fc_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ClassicNet(nn.Module):
    """Overall model architecture that applies a classical version of our model"""
    def __init__(self, input_size=8):
        super(QuanvNet, self).__init__()

        self.fc_size = (input_size - 3)**2 * 16  # output size of convloving layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3)
        # self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(self.fc_size, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # this is where we build our entire network
        # whatever layers of quanvolution, pooling,
        # convolution, dropout, flattening,
        # fully connectecd layers, go here
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.fc_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
