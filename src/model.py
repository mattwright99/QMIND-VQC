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


from circuits import randomLayer, featureMap


class QuanvCircuit:
    def __init__(
            self,
            kernel_size=2,
            backend=None,
            shots=1024) -> None:
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
        
        fMap = featureMap(self.n_qubits)  # feature map to encode input data
        self.qc.compose(fMap, inplace=True)
        
        ansatz = randomLayer(  # parameterized ansatz
            self.n_qubits  
            entanglement='full', 
            gates=['rx','ry'], 
            reps=2
        )
        
        # Save useful parameter sizes for variational weights and input data
        self.weight_vars = ansatz.parameters
        self.n_weights = ansatz.num_parameters

        self.input_vars = fMap.parameters
        self.n_inputs = self.n_qubits
        
        # Configure quantum instance
        self.backend = backend
        self.shots = shots
        
        self.q_instance = QuantumInstance(self.backend, shots=self.shots, seed_simulator=2718, seed_transpiler=2718)
        self.sampler = CircuitSampler(self.q_instance)
        self.shifter = Gradient()  # parameter-shift rule is the default
        self.hamiltonian = Z ^ Z ^ Z ^ Z
    
    def execute(self, input_data, weights):
        # Set measurement expectation
        expectation = StateFn(self.hamiltonian, is_measurement=True) @ StateFn(self.qc)
        in_pauli_basis = PauliExpectation().convert(expectation)        

        # Dind values to circuit and get expectation value
        value_dict = dict(zip(self.weight_vars + self.input_vars, weights + input_data))
        result = self.sampler.convert(in_pauli_basis, params=value_dict).eval()
        
        return np.real(np.array([result]))
    
    def grad_weights(self, input_data, weights):
        
        expectation = StateFn(self.hamiltonian, is_measurement=True) @ StateFn(self.qc)
        expectation = expectation.bind_parameters(dict(zip(self.input_vars, input_data)))
        
        grad = self.shifter.convert(expectation)
        gradient_in_pauli_basis = PauliExpectation().convert(grad)
        value_dict = dict(zip(self.weight_vars, weights))
        
        result = np.array(self.sampler.convert(gradient_in_pauli_basis, params=value_dict).eval())
    
        return np.real(result)

    def grad_input(self, input_data, weights):
        pass


class QuanvFunction(Function):
    """Variational Quanvolution function definition."""
    
    @staticmethod
    def forward(ctx, input_data, weights, quantum_circuit):
        # forward pass of the quanvolutional function
        
        ctx.save_for_backwards(input_data, weights)
        ctx.qc = quantum_circuit
        
        expectations = quantum_circuit.execute(input_data, weights)
        result = torch.tensor([expectations])
        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        # backwards pass of the quanvolutional function
        
        # access saved objects
        input_data, params = ctx.saved_tensors
        qc = ctx.qc

        # Gradients w.r.t each inputs to the function
        grad_input = grad_params = grad_qc = None
        
        input_list = np.array(input_data.tolist())
        param_list = np.array(params.tolist())
        # Compute gradients
        if ctx.needs_input_grad[0]:
            gradients = qc.grad_input(input_list, param_list).T
            grad_input = torch.tensor([gradients.tolist()]).float() * grad_output.float()
        if ctx.needs_input_grad[1]:
            gradients = qc.grad(input_list, param_list).T
            grad_params = torch.tensor([gradients.tolist()]).float() * grad_output.float()

        return grad_input, grad_params, grad_qc



class QuanvLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels=4,
            kernel_size=2,
            stride=1,
            shots=100,
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
        
        self.quantum_circuits = [
            QuanvCircuit(kernel_size=kernel_size, backend=backend, shots=shots)
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

        bs, h, w, ch = imgs.size()
        h_out = (int(h) - self.kernel_size) // self.stride + 1
        w_out = (int(w) - self.kernel_size) // self.stride + 1
        return bs, h_out, w_out, self.out_channels

    def convolve(self, imgs):
        """Get input to circuit following a convolution pattern"""

        _, height, width, _ = imgs.size()
        # Iterate over all images in batch
        for batch_idx, img in enumerate(imgs):
            # Rows
            for r in range(0, height - self.kernel_size, self.stride):
                # Columns
                for c in range(0, width - self.kernel_size, self.stride):
                    # Grab section of image under filter
                    data = img[r : r + self.kernel_size, c : c + self.kernel_size]
                    yield data.flatten(), batch_idx, r, c

    def forward(self, imgs):
        """Apply variational quanvolution layer to image
        
        Parameters
        ----------
        imgs : np.ndarray
            A vector of input images. Should have shape [batch_size, height, width, n_channels].
        
        Returns
        -------
        # TODO
        """
        
        output = torch.empty(self._get_out_dim(imgs), dtype=torch.float64)
        
        # Convolve over given images
        for data, batch_idx, row, col in self.convolve(imgs):
            # Process data with each quanvolutional circuit
            for channel in range(self.out_channels):
                qc = self.quantum_circuits[channel]
                weights = self.weights[channel]
                
                res = QuanvFunction.apply(data, weights, qc)
                output[batch_idx, row // self.stride, col // self.stride, channel] = res


class QuanvNet(nn.Module):
    """ Overall model architecture that applies the quanvolutional layer """
    def __init__(self):
        super(QuanvNet, self).__init__()
        self.quanv = QuanvLayer(in_channels=1, out_channels=4, kernel_size=2)
        self.conv = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        #this is where we build our entire network
        #whatever layers of quanvolution, pooling,
        #convolution, dropout, flattening,
        #fully connectecd layers, go here
        x = F.relu(self.quanv(x))
        x = self.conv(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x