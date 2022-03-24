import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

import qiskit
from qiskit.visualization import *
from qiskit.opflow import Gradient, NaturalGradient, QFI, Hessian
from qiskit.opflow import Z, I, X, Y
from qiskit.opflow import StateFn
from qiskit.opflow import PauliExpectation, CircuitSampler
from qiskit.utils import QuantumInstance


from circuits import randomLayer, featureMap


# @Julia and @Tristan
class QuanvCircuit:
    """ Parameterizes Quanvolution circuit wrapper """
    def __init__(
            self,
            kernel_size=2,
            backend=None,
            shots=1024,
            ansatz='') -> None:
        
        # Instantiate quantum circuit
        self.n_qubits = kernel_size**2
        self.qc = qiskit.QuantumCircuit(self.n_qubits)
        
        fMap = featureMap(self.n_qubits)
        
        self.qc.compose(fMap, inplace=True)
        
        ansatz = randomLayer(self.n_qubits
            entanglement='full', 
            gates=['rx','ry'], 
            reps = 2)
        
        
        # create param vector 
        # create input param vector
        # apply appropriate gates
        self.params = ansatz.parameters
        self.n_params = ansatz.num_parameters
        
        self.input_data = fMap.parameters
        self.n_inputs = self.n_qubits
        
        self.backend = backend
        self.shots = shots
        
        self.q_instance = QuantumInstance(self.backend, shots = self.shots, seed_simulator = 2718, seed_transpiler = 2718)
        self.sampler = CircuitSampler(self.q_instance)
        self.shifter = Gradient(grad_method=grad)  # parameter-shift rule is the default
        self.hamiltonian = Z ^ Z ^ Z ^ Z
    
    def execute(self, input_data, params):
        # bind data to circuit
        # execute
        # extract ouput expectations
        
        expectation = StateFn(self.hamiltonian, is_measurement=True) @ StateFn(self._circuit.remove_final_measurements(inplace=False))
        
        value_dict = dict(zip(self.params + self.input_data, params + input_data))
        
        in_pauli_basis = PauliExpectation().convert(expectation)        
        result = self.sampler.convert(in_pauli_basis, params=value_dict).eval()
        
        return np.real(np.array([result]))
    
    def grad(self, input_data, params):
        
        expectation = StateFn(self.hamiltonian, is_measurement=True) @ StateFn(self._circuit.remove_final_measurements(inplace=False))
        
        expectation = expectation.bind_parameters(dict(zip(self.input_data, input_data)))
        
        grad = self.shifter.convert(expectation)
        gradient_in_pauli_basis = PauliExpectation().convert(grad)
        value_dict = dict(zip(self.params, params))
        
        result = np.array(self.sampler.convert(gradient_in_pauli_basis, params=value_dict).eval())
    
        return np.real(result)
        

class QuanvFunction(Function):
    """ Variational Quanvolution function definition """
    
    @staticmethod
    def forward(ctx, input_data, params, quantum_circuit):
        # forward pass of the quanvolutional function
        
        ctx.save_for_backwards(input_data, params)
        ctx.qc = quantum_circuit
        
        expectations = quantum_circuit.execute(input_data, params)
        result = torch.tensor([expectations])
        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        #backwards pass of the quanvolutional function
        
        # access saved objects
        input_data, params = ctx.saved_tensors
        
        # Gradients w.r.t each inputs to the function
        grad_input = grad_params = grad_qc = None
        
        # Compute gradients
        # @Tristan
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        
        gradients = ctx.quantum_circuit.grad(input_list).T
                
        grad_params = torch.tensor([gradients.tolist()]).float() * grad_output.float()
        
        return grad_params, grad_input, grad_qc



class QuanvLayer(nn.Module):
    """ Quanvolution layer definition """
    
    def __init__(
            self,
            in_channels,
            out_channels=4,
            kernel_size=2,
            stride=1,
            shots=100,
            backend=qiskit.Aer.get_backend('qasm_simulator')):
            
        super(QuanvLayer, self).__init__()
        
        self.qc = QuanvCircuit(kernel_size=kernel_size, backend=backend, shots=shots) # TODO: multiple circuits? analogue to multiple kernel CNN
                
        self.in_channels = in_channels
        self.out_channels = out_channels  # TODO: what do we do with out_channels - look at CNN
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.parameters = nn.Parameter(torch.empty(self.get_parameter_shape()))
        nn.init.uniform_(self.parameters, -0.1, 0.1)

    def _get_parameter_shape(self):
        """Computes the number of trainable parameters required by the quantum circuit function"""
        
        # TODO: implement based on some ansatz specification (see get_quanv_fn) that is either provided
        # to the object or a global default
        
        return (4,)
        

    def _get_out_dim(self, img):
        bs, h, w, ch = img.size()
        h_out = (int(h) - self.kernel_size) // self.stride + 1
        w_out = (int(w) - self.kernel_size) // self.stride + 1
        return bs, h_out, w_out, self.out_channels


    def convolve(self, imgs):
        """Get input to circuit following a convolution pattern"""
        
        # @Robbie
    
        yield data, batch_idx, row, col


    def forward(self, imgs):
        """Apply variational quanvolution layer to image
        
        Parameters
        ----------
        imgs : np.ndarray
            A vector of input images. Should have shape [batch_size, height, width, n_channels].
        """
        
        out = torch.empty(self._get_out_dim(imgs), dtype=torch.float32)
        
        # @Robbie - iterate over image and apply function. Ex:
        for data, _, _, _ in self.convolve(imgs):
            res = QuanvFunction.apply(data, self.parameters, self.qc)
            
            out[batch_idx, row // self.stride, col // self.stride] = res


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