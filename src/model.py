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
from qiskit.circuit.random import random_circuit

from circuits import randomLayer


# @Julia and @Tristan
class QuanvCircuit:
    """ Parameterizes Quanvolution circuit wrapper """
    def __init__(
            self,
            kernel_size=2,
            backend=None,
            shots=1024,
            threshold=127,
            ansatz='') -> None:
        
        # Instantiate quantum circuit
        self.qc = qiskit.QuantumCircuit()
        # create param vector 
        # create input param vector
        # apply appropriate gates
    
    def execute(self, input_data, params):
        # bind data to circuit
        # execute
        # extract ouput expectations
        expectation = StateFn(self.qc.remove_final_measurements(inplace=False))
        
        value_dict = dict(zip(self.params + self.inputs, params + input_data))
        
        in_pauli_basis = PauliExpectation().convert(expectation)        
        result = self.sampler.convert(in_pauli_basis, params=value_dict).eval()
        return result.to_matrix()[0]
    
    def grad(self, input_data, params):
        
        expectation = StateFn(self._circuit.remove_final_measurements(inplace=False))
        
        expectation = expectation.bind_parameters(dict(zip(self.inputs, input_data)))
        
        grad = self.shifter.convert(expectation)
        gradient_in_pauli_basis = PauliExpectation().convert(grad)
        value_dict = dict(zip(self.params, params))
        
        result = self.sampler.convert(gradient_in_pauli_basis, params=value_dict).eval()
        
        gradInputs = np.array([g.toarray() for g in result])
    
        return gradInputs[:,0]
        

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
        qc_fn = ctx.qc_fn
        
        # Gradients w.r.t each inputs to the function
        grad_input = grad_params = grad_qc = None
        
        # Compute gradients
        # @Tristan
        
        return grad_input, grad_params, grad_qc



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
        return 0