from qiskit.circuit import QuantumCircuit, ParameterVector, Gate, Parameter
from typing import Union
import numpy as np


def quanvolutionESU2(N_dims, gates=['rx', 'rz'], reps=1, entanglement='circular', insert_barrier=False):
    
    # Function to apply rotation gates to all of our qubits
    def rotation(gate, start):
        qc = QuantumCircuit(N_dims, name=gate)
        for i in range(N_dims):
            if gate == 'rx':
                qc.rx(parameters[start+i], i)
            if gate == 'ry':
                qc.ry(parameters[start+i], i)
            if gate == 'rz':
                qc.rz(parameters[start+i], i)
        return qc
    
    # Function apply the entanglement
    def entanglement(type=entanglement):
        qc = QuantumCircuit(N_dims, name=type)
        if (type == "circular"):
            qc.cx(N_dims-1, 0)
            for i in range(N_dims-1):
                target = i + 1
                qc.cx(i, target)
        if (type == "linear"):
            for i in range(N_dims - 1):
                target = i + 1
                qc.cx(i, target)
        if (type == "full"):
            for i in range(N_dims):
                for j in range(N_dims):
                    if (i != j):
                        qc.cx(i, j)
        return qc
    
    # Calculate the number of parameters we will need
    num_params = len(gates)*N_dims*(reps + 1)
    parameters = ParameterVector('theta', num_params)
    qc = QuantumCircuit(N_dims, name="EfficientSU2")
    start = 0
    
    for i in range(reps):
        for gate in gates:
            qc.compose(rotation(gate, start), range(N_dims), inplace=True)
            start += N_dims
        
        if N_dims > 1: qc.compose(entanglement(), range(N_dims), inplace=True)
        
        if (reps == 1 or i == reps-1):
            if insert_barrier: qc.barrier()
            for gate in gates:
                qc.compose(rotation(gate, start), range(N_dims), inplace=True)
                start += N_dims
        if insert_barrier: qc.barrier()
    
    return qc, parameters

def randomLayer(numQbits, gates=['rx', 'rz', 'ry'], entanglement='linear', reps=1, to_gate=True) -> Union[Gate, QuantumCircuit]:
    qc = QuantumCircuit(numQbits)
    insert_barrier = False if to_gate else True 
    qc.compose(quanvolutionESU2(numQbits, gates=gates, entanglement=entanglement, reps=reps, 
                                insert_barrier=insert_barrier), inplace=True)
    return qc


def featureMap(n_qubits, to_gate='False') -> Union[Gate, QuantumCircuit]:
    qc = QuantumCircuit(n_qubits)
    parameters = ParameterVector('input', n_qubits)
    for i in range(n_qubits):
        qc.ry(parameters[i], i)
    return qc, parameters


def IQP_Mat(int_mat, insert_barrier=False):
    n = len(int_mat[0][:])
    
    for i in range(n):
        for j in range(n-1, i, -1):
            if(int_mat[i][j] != int_mat[j][i]):
                print("Not a symmetrical interaction matrix!")
                return
    
    qc = QuantumCircuit(n)
    qc.h(range(n))
    
    if(insert_barrier):
        qc.barrier()
    
    for i in range(n):
        for j in range(n-1, i, -1):
            x = int_mat[i][j]
            qc.cp(x, i, j)
    if(insert_barrier):
        qc.barrier()
    
    for i in range(n):
        qc.p(int_mat[i][i], i)
        
    qc.h(range(n))
 
    return qc

def IQP(n, insert_barrier=False):
    num_params = np.sum([i for i in range(n+1)])
    params = ParameterVector('θ', num_params)
    paramMat = np.empty([n,n], dtype=Parameter)
    for i in range(n):
        paramMat[i][i] = params[i]
        for j in range(n):
            if(i != j):
                paramMat[i][j] = params[n-1+i+j]
                paramMat[j][i] = paramMat[i][j]
    
    return IQP_Mat(paramMat, insert_barrier=insert_barrier), params

#def threshold_feature_map(n_qubits, to_gate='False') -> Union[Gate, QuantumCircuit]:
    
        
