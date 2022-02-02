def real_amplitude_ansatz(gates, entanglement_type, repititions, barrier):
    qubits = gates
    qc = QuantumCircuit(qubits)
    parameters = ParameterVector('θ', qubits*(qubits+1))
    for reps in range(repititions):
        for i in range(qubits):
            qc.ry(theta = parameters[i], qubit = i)
        
        for i in range(qubits - 1):
            if entanglement_type == 'linear':
                qc.cx(i, i+1)
            
        for i in range(qubits):
            if entanglement_type == 'circular':
                qc.cx(i-1, i)
            
        for i in range(qubits):
            for j in range(i):
                if entanglement_type == 'full':
                    qc.cx(j,i)
                    
        if barrier == True:
            qc.barrier()
            
    return qc
    
a = real_amplitude_ansatz(4, 'circular', 2, True)
a.draw()        
    