import sys
import os
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import neuron_models as nm
import ans as ans

def test_input_current():
    # Test current variable, s
    net = nm.SpikingNetwork()
    # One spike at t=0, which will affect A's input current in the next time step.
    inA = nm.InputNeuron([0.], label='inA')
    A = ans.LIFNeuron(label='A')
    net.add_neuron(inA)     # neuron index 0
    net.add_neuron(A)       # neuron index 1
    inA.connect_to(A, 0.5)  # connect inA -> A
    net.simulate(0.003, 0.001)

    assert net.neur[1].get_s_history() == pytest.approx(np.array([0., 0., 10.]), abs=1e-4)
