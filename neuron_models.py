# (C) Jeff Orchard, 2025
# University of Waterloo

import numpy as np
import matplotlib.pyplot as plt


#================================
#
#  Helper Functions
#
#================================
def generate_spike_train(times_rates, jitter=0.):
    '''
    spike_times = generate_spike_train(times_rates, T)
    
    Creates a spike train (as an array of time stamps).
    
    Input:
    times_rates is an array or list of (t,r) pairs, so that the
        firing rate is changed to r (Hz) at time t (seconds).
        If the last firing rate is not zero, the last interval is
        extended for 1s, and then the rate goes to zero.
    jitter is a scalar that determines how much the spikes
        are randomly moved
        
    Output:
    spike_times is an array of times when spikes occurred
    
    Example: To create a spike train of 10Hz for 0.5s, followed
             by 25Hz that starts at 0.5s and ends at 2s, use
             
             s = generate_spike_train([[0.,10], [0.5,25], [2,0]])
    '''
    s = []
    tr = list(times_rates)
    rate_last = tr[-1][1]
    # Make sure it ends with a zero
    if rate_last != 0.:
        # If the last rate was not zero, extend for 1s and then set to 0.
        tr.append([tr[-1][0]+1., 0.])
    t = tr[0][0]
    for idx in range(0,len(tr)-1):
        t_interval = tr[idx+1][0] - t
        r = tr[idx][1]
        if r!=0:
            delta = 1. / r
            N = r * t_interval
            times = np.arange(t+delta/2., t+t_interval, delta)
            if jitter!=0.:
                times += np.random.normal(scale=delta*jitter, size=np.shape(times))
            s.extend(times)
        t = tr[idx+1][0]
    s.sort()
    return np.array(s)


def plot_spike_raster(st, y_range=None):
    '''
    plot_spike_raster(spiketimes, y_range=[0, 1.])

    Plots a spike raster plot for a list of arrays of spike times.

    Input:
      spiketimes  list of arrays of spike times
      y_range     2-tuple that holds the y-values that the raster ticks
                  should be drawn between
    '''
    N = len(st)  # number of neurons
    
    if y_range is None:
        y_range = [0, N-1]

    #loc = np.linspace(y_range[0], y_range[1], N)
    loc = np.linspace(0, N-1, N)
    if N==1:
        bin_radius = 0.5
    else:
        bin_radius = ( loc[1] - loc[0] ) / 2.
    ctr = []
    for n in range(N):
        nspikes = len(st[n])
        y = [ [loc[n]-bin_radius]*nspikes , [loc[n]+bin_radius]*nspikes ]
        plt.plot(np.vstack((st[n],st[n])), y, color=np.random.rand(3))
    plt.ylim([y_range[0]-bin_radius, y_range[-1]+bin_radius])
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron index')
    return


def spikes_between(spiketrain, t_start, t_end):
    '''
     numspikes = spikes_between(spiketrain, t_start, t_end)

     Returns the number of times between t_start and t_end.
     Specifically, it counts a spike if it occurred at t, where
     t_start <= t < t_end
    
     Inputs:
       spiketrain   array-like list of spike times
       t_start      start time
       t_end        end time
       
     Output:
       numspikes    number of spikes, where t_start <= t < t_end
    '''
    sp_bool = np.logical_and( np.array(spiketrain)>=t_start, np.array(spiketrain)<t_end )
    return np.sum(sp_bool)


def random_array(n):
    '''
     w = random_array(n)
     
     Returns an array of n random numbers between 0 and 1.
     Note: The values are smoothed.
    '''
    a = np.random.random(size=(n,))
    b = np.zeros_like(a)
    omega = max(min(int(n/3.), 7), 1)
    b[:omega] = 1.
    b /= np.sum(b)
    return np.real(np.fft.ifft(np.fft.fft(a)*np.fft.fft(b)))



#================================
#
#  SpikingNeuron base class
#
#================================
class SpikingNeuron():
    '''
     sn = SpikingNeuron(label=None)

     This is the base class for spiking neuron models, though it can
     be instantiated as-is (it just doesn't do anything).

     Neuron models derived from this class are governed by differential
     equations. Each derived class should implement:
     - slope, which evaluates the rate of change of the dynamic variables, and
     - step, which uses those slopes to take a time step.
    '''
    def __init__(self, label=None):
        self.t = 0.
        self.label = label
        self.spikes = [] # list or array of times when the neuron spiked

        self.axon = []    # list of outgoing connections
        # Each connection is a 2-tuple of the form (post,w), where "post" is the
        # receiving neuron, and w is the connection weight.

        # The incoming_spikes_buffer tabulates all the spikes that arrive in a time step.
        # Each incoming spike should be weighted by the corresponding connection weight.
        # This buffer simply adds together all the incoming spikes for one time step.
        self.incoming_spikes_buffer = 0.  # weighted sum of incoming spikes (for one time step)

    def get_spikes(self):
        '''
         spikes = sn.get_spikes()
         Returns an array of spike times.
        '''
        return self.spikes
    def add_spike(self, t):
        '''
        sn.add_spike(t)
        Adds a spike at time t to the neuron's spike train.
        '''
        self.spikes.append(t)
    
    def connect_to(self, post, w):
        '''
         sn.connect_to(post, w)
         
         Connect to another neuron with weight w.
         
         Inputs:
          post  neuron to receive this neuron's spikes
          w     weight of the connection
          
          Usage:
          To connect neuron sn to neuron B with weight 0.1
          > sn.connect_to(B, 0.1)
        '''
        self.axon.append((post, w))

    def slope(self):
        '''
         sn.slope()
        
         Evaluates the right-hand side of the differential equations that
         govern the internal state of the neuron.
         
         There is no output; the method simply updates member variables.
        '''
        return
    
    def step(self, dt):
        '''
        sn.step(dt)

        Takes a step in time, updating the dynamic variables.
        '''
        return

    def send_spike(self, n=1):
        '''
         sn.send_spike(n=1)
         
         Sends n spike(s) to all connected-to neurons.
        '''
        # Send spike out to each connected neuron
        for p,w in self.axon:
            p.incoming_spikes_buffer += n*w

    def get_input_buffer(self):
        '''
        in_current = sn.get_input_buffer()
        
        Returns the input current that resulted from incoming spikes over
        the last time step.
        '''
        return self.incoming_spikes_buffer
    
    def set_input_buffer(self, c):
        '''
        sn.set_input_buffer(c)
        Resets the neuron's input spike buffer to c.
        '''
        self.incoming_spikes_buffer = c


#================================
#
#  InputNeuron class (based on SpikingNeuron)
#
#================================
class InputNeuron(SpikingNeuron):
    '''
     A = InputNeuron(spiketrain, label=None)

     Constructor for InputNeuron class.

     InputNeuron is a class of neuron that can be used to inject spikes into
     the network. When involved in a simulation, an InputNeuron will generate
     spikes at the times specified during its construction.

     Inputs:
       spiketrain  array or list of spike times
       label       optional string

     Usage:
     > A = InputNeuron( generate_spike_train([[0,10]]), label='Input')
    '''    
    def __init__(self, spiketrain, label=None):
        super().__init__(label=label)
        self.spikes = np.array(spiketrain)  # 1D array of spike times
        
    def slope(self):
        # Do nothing. This neuron model is not governed by a DE
        return
    
    def step(self, dt):
        # How many spikes are in this time step?
        n_spikes = spikes_between(self.spikes, self.t, self.t+dt)
        self.t += dt
        if n_spikes>0:
            self.send_spike(n_spikes)


#================================
#
#  SpikingNetwork class
#
#================================
class SpikingNetwork():
    
    def __init__(self):
        '''
        net = SpikingNetwork()
        
        Constructor for SpikingNetwork class.
        
        The SpikingNetwork class contains a collection of neurons.
        It also has the code to simulate the network and display
        the resulting spike raster plots.
        '''
        self.neur = []         # List of neurons (of various kinds)
        self.t_history = []    # List of time stamps for the Euler steps
                               # (Useful for plotting)
            
    def add_neuron(self, neur):
        '''
        net.add_neuron(neuron)
        
        Adds a neuron to the network.
        
        Input:
         neuron is an object of type SpikingNeuron
        '''
        self.neur.append(neur)
                
    def simulate(self, T, dt):
        '''
        net.simulate(T, dt)
        
        Simulates the network for T seconds by taking Euler steps
        of size dt.
        
        Inputs:
         T    how long to integrate for
         dt   time step for Euler's method
         
        The results of the simulation are stored in the individual neuron
        histories.

        NOTE: Multiple calls to simulate will extend the simulation, not
              restart it. To restart a simulation, you should rebuild
              the network.
        '''
        
        current = 0 if len(self.t_history)==0 else self.t_history[-1]
        t_segment = np.arange(current, current+T, dt)
        
        for tt in t_segment:
            self.t_history.append(tt)
            
            # Compute slopes for all neurons first...
            for neur in self.neur:
                neur.slope()
                
            # ... then update the neurons using an Euler step.
            for neur in self.neur:
                neur.step(dt)
                
    def all_spike_times(self):
        '''
        all_spikes = net.all_spike_times()
        
        Returns all the spikes of all the neurons in the network.
        Useful for making spike-raster plots of network activity.
        
        Output:
         all_spikes  a list of sublists, where each sublist holds
                     the spike times of one of the neurons
        '''
        all_spikes = []
        for neur in self.neur:
            all_spikes.append(np.array(neur.get_spikes()))
        return all_spikes
    
    def spike_raster(self):
        '''
         net.spike_raster()
         
         Generate a spike raster plot.
        '''
        plot_spike_raster(self.all_spike_times())
        xlim = plt.axis()
        x = xlim[1]*1.01
        for idx,n in enumerate(self.neur):
            if n.label is not None:
                plt.text(x, idx, n.label);


