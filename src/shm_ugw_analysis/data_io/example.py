# We start by importing the relevant functions:
from shm_ugw_analysis.data_io.load_signals import Signal, signal_collection

# To access just one specific signal, create an instance of the Signal class. The format is as follows:
# s = Signal(cycle, signal_type, emitter, receiver, frequency)
# So an example of the above would be:

s = Signal('1000', 'excitation', 1, 4, 120)

# This accesses the excitation signal at cycle 1000, from emitter 1 to receiver 4, at 120 kHz. You can try to change
# these numbers and see how it changes. The acceptable values are:

# allowed_cycles = ('AI', '0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000', 'Healthy')
# allowed_signal_types = ('excitation', 'received')
# allowed_emitters = (1, 2, 3, 4, 5, 6)
# allowed_receivers = (1, 2, 3, 4, 5, 6)
# allowed_frequencies = (100, 120, 140, 160, 180)

# Additionally, as we are only considering the paths that go through the damage location, only signals that start at
# one of sensors 1, 2, or 3 and end at 4, 5, or 6 are valid, and vice versa.

# We can then access attributes of the signal as follows. This means that you can pull out only what information you're
# looking for. Some examples are:

s.cycle             # '1000'
s.signal_type       # 'excitation'
s.emitter           # 1
s.receiver          # 4
s.frequency         # 120

# You can also get the oscilloscope sampling time interval and sampling frequency:

s.sample_interval   # 4e-8
s.sample_frequency  # 25000000

# To access the data, use:

s.t                 # t, the time data
s.x                 # x(t), the oscilloscope readings

# Note that these are not editable, to prevent you from accidentally modifying the data and unknowingly reusing that
# modified data later on. If you want to be editing the data, you can get a copy instead. Do note that this is slower
# and duplicates the entire array in memory, so you don't want to do this unless you have to.

s.copy_t()          # editable copy of t
s.copy_x()          # editable copy of x(t)

# Using this method is good for one signal, but once you are trying to iterate over multiple signals you should instead
# use a 'signal collection'. This is a collection of all the signals that satisfy some criteria. For instance, if you
# want all the signals from cycles 0 and cycles 1 that are excitation signals and go from sensor 1 to either sensor 4,
# 5, or 6, and have a frequency of either 100 or 120 kHz, you can do the following:

sc = signal_collection(
    cycles=['0', '1'],
    signal_types=['excitation'],
    emitters=[1],
    receivers=[4, 5, 6],
    frequencies=[100, 120]
)

# You can then iterate over each signal and access the data from each signal using the methods described above:

for s in sc:
    # your data processing here; for example:
    print(s)

# The above code gives:
# Signal('0', 'excitation', 1, 4, 100)
# Signal('0', 'excitation', 1, 5, 100)
# Signal('0', 'excitation', 1, 6, 100)
# Signal('0', 'excitation', 1, 4, 120)
# Signal('0', 'excitation', 1, 5, 120)
# Signal('0', 'excitation', 1, 6, 120)
# Signal('1', 'excitation', 1, 4, 100)
# Signal('1', 'excitation', 1, 5, 100)
# Signal('1', 'excitation', 1, 6, 100)
# Signal('1', 'excitation', 1, 4, 120)
# Signal('1', 'excitation', 1, 5, 120)
# Signal('1', 'excitation', 1, 6, 120)
