import pathlib

import numpy as np
import math

from shm_ugw_analysis.data_io.load import load_data


class Signal:

    allowed_cycles = ['AI', '0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000', 'Healthy']
    allowed_signal_types = ['excitation', 'received']
    allowed_emitters = [i for i in range(1, 7)]
    allowed_receivers = allowed_emitters
    allowed_frequencies = [100, 120, 140, 160, 180]

    def __init__(self, cycle: str, signal_type: str, emitter: int, receiver: int, frequency: int) -> None:
        self._cycle: str = cycle
        self._signal_type: str = signal_type
        self._emitter: int = emitter
        self._receier: int = receiver
        self._frequency: int = frequency

        self._x: np.ndarray
        self._t: np.ndarray
        self._desc: np.ndarray
        self._folder: str
        self._filename: str
        self._full_path: pathlib.Path

        self._x, self._t, self._desc, self._folder, self._filename, self._full_path = load_data(
            cycle, signal_type, emitter, receiver, frequency
        )

        self._x.setflags(write=False)
        self._t.setflags(write=False)
        self._desc.setflags(write=False)

        self._sample_interval: float = np.around(self._desc[0], decimals=8)
        self._sample_frequency: int = int(np.around(self._desc[1]))
        
        self._length = self._x.size
        print(self._length)
        
        return
    
    @property
    def cycle(self) -> str:
        """Cycle name: one of 'AI', '0', '1', ... , '60000', '70000', 'Healthy'."""
        return self._cycle
    
    @property
    def signal_type(self) -> str:
        """If the signal is the excitation signal or the received signal."""
        return self._signal_type
    
    @property
    def emitter(self) -> int:
        """Number of the emitting PZT."""
        return self._emitter
    
    @property
    def receiver(self) -> int:
        """Number of the receiving PZT."""
        return self._emitter
    
    @property
    def frequency(self) -> int:
        """Signal frequency in [kHz]."""
        return self._frequency

    @property
    def t(self) -> np.ndarray:
        """Time series data, cannot be modified: use '.copy_t()' instead."""
        return self._t

    @property
    def x(self) -> np.ndarray:
        """Oscilloscope readings, cannot be modified: use '.copy_x()' instead."""
        return self._x
    
    @property
    def sample_interval(self) -> float:
        """Time between oscilloscope samples, in [s]."""
        return self._sample_interval
    
    @property
    def sample_frequency(self) -> float:
        """Oscilloscope sample frequency, in [Hz]."""
        return self._sample_frequency
    
    def copy_t(self) -> np.ndarray:
        """Editable copy of the time series data."""
        return np.copy(self._t)
    
    def copy_x(self) -> np.ndarray:
        """Editable copy of the oscilloscope readings."""
        return np.copy(self._x)
    
    def data(self) -> np.ndarray:
        """Returns x and t in one array of shape (125002, 2).
        
        Equivalent indexing:

        t = data[:, 0]
        x = data[:, 1]

        t[i] = data[i, 0]
        x[i] = data[i, 1]

        data[i, :] = [t[i], x[i]]
        """
        return np.vstack((self._t, self._x.T)).T


    
s = Signal('AI', 'received', 1, 6, 100)
