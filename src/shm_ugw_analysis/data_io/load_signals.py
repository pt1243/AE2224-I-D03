import pathlib
from itertools import product
from typing import Iterable, Iterator, Optional, Final

import numpy as np

from .paths import NPZ_DIR


allowed_cycles: Final = ('AI', '0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000', 'Healthy')
allowed_signal_types: Final = ('excitation', 'received')
allowed_emitters: Final = (1, 2, 3, 4, 5, 6)
allowed_receivers: Final = allowed_emitters
allowed_frequencies: Final = (100, 120, 140, 160, 180)

_all_paths = []
for i in range(1, 4):
    for j in range(4, 7):
        _all_paths.append((i, j))
        _all_paths.append((j, i))

all_paths: Final = tuple(_all_paths)

class InvalidSignalError(ValueError):
    """Raised when attempting to load a signal """
    pass


class _SignalValidator():
    def __init__(self,
            allowed_cycles: Optional[Iterable[str]],
            allowed_signal_types: Optional[Iterable[str]],
            allowed_emitters: Optional[Iterable[int]],
            allowed_receivers: Optional[Iterable[int]],
            allowed_frequencies: Optional[Iterable[int]]
    ) -> None:
        self.allowed_cycles = set(allowed_cycles)
        self.allowed_signal_types = set(allowed_signal_types)
        self.allowed_emitters = set(allowed_emitters)
        self.allowed_receivers = set(allowed_receivers)
        self.allowed_frequencies = set(allowed_frequencies)

    @classmethod
    def from_defaults(cls):
        return cls(allowed_cycles, allowed_signal_types, allowed_emitters, allowed_receivers, allowed_frequencies)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.allowed_cycles}, {self.allowed_signal_types}, ' \
        f'{self.allowed_emitters}, {self.allowed_receivers}, {self.allowed_frequencies})'

    def validate_emitter_receiver_pair(self, emitter: int, receiver: int) -> None:
        """Validate a receiver and emitter pair.
        
        Raises InvalidSignalError if an invalid combination is found.
        """
        if emitter not in self.allowed_emitters:
            raise InvalidSignalError(f'invalid emitter {emitter}, must be one of {allowed_emitters}')
        if receiver not in self.allowed_receivers:
            raise InvalidSignalError(f'invalid receiver {receiver}, must be one of {allowed_receivers}')
        top, bottom = (1, 2, 3), (4, 5, 6)
        if emitter in top and receiver in top:
            raise InvalidSignalError(f'invalid pairing of emitter {emitter} and receiver {receiver}')
        if emitter in bottom and receiver in bottom:
            raise InvalidSignalError(f'invalid pairing of emitter {emitter} and receiver {receiver}')
        return

    def validate_cycles(self, cycles: Iterable[str]) -> None:
        if not set(cycles).issubset(self.allowed_cycles):
            raise InvalidSignalError(f'invalid cycle in {cycles}, must be in {allowed_cycles}')
        return

    def validate_signal_types(self, signal_types: Iterable[str]) -> None:
        if not set(signal_types).issubset(self.allowed_signal_types):
            raise InvalidSignalError(f'invalid signal type in {signal_types}, must be in {allowed_signal_types}')
        return

    def validate_frequencies(self, frequencies: Iterable[int]) -> None:
        if not set(frequencies).issubset(self.allowed_frequencies):
            raise InvalidSignalError(f'invalid frequency in {frequencies}, must be in {allowed_frequencies}')
        return

    def validate_transducer_numbers(self, transducer_numbers: Iterable[int]) -> None:
        if not set(transducer_numbers).issubset(self.allowed_emitters):
            raise InvalidSignalError(f'invalid transducer number in {transducer_numbers}, must be in {allowed_emitters}')
        return
    
    def validate_all(self, cycles: Iterable[str], signal_types: Iterable[str], frequencies: Iterable[int]) -> None:
        self.validate_cycles(cycles)
        self.validate_signal_types(signal_types)
        self.validate_frequencies(frequencies)


validator = _SignalValidator.from_defaults()


def _load_npz(filename: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open the data from a file."""
    contents = np.load(filename)
    t, x, desc = contents['x'], contents['y'], contents['desc']
    return x, t, desc


def load_data(
        cycle: str, signal_type: str, emitter: int, receiver: int, frequency: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, pathlib.Path]:
    """Loads the data for a given measurement.
    
    Note: most use cases should use signal_collection, frequency_collection, or path_collection for iteration instead.
    """
    validator.validate_all((cycle,), (signal_type,), (frequency,))
    validator.validate_emitter_receiver_pair(emitter, receiver)

    try:
        folder = f'L2_S2_cycle_{int(cycle)}'
    except ValueError:
        folder = f'L2_S2_{cycle}'

    c1_f1 = 'C1' if signal_type == 'excitation' else 'F1'
    freq = str(frequency // 20)
    if cycle == 'Healthy':
        filename = f'{c1_f1}_L2_S2_healthy_emetteur_{emitter}_recepteur_{receiver}_excitation_{freq}_00000.npz'
    else:
        filename = f'{c1_f1}_mono_1_saine_emetteur_{emitter}_recepteur_{receiver}_excitation_{freq}_00000.npz'

    full_path = NPZ_DIR.joinpath(folder, filename)

    x: np.ndarray
    t: np.ndarray
    desc: np.ndarray

    x, t, desc = _load_npz(full_path)
    return x, t, desc, folder, filename, full_path


class Signal:
    def __init__(self, cycle: str, signal_type: str, emitter: int, receiver: int, frequency: int) -> None:
        self._cycle: str = cycle
        self._signal_type: str = signal_type
        self._emitter: int = emitter
        self._receiver: int = receiver
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

        return

    def __repr__(self) -> str:
        return f"Signal('{self._cycle}', '{self._signal_type}', {self._emitter}, {self._receiver}, {self._frequency})"

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
        return self._receiver

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
        """Returns x and t in one array, of shape (number of samples, 2).

        Equivalent indexing:

        t = data[:, 0]
        x = data[:, 1]

        t[i] = data[i, 0]
        x[i] = data[i, 1]

        data[i, :] = [t[i], x[i]]
        """
        return np.vstack((self._t, self._x.T)).T


def signal_collection(
        cycles: Iterable[str],
        signal_types: Iterable[str],
        emitters: Iterable[int],
        receivers: Iterable[int],
        frequencies: Iterable[int]
) -> Iterator[Signal]:
    """Iterate over all signals that are the product of the arguments."""
    validator.validate_signal_types(signal_types)
    validator.validate_cycles(cycles)
    validator.validate_transducer_numbers(emitters)
    validator.validate_transducer_numbers(receivers)
    validator.validate_frequencies(frequencies)

    for cycle, signal_type, frequency in product(cycles, signal_types, frequencies):
        for emitter in emitters:
            if emitter in (1, 2, 3):
                for receiver in receivers:
                    if receiver in (4, 5, 6):
                        yield Signal(cycle, signal_type, emitter, receiver, frequency)
            else:
                for receiver in receivers:
                    if receiver in (1, 2, 3):
                        yield Signal(cycle, signal_type, emitter, receiver, frequency)


def frequency_collection(
        cycles: Iterable[str],
        signal_types: Iterable[str],
        frequency: int,
        paths: Optional[Iterable[tuple[int, int]]],
) -> Iterator[Signal]:
    """Iterate over all signal paths for a given frequency.
    
    Paths should be a list of emitter/receiver tuples, eg. [(1, 4), (1, 5), (1, 6)], or left as none to use all paths
    by default.
    """
    validator.validate_cycles(cycles)
    validator.validate_signal_types(signal_types)
    validator.validate_frequencies((frequency))
    if paths is not None:
        for (emitter, receiver) in paths:
            validator.validate_emitter_receiver_pair(emitter, receiver)
    else:
        paths = all_paths

    for cycle, signal_type in product(cycles, signal_types):
        for path in paths:
            yield Signal(cycle, signal_type, *path, frequency)


def path_collection(
        cycles: Iterable[str],
        signal_types: Iterable[str],
        path: tuple[int, int],
        frequencies: Iterable[int],
) -> Iterator[Signal]:
    """Iterate over all frequencies for a given signal path.
    
    The signal path is given as a tuple of the emitter and receiver, eg. (1, 4).
    """
    validator.validate_cycles(cycles)
    validator.validate_signal_types(signal_types)
    validator.validate_emitter_receiver_pair(*path)
    validator.validate_frequencies(frequencies)

    for cycle, signal_type, frequency in product(cycles, signal_types, frequencies):
        yield Signal(cycle, signal_type, *path, frequency)
