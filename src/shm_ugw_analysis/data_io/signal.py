import pathlib
from itertools import product
from typing import Iterable, Iterator, Optional

import numpy as np

from shm_ugw_analysis.data_io.load import (
    load_data,
    allowed_cycles,
    allowed_signal_types,
    allowed_emitters,
    allowed_receivers,
    allowed_frequencies
)


class InvalidSignalError(ValueError):
    pass


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
        """Returns x and t in one array, of shape (number of samples, 2).

        Equivalent indexing:

        t = data[:, 0]
        x = data[:, 1]

        t[i] = data[i, 0]
        x[i] = data[i, 1]

        data[i, :] = [t[i], x[i]]
        """
        return np.vstack((self._t, self._x.T)).T


def _validate_emitter_receiver_pair(emitter: int, receiver: int) -> None:
    """Validate a receiver and emitter pair.
    
    Raises InvalidSignalError if an invalid combination is found.
    """
    if emitter not in allowed_emitters:
        raise InvalidSignalError(f'invalid emitter {emitter}, must be one of {allowed_emitters}')
    if receiver not in allowed_receivers:
        raise InvalidSignalError(f'invalid receiver {receiver}, must be one of {allowed_receivers}')
    top, bottom = (1, 2, 3), (4, 5, 6)
    if emitter in top and receiver in top:
        raise InvalidSignalError(f'invalid pairing of emitter {emitter} and receiver {receiver}')
    if emitter in bottom and receiver in bottom:
        raise InvalidSignalError(f'invalid pairing of emitter {emitter} and receiver {receiver}')
    return


def _validate_cycles(cycles: Iterable[str]) -> None:
    if not set(cycles).issubset(allowed_cycles):
        raise InvalidSignalError(f'invalid cycle in {cycles}, must be in {allowed_cycles}')
    return


def _validate_signal_types(signal_types: Iterable[str]) -> None:
    if not set(signal_types).issubset(allowed_signal_types):
        raise InvalidSignalError(f'invalid signal type in {signal_types}, must be in {allowed_signal_types}')
    return


def _validate_frequencies(frequencies: Iterable[int]) -> None:
    if not set(frequencies).issubset(allowed_frequencies):
        raise InvalidSignalError(f'invalid frequency in {frequencies}, must be in {allowed_frequencies}')
    return


def _validate_transducer_numbers(transducer_numbers: Iterable[int]) -> None:
    if not set(transducer_numbers).issubset(allowed_emitters):
        raise InvalidSignalError(f'invalid transducer number in {transducer_numbers}, must be in {allowed_emitters}')
    return


def signal_collection(
        cycles: Iterable[str],
        signal_types: Iterable[str],
        emitters: Iterable[int],
        receivers: Iterable[int],
        frequencies: Iterable[int]
) -> Iterator[Signal]:
    """Iterate over all signals that are the product of the arguments."""
    _validate_cycles(cycles)
    _validate_signal_types(signal_types)
    _validate_transducer_numbers(emitters)
    _validate_transducer_numbers(receivers)
    _validate_frequencies(frequencies)

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


all_paths = []
for i in range(1, 4):
    for j in range(4, 7):
        all_paths.append((i, j))
        all_paths.append((j, i))


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
    _validate_cycles(cycles)
    _validate_signal_types(signal_types)
    _validate_frequencies((frequency))
    if paths is not None:
        for (emitter, receiver) in paths:
            _validate_emitter_receiver_pair(emitter, receiver)
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
    _validate_cycles(cycles)
    _validate_signal_types(signal_types)
    _validate_emitter_receiver_pair(path)
    _validate_frequencies(frequencies)

    for cycle, signal_type, frequency in product(cycles, signal_types, frequencies):
        yield Signal(cycle, signal_type, *path, frequency)
