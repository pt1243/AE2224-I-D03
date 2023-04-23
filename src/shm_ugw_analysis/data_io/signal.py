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


def signal_collection(
        cycles: Iterable[str],
        signal_types: Iterable[str],
        emitters: Iterable[int],
        receivers: Iterable[int],
        frequencies: Iterable[int]
) -> Iterator[Signal]:
    if not set(cycles).issubset(allowed_cycles):
        raise ValueError(f'invalid cycle in {cycles}, must be in {allowed_cycles}')
    if not set(signal_types).issubset(allowed_signal_types):
        raise ValueError(f'invalid signal type in {signal_types}, must be in {allowed_signal_types}')
    if not set(emitters).issubset(allowed_emitters):
        raise ValueError(f'invalid emitter in {emitters}, must be in {allowed_emitters}')
    if not set(receivers).issubset(allowed_receivers):
        raise ValueError(f'invalid receiver in {receivers}, must be in {allowed_receivers}')
    if not set(frequencies).issubset(allowed_frequencies):
        raise ValueError(f'invalid frequency in {frequencies}, must be in {allowed_frequencies}')

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
        paths: Optional[Iterable[tuple[int, int]]],
        frequency: int,
) -> Iterator[Signal]:
    if paths is None:
        return