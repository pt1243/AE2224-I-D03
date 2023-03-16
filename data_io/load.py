import numpy as np

from paths import NPZ_DIR


def load_npz(filename):
    """Open the data from a file."""
    contents = np.load(filename)
    x, t, desc = contents['x'], contents['y'], contents['desc']
    return x, t, desc


def load_data(cycle: str, signal_type: str, emitter: int, receiver: int, frequency: int):
    """Loads the data for a given measurement."""

    allowed_cycles = ['AI', '0', '1', '1000', '10000', '20000', '30000', '40000', '50000', '60000', '70000', 'Healthy']
    allowed_signal_types = ['excitation', 'received']
    allowed_emitters = [i for i in range(1, 7)]
    allowed_receivers = allowed_emitters
    allowed_frequencies = [100, 120, 140, 160, 180]

    if cycle not in allowed_cycles:
        raise ValueError(f"Invalid cycle '{cycle}', must be one of {allowed_cycles}")
    if signal_type not in allowed_signal_types:
        raise ValueError(f"Invalid signal type '{signal_type}, must be either 'excitation' or 'received'")
    if emitter not in allowed_emitters:
        raise ValueError(f"Invalid emitter number {emitter}, must be from 1 to 6")
    if receiver not in allowed_receivers:
        raise ValueError(f"Invalid receiver number {receiver}, must be from 1 to 6")
    if emitter in [1, 2, 3]:
        if receiver in [1, 2, 3]:
            raise ValueError(f"Invalid combination of emitter number {emitter} and receiver number {receiver}")
    elif receiver in [4, 5, 6]:
        raise ValueError(f"Invalid combination of emitter number {emitter} and receiver number {receiver}")
    if frequency not in allowed_frequencies:
        raise ValueError(f"Invalid frequency '{frequency}', must be either 100, 120, 140, 160, or 180 [kHz]")
    

    try:
        folder = f'L2_S2_cycle_{int(cycle)}'
    except ValueError:
        folder = f'L2_S2_{cycle}'

    c1_f1 = 'C1' if signal_type == 'excitation' else 'F1'
    freq = str(frequency // 20)
    filename = f'{c1_f1}_mono_1_saine_emetteur_{emitter}_recepteur_{receiver}_excitation_{freq}_00000.npz'

    full_path = NPZ_DIR.joinpath(folder, filename)

    x, t, desc = load_npz(full_path)
    return x, t, desc


"""
Sample usage


x_1, t_1, desc_1 = load_data(
    cycle='0',
    signal_type='received',
    emitter=1,
    receiver=4,
    frequency=100
)
"""
