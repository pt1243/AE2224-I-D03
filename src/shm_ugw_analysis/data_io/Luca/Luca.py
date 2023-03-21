from shm_ugw_analysis.data_io.load import load_data
import matplotlib.pyplot as plt

#x_1, t_1, desc_1 = load_data(cycle='0',    signal_type='received',    emitter=1,    receiver=4,    frequency=100)


def plot(cycle: str, signal_type: str, emitter: int, receiver: int, frequency: int):
    y, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
    label = "cycle_" + str(cycle) + "-" + str(signal_type) + "-emitter_" + str(emitter) + "-receiver_" + str(receiver) + "-frequency_" + str(frequency)
    plt.plot(t, y)
    plt.xlim(0,0.00005)
    plt.ylim(-0.1,0.1)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title(label)
    plt.savefig('test1.png')

plot('1000', 'received', 1, 4, 100)
