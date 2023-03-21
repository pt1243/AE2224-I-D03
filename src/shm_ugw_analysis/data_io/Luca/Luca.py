from shm_ugw_analysis.data_io.load import load_data
import matplotlib.pyplot as plt

#x_1, t_1, desc_1 = load_data(cycle='0',    signal_type='received',    emitter=1,    receiver=4,    frequency=100)


def plot(cycle: str, signal_type: str, emitter: int, receiver: int, frequency: int, x_bounds: list, y_bounds: list):
    y, t, desc = load_data(cycle, signal_type, emitter, receiver, frequency)
    label = "cycle_" + str(cycle) + "-" + str(signal_type) + "-emitter_" + str(emitter) + "-receiver_" + str(receiver) + "-frequency_" + str(frequency)
    plt.plot(t, y, label=label)
    plt.xlim(x_bounds[0], x_bounds[1])
    plt.ylim(y_bounds[0], y_bounds[1])
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('test1.png')

x_bounds = [0,0.00002] #x domain shown
y_bounds = [-10,10] #y domain shown

def r_vs_em(cycle: str, emitter: int, receiver:int, frequency: int, x_bounds: list, y_bounds: list):
    plot(cycle, 'excitation', emitter, receiver, frequency, x_bounds, y_bounds)
    plot(cycle, 'received', emitter, receiver, frequency, x_bounds, y_bounds)
    


r_vs_em('70000', 4, 2, 100, x_bounds, y_bounds)
