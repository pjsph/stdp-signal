import numpy as np

c1 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1]
c2 = [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1]

draw = [('c1', c1), ('c2', c2)]

def get_random_signal():
    rand = np.random.randint(0, len(draw))
    return (draw[rand][0], signal_to_input(draw[rand][1], 0.05))


def signal_to_input(sig, noise_level, h = 1):
    signal = [h * digit + np.random.normal(0, noise_level) for digit in sig]
    signal = [round(min(max(digit, 0), 1) * 255) for digit in signal]
    return signal


def get_labels():
    return [d[0] for d in draw]

