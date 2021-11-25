import numpy as np
from matplotlib import pyplot as plt


def scatter(x, y):
    fig1 = plt.figure()  # create a figure with the default size
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_aspect('equal')
    ax1.scatter(x, y, marker='x', )
    plt.axis('off')


def make_non_mandelbrot_set(nsamples: int, max_iterations: int) -> np.ndarray:
    """Generate a set of complex numbers that are not in the Mandelbrot set.
    This employs some of the optimizations from this page,
    http://en.wikipedia.org/wiki/Mandelbrot_set#Optimizations
    In order to minimize run time, we are trying to reduce the number of points
    that we already know that are in the mandelbrot set. Points inside the within the
    cardioid and in the period-2 bulb can be eliminiated.
    Args:
        nsamples (int): Number of samples to generate.
        max_iterations (int): Maximum number of iterations to perform.
    Returns:
        np.ndarray: Array of complex numbers that are not in the Mandelbrot set.
    """

    non_mandels = np.zeros(nsamples, dtype=np.complex128)
    n_non_mandels = 0

    cardioid = (
            np.random.random(nsamples) * 4 - 2 + (np.random.random(nsamples) * 4 - 2) * 1j
    )

    p = (((cardioid.real - 0.25) ** 2) + (cardioid.imag ** 2)) ** 0.5

    cardioid = cardioid[cardioid.real > p - (2 * p ** 2) + 0.25]

    cardioid = cardioid[((cardioid.real + 1) ** 2) + (cardioid.imag ** 2) > 0.0625]

    z = np.copy(cardioid)

    for _ in range(max_iterations):
        z = z ** 2 + cardioid

        mask = np.abs(z) < 2
        new_non_msets = cardioid[~mask]
        non_mandels[n_non_mandels: n_non_mandels + len(new_non_msets)] = new_non_msets
        n_non_mandels += len(new_non_msets)

        cardioid = cardioid[mask]
        z = z[mask]
    i_set = non_mandels[:n_non_mandels]
    vect_set = np.stack([i_set.real, i_set.imag], axis=1)
    return np.expand_dims(vect_set, axis=1)


c = make_non_mandelbrot_set(100, 1000)
print(c)
print(c.shape)
scatter(c.real, c.imag)
plt.show()
