# %%
import taichi as ti
import numpy as np
from matplotlib import pyplot as plt

ti.init(arch=ti.gpu)

xmin, xmax = -2, 2
ymin, ymax = -1.5, 1.5
mins = np.array([xmin, ymin])
maxs = np.array([xmax, ymax])
n = 1024
samples_count = n * n
xn = 2 * n
yn = n
pixels = ti.field(dtype=ti.f32, shape=(xn, yn))
samples = ti.Vector.field(2, dtype=ti.f64, shape=(samples_count, 1))


# %%
def to_image(pixels):
    im = pixels.to_numpy()
    mean = np.mean(im)
    std = np.std(im)
    normalized = im / (mean + 10 * std)
    normalized[normalized > 1] = 1
    normalized[normalized < 0] = 0
    return normalized


def plot(im):
    fig1 = plt.figure()  # create a figure with the default size
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_aspect('equal')
    ax1.imshow(im)
    plt.axis('off')


# %%

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])


@ti.kernel
def paint(max_iter: int):
    for i, j in samples:  # Parallelized over all pixels
        c = samples[i, j]
        z = samples[i, j]

        p = (((c[0] - 0.25) ** 2) + (c[1] ** 2)) ** 0.5
        is_not_cardoid = c[0] > p - (2 * p ** 2) + 0.25
        is_not_circle = (c[0] + 1) ** 2 + (c[1] ** 2) > 0.0625

        iterations = 0
        while is_not_cardoid and is_not_circle and z.norm() < 2 and iterations < max_iter:
            z = complex_sqr(z) + c
            iterations += 1
        if z.norm() >= 2:
            c = samples[i, j]
            z = samples[i, j]
            while z.norm() < 2:
                z = complex_sqr(z) + c
                ic = int(xn * (z[0] - xmin) / (xmax - xmin))
                jc = int(yn * (z[1] - ymin) / (ymax - ymin))
                pixels[ic, jc] += 1.0


def new_samples():
    np_samples = mins + (maxs - mins) * np.random.random((samples_count, 1, 2))
    samples.from_numpy(np_samples)


repeat = 10000
max_iter = 100
for i in range(0, repeat):
    print(i)
    new_samples()
    paint(max_iter)

im = to_image(pixels)
np.save('bbrod/bbrod_x{:d}_iter{:d}_repeat{:d}.npy'.format(n, max_iter, repeat), pixels.to_numpy())
# rgb = np.stack([im100, im1000 * (1 - im100), im10000 * (1 - im1000) * (1-im100)], axis=2)
plot(im)
plt.savefig('b.png', dpi=1000)
