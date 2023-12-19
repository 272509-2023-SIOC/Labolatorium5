# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Importowanie niezbędnych bibliotek
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

# Ładowanie obrazu z filtru Bayera
image = np.load("C:/Users/Szymon Nowicki/Desktop/CFA/Bayer/namib.jpg")
image_conv = np.copy(image)

# Podstawowe informacje o obrazie
image.shape
image[:4, :4, 1]
image.dtype

# Wyświetlanie oryginalnego obrazu
io.imshow(image)

# Ekstrakcja poszczególnych kanałów kolorów
green = image[:, :, 1]
red = image[:, :, 0]
blue = image[:, :, 2]

# Wyświetlanie kanału zielonego
io.imshow(green)

# Przygotowanie do interpolacji
np.arange(10)[1::2]
to_interp = green
to_interp.shape
np.sum(green)
np.sum(to_interp)
to_interp = green[::2, :]
to_interp = to_interp[:, ::2]
io.imshow(to_interp)

# Definicja funkcji jądra liniowego do interpolacji
def linear_kernel(x, offset: float, width: float):
    return (1 - np.abs((x - offset) / width)) * (np.abs((x - offset) / width) < 1)

# Tworzenie i wyświetlanie jądra interpolacji
x = np.linspace(-3, 3, 1000)
y = linear_kernel(x, offset=0.0, width=1)
plt.plot(x, y)

# Interpolacja wierszy
def interpolate_row(row):
    kernels = []
    space = np.linspace(0, 1, 2 * len(row))
    for x, y in zip(space.tolist(), row.tolist()):
        kernel = linear_kernel(space, offset=2 * x, width=1 / len(row))
        kernels.append(y * kernel)
    return space, np.sum(np.asarray(kernels), axis=0)

# Interpolacja kanału zielonego
iterpolated = []
for row in to_interp:
    _, i = interpolate_row(row)
    iterpolated.append(i)
iterpolated = np.asarray(iterpolated)
io.imshow(iterpolated)

# Interpolacja kolumn
iterpolated2 = []
for column in iterpolated.T:
    _, i = interpolate_row(column)
    iterpolated2.append(i)
iterpolated2 = np.asarray(iterpolated2).T
io.imshow(iterpolated2)
result_green = iterpolated2

# Interpolacja kanałów czerwonego i niebieskiego
red[::2].shape

red_row_inter = []
for row in red[::2]:
    _, i = interpolate_row(row[1::2])
    red_row_inter.append(i)

red_row_inter = np.asarray(red_row_inter)
red_row_inter.shape

red_col_inter = []
for col in red_row_inter.T:
    # print(col)
    # break
    _, i = interpolate_row(col)
    red_col_inter.append(i)

red_col_inter = np.asarray(red_col_inter)
red_col_inter.T.shape

result_red = red_col_inter

blue_row_inter = []
for row in blue[1::2]:
    _, i = interpolate_row(row[::2])
    blue_row_inter.append(i)

blue_row_inter = np.asarray(blue_row_inter)
blue_row_inter.shape

blue_col_inter = []
for col in blue_row_inter.T:
    # print(col)
    # break
    _, i = interpolate_row(col)
    blue_col_inter.append(i)

blue_col_inter = np.asarray(blue_col_inter)
blue_col_inter.T.shape

result_blue = blue_col_inter

result_red.shape, result_green.T.shape, result_blue.shape
# Analogicznie do kanału zielonego, ale dla innych wzorów na filtrze Bayera

# Łączenie zinterpolowanych kanałów w jeden obraz
result_image_interpolated = np.dstack([result_red.T, result_green, result_blue.T])
io.imshow(result_image_interpolated)

# Przygotowanie kanałów do konwolucji
red_conv = image_conv[:, :, 0] * (np.zeros_like(image_conv[:,:,0])) + (np.zeros_like(image_conv[:,:,0][::2, 1::2] == 1))
green_conv = image_conv[:, :, 1] * (np.zeros_like(image_conv[:,:,1]) + (np.zeros_like(image_conv[:,:,1])[::2, ::2] == 1) + (np.zeros_like(image_conv[:,:,1])[1::2, 1::2] == 1))
blue_conv = image_conv[:, :, 2] * (np.zeros_like(image_conv[:,:,2]) + (np.zeros_like(image_conv[:,:,2])[1::2, ::2] == 1))

# Tworzenie masek dla każdego kanału
kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4

# Konwolucja dla każdego kanału z użyciem określonego jądra
red_conv = convolve2d(red_conv, kernel, mode='same', boundary='symm')
green_conv = convolve2d(green_conv, kernel, mode='same', boundary='symm')
blue_conv = convolve2d(blue_conv, kernel, mode='same', boundary='symm')

# Łączenie skonwoluowanych kanałów w jeden obraz
result_image_convolved = np.dstack([red_conv, green_conv, blue_conv])

# Wyświetlanie obu wyników dla porównania
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Interpolacja")
io.imshow(result_image_interpolated)
plt.subplot(1, 2, 2)
plt.title("Konwolucja")
io.imshow(result_image_convolved)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
