from scipy.io import loadmat
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

#QUIZ DESARROLLADO EN AMBIENTE LOCAL, POR ESO NO TIENE LAS LIBRERIAS DE google.colab

def _bilinear_interpolate_gray(img, x, y, fill_value=0.0):
    h, w = img.shape
    # vecinos enteros
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    mask = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)

    out = np.full(x.shape, fill_value, dtype=np.float64)
    if not np.any(mask):
        return out

    xf = x[mask] - x0[mask]
    yf = y[mask] - y0[mask]

    Ia = img[y0[mask], x0[mask]].astype(np.float64)
    Ib = img[y1[mask], x0[mask]].astype(np.float64)
    Ic = img[y0[mask], x1[mask]].astype(np.float64)
    Id = img[y1[mask], x1[mask]].astype(np.float64)

    wa = (1 - xf) * (1 - yf)
    wb = (1 - xf) * yf
    wc = xf * (1 - yf)
    wd = xf * yf

    out[mask] = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out

def rotate_image(img, angle_deg, center=None, scale=1.0, fill_value=0.0):
    h, w = img.shape
    if center is None:
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    else:
        cx, cy = float(center[0]), float(center[1])

    theta = np.deg2rad(angle_deg)
    alpha = scale * np.cos(theta)
    beta  = scale * np.sin(theta)

    xx, yy = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

    x_src = alpha * xx + beta * yy + ((1 - alpha) * cx - beta * cy)
    y_src = -beta * xx + alpha * yy + (beta * cx + (1 - alpha) * cy)

    return _bilinear_interpolate_gray(img, x_src, y_src, fill_value=fill_value)

def scale_image_gray(img, sx=1.0, sy=1.0, fill_value=0.0):
    h, w = img.shape
    new_w = max(1, int(round(w * sx)))
    new_h = max(1, int(round(h * sy)))

    xx, yy = np.meshgrid(np.arange(new_w), np.arange(new_h))
    x_src = xx / sx
    y_src = yy / sy

    return _bilinear_interpolate_gray(img, x_src, y_src, fill_value=fill_value)

data = loadmat('Quiz_1_PDI.mat')
image = data['Imagen']
if image.ndim > 2:
    image = np.squeeze(image)

print(f'El tipo de datos de la imagen es: {image.dtype}')

# normalizar a float64 para procesar estable
image_f = image.astype(np.float64)

h, w = image_f.shape[:2]
center = (w // 2, h // 2)

angle = 105
rotated = rotate_image(image_f, angle_deg=angle, center=((w - 1)/2.0, (h - 1)/2.0))

# escalado a 0-255 tras la rotación
min_val = rotated.min()
max_val = rotated.max()
if max_val > min_val:
    rotated_scaled = 255.0 * (rotated - min_val) / (max_val - min_val)
else:
    rotated_scaled = np.zeros_like(rotated)
rotated_scaled = rotated_scaled.astype(np.float64)

print("Imagen rotada y escalada:", rotated_scaled.shape)

x = 0
y = 426
width = 1920
height = 1356
roi = rotated_scaled[y:y+height, x:x+width].copy()

scale_y = 0.9975
scale_x = 1.06257
scaled = scale_image_gray(roi, sx=scale_x, sy=scale_y, fill_value=0.0)

cv2.imshow('Original', image.astype(np.uint8) if image.dtype != np.uint8 else image)
cv2.imshow('Rotada (esc 0-255)', np.clip(rotated_scaled, 0, 255).astype(np.uint8))
cv2.imshow('Region de interés (ROI), escalada', np.clip(scaled, 0, 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Imagen Escalada:", scaled.shape)

roi_flat = scaled.flatten()
plt.hist(roi_flat, bins=255, edgecolor='black')
plt.xlabel("Intervalo")
plt.ylabel("Frecuencia")
plt.title("Histograma por intervalos de color (Escalada)")
plt.show()

scaled_u8 = np.clip(scaled, 0, 255).astype(np.uint8)
equalized_img = cv2.equalizeHist(scaled_u8)

cv2.imshow('Equalizada', equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Imagen Ecualizada:", equalized_img.shape)

equa = equalized_img.flatten()
plt.hist(equa, bins=255, edgecolor='black')
plt.xlabel("Intervalo")
plt.ylabel("Frecuencia")
plt.title("Histograma por intervalos de color (Equalizada)")
plt.show()
