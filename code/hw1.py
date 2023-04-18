#reference:https://github.com/JCly-rikiu/HDR
import cv2
import random as rnd
import numpy as np
import math
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import Adaptive_Logarithmic_Mapping as ALM
import gc
import image_alignment as Align

#parameters
#For HDR convert
Z_max, Z_min, Z_mid = 255, 0, 127
ld = 20

#For global tone mapping
key = 0.35
delta = 1e-6
L_white = 1.1
ratio = [0.06, 0.67, 0.27] #BGR

#For local tone mapping
a = 0.18
eps = 0.05
phi = 6.0
s = 0.35
k_size = 5
alpha = 1.6

def hat_weight(z):
    if z <= Z_mid:
        return z - Z_min + 1e-3
    return Z_max + 1e-3 - z

def get_sol(z, i):
    return sol[i][z]

#open file
time = []
imgs = []
p = 0
try:
    while True:
        file = input().split()
        imgs.append(cv2.imread(file[0]))
        time.append(1/float(file[1]))
        p += 1
except EOFError:
    pass
time_log = np.log(time)
print(time_log)

#Alignment
imgs = Align.alignment(np.float32(imgs), int(len(imgs)/2))
imgs = np.array(imgs)
imgs = imgs.astype(np.uint8)
print("Alignment complete")

#sampling 50 pixels
w, h, _ = imgs[0].shape
N = 50
sample_point = []
while len(sample_point) != 50:
    x = rnd.randint(int(w / 8), int(w * 7 / 8))
    y = rnd.randint(int(h / 8), int(h * 7 / 8))
    if (x, y) not in sample_point:
        sample_point.append((x, y))
print(sample_point)

#fill in metrix A and array b
A = np.zeros((3, N * p + 255, 256 + N), dtype=float)
b = np.zeros((3, N * p + 255), dtype=float)
sol = []
for color in range(3):
    for i in range(N * p + 255):
        if i <= (N * p - 1):
            P = i // N
            n = i % N
            Z_ij = imgs[P][sample_point[n][0]][sample_point[n][1]][color]
            W_zij = hat_weight(Z_ij)
            A[color][i][Z_ij] = W_zij
            A[color][i][n + 256] = -W_zij
            b[color][i] = time_log[P] * W_zij
        elif i != N * p:
            ld_power2 = ld ** 2
            w_z = hat_weight(i - (N * p + 1) + 1)
            A[color][i][i - (N * p + 1)] = ld_power2 * w_z
            A[color][i][i - (N * p + 1) + 1] = -2 * ld_power2 * w_z
            A[color][i][i - (N * p + 1) + 2] = ld_power2 * w_z
        else:
            A[color][i][Z_mid] = 1
    #least square solution
    x, res, rank, s  = linalg.lstsq(A[color], b[color], rcond=None)
    sol.append(x)
    for n_ in range(256):
        plt.plot(x[n_], n_, 'o', label='solution', markersize=5)
    #plt.show()
    plt.savefig("../curve/Curve_{:d}.png".format(color))
    plt.clf()
sol = np.array(sol)

del A
del b
gc.collect()

#generate HDR
hdr_img = np.zeros((w, h, 3), dtype=float)
v_func_w = np.vectorize(hat_weight)
#sum_w = np.sum(w_z, axis=0)     #(w, h, color)
v_func_sol = np.vectorize(get_sol)
sum_E = np.zeros((w, h, 3))
sum_w = np.zeros((w, h, 3))
for i in range(p):
    time = np.zeros((w, h, 3))
    time[:, :, :] = time_log[i]
    g_z = np.zeros((w, h, 3))
    w_z = v_func_w(imgs[i])
    for j in range(3):
        g_z[:, :, j] = v_func_sol(imgs[i, :, :, j], j)
    sum_E += w_z * (g_z - time)
    sum_w += w_z
hdr_img[:, :, :] = np.exp(sum_E / sum_w)
'''
time = np.zeros((p, w, h, 3))   #(p, w, h, color)
g_z = np.zeros((p, w, h, 3))    #(p, w, h, color)
w_z = v_func_w(imgs)            #(p, w, h, color)
for i in range(p):
    time[i, :, :, :] = time_log[i]
for i in range(3):
    g_z[:, :, :, i] = v_func_sol(imgs[:, :, :, i], i)
sum_E = np.sum(w_z * (g_z - time), axis=0)  #(w, h, color)
hdr_img[:, :, :] = np.exp(sum_E / sum_w)'''

cv2.imwrite("../data/HDR_img.hdr", hdr_img.astype(np.float32))
print("Write HDR complete")

del w_z
del sum_w
del g_z
del sum_E
gc.collect()

#Tone mapping(photographic reproduction/global)
L_w = np.zeros((w, h))
L_w[:, :] = ratio[0] * hdr_img[:, :, 0] + ratio[1] * hdr_img[:, :, 1] + ratio[2] * hdr_img[:, :, 2]
L_log = np.log((L_w + delta))
s = np.sum(L_log) / (w * h)
L_w_bar = np.exp(s)
L_m = key  * L_w / L_w_bar
L_d = L_m * (1 + L_m / (L_white ** 2)) / (1 + L_m)
L = L_d / L_w
Ldr_img  = np.copy(hdr_img)
Ldr_img_B = np.zeros((w, h, 3))
Ldr_img_G = np.zeros((w, h, 3))
Ldr_img_R = np.zeros((w, h, 3))

del L_log
del L_w_bar
del L_d
gc.collect()

for k in range(3):
    Ldr_img[:, : , k] = Ldr_img[:, :, k] * L[:, :] * 225
Ldr_img = np.clip(Ldr_img, 0, 255)
cv2.imwrite("../data/Ldr_global.png", Ldr_img.astype(np.uint8))
print("tonemapping photographic global complete")

#for test
'''test = cv2.imread("./test.jpg")
test_img_B = np.zeros((w, h, 3))
test_img_G = np.zeros((w, h, 3))
test_img_R = np.zeros((w, h, 3))
test_img_B[:, :, 0] = test[:, :, 0]
test_img_G[:, :, 1] = test[:, :, 1]
test_img_R[:, :, 2] = test[:, :, 2]
cv2.imwrite("test_B.jpg", test_img_B.astype(np.uint8))
cv2.imwrite("test_G.jpg", test_img_G.astype(np.uint8))
cv2.imwrite("test_R.jpg", test_img_R.astype(np.uint8))'''

#Tone mapping(photographic reproduction/local)
Guassian_list = []
L_s = np.zeros((w, h))
L_ld = np.zeros((w, h))
for i in range(16):
    Guassian_list.append(cv2.GaussianBlur(L_m, (k_size, k_size), s))
    s *= alpha
for i in range(w):
    for j in range(h):
        L_s[i][j] = Guassian_list[0][i][j]
        s = 0.35
        for k in range(15):
            V_s = (Guassian_list[k][i][j] - Guassian_list[k + 1][i][j]) / (pow(2, phi) * a / pow(s, 2) + Guassian_list[k][i][j])
            if abs(V_s) < eps:
                L_s[i][j] = Guassian_list[k][i][j]
            s *= alpha
        L_ld[i][j] = L_m[i][j] * (1 + L_m[i][j] / (L_white ** 2)) / (1 + L_s[i][j])

L_l = L_ld / L_w
Ldr_img  = np.copy(hdr_img)
for k in range(3):
    Ldr_img[:, :, k] = Ldr_img[:, :, k] * L_l[:, :] * 255
Ldr_img = np.clip(Ldr_img, 0, 255)
Ldr_img_B[:, :, 0] = Ldr_img[:, :, 0]
Ldr_img_G[:, :, 1] = Ldr_img[:, :, 1]
Ldr_img_R[:, :, 2] = Ldr_img[:, :, 2]

cv2.imwrite("../data/Ldr_local.png", Ldr_img)
'''cv2.imwrite("Ldr_B.jpg", Ldr_img_B.astype(np.uint8))
cv2.imwrite("Ldr_G.jpg", Ldr_img_G.astype(np.uint8))
cv2.imwrite("Ldr_R.jpg", Ldr_img_R.astype(np.uint8))'''
print("tonemapping photographic local complete")

#adaptive logarithmic mapping
ldr_img_alm = ALM.Adaptive_Logarithmic_Mapping(hdr_img.astype(np.float32))
ldr_img_alm = np.clip(ldr_img_alm * 255, 0, 255).astype('uint8')
cv2.imwrite("../data/Ldr_log.png", ldr_img_alm)
print("tonemapping Logarithmic complete")
