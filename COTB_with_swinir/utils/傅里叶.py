import cv2
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25
# 读取低分辨率图像和原始高清图像
# lr_image_path =
hr_image_path = "E:\\pycharm\\projects\\KAIR-master\\results\\swinir_lightweight_sr_x2\\img_001_SwinIR.png"
swinir_image_path_1 = "E:\\pycharm\\projects\\KAIR-master\\results\\swinir_lightweight_sr_x4_yuan\\111 (2).png"
swinir_image_path_2 = "E:\\pycharm\\projects\\KAIR-master\\results\\swinir_lightweight_sr_x3_yuan\\12 (1).png"

# 加载图像
# lr_image = cv2.imread(lr_image_path)
hr_image = cv2.imread(hr_image_path)
swinir_image_1 = cv2.imread(swinir_image_path_1)
swinir_image_2 = cv2.imread(swinir_image_path_2)

# 定义一个函数来计算频谱
def compute_frequency_spectrum(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_transform = np.fft.fft2(gray_image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    return magnitude_spectrum

# 计算频谱
# lr_spectrum = compute_frequency_spectrum(lr_image)
hr_spectrum = compute_frequency_spectrum(hr_image)
swinir_spectrum_1 = compute_frequency_spectrum(swinir_image_1)
swinir_spectrum_2 = compute_frequency_spectrum(swinir_image_2)

# 可视化结果
plt.figure(figsize=(20, 12))

# 显示低分辨率图像的频谱
# plt.subplot(2, 3, 1)
# plt.imshow(lr_spectrum, cmap='gray')
# plt.title('Frequency Spectrum of Low Resolution Image')
# plt.axis('off')

# 显示SwinIR处理后的第一张图像频谱
plt.subplot(1, 3, 3)
plt.imshow(swinir_spectrum_1, cmap='gray')
plt.title('Frequency Spectrum of scale ×4')
plt.axis('off')

# 显示原始高清图像的频谱
plt.subplot(1, 3, 1)
plt.imshow(hr_spectrum, cmap='gray')
plt.title('Frequency Spectrum of scale ×2')
plt.axis('off')

# 显示SwinIR处理后的第二张图像的频谱
plt.subplot(1, 3, 2)
plt.imshow(swinir_spectrum_2, cmap='gray')
plt.title('Frequency Spectrum of scale ×3')
plt.axis('off')

plt.tight_layout()
plt.show()
