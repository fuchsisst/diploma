import matplotlib.pyplot as plt
import cv2
import numpy as np

image = cv2.imread('Uploads/etalon/granata.jpg')

# Преобразование в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Изменение размера до новых ширины и высоты
resized_image = cv2.resize(gray_image, (400, 400))

# Бинаризация изображения
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Применение медианного фильтра для устранения шума
denoised_image = cv2.medianBlur(binary_image, 5) // binary_image

import numpy as np

def homomorphic_filter(image, cutoff_frequency=30, high_boost_parameter=2):
    # Преобразование изображения в логарифмическую шкалу
    log_image = np.log1p(np.array(image, dtype="float"))

    # Применение быстрого преобразования Фурье
    fft = np.fft.fft2(log_image)

    # Центрирование нулевой частоты
    fft_shifted = np.fft.fftshift(fft)

    # Создание фильтра Гаусса
    rows, cols = log_image.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = cutoff_frequency
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0

    # Применение фильтра
    fft_shifted = fft_shifted * mask

    # Обратное преобразование Фурье
    ifft_shifted = np.fft.ifftshift(fft_shifted)
    ifft = np.fft.ifft2(ifft_shifted)

    # Преобразование обратно из логарифмической шкалы
    result = np.expm1(np.real(ifft))

    # Увеличение контраста с помощью усиления высоких частот
    result = high_boost_parameter * result + image

    return np.uint8(result)

# Пример использования
filtered_image = homomorphic_filter(binary_image)

# Применение гауссова размытия для сглаживания шумов
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Применение алгоритма Кэнни для обнаружения границ
edges = cv2.Canny(blurred_image, 50, 150)

# Нахождение контуров на изображении
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Создание копии изображения для отрисовки контуров
contour_image = image.copy()

# Отрисовка контуров на копии изображения
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Применение морфологического преобразования "замыкание" (closing)
kernel = np.ones((5, 5), np.uint8)
closing_result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Применение морфологического преобразования "раскрытие" (opening)
opening_result = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Применение морфологического преобразования "эрозия"
erosion_result = cv2.erode(binary_image, kernel, iterations=1)

# Применение морфологического преобразования "диляция"
dilation_result = cv2.dilate(binary_image, kernel, iterations=1)

contur = image 
# Поиск контуров на бинарном изображении
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Определение компактности для каждого контура
for contour in contours:
    # Вычисление площади контура
    area = cv2.contourArea(contour)

    # Вычисление периметра контура
    perimeter = cv2.arcLength(contour, True)

    # Вычисление компактности (отношение площади к периметру)
    compactness = (4 * np.pi * area) / (perimeter**2)

    # Определение порога компактности (может потребоваться настройка)
    compactness_threshold = 0.05

    # Если компактность объекта удовлетворяет условию
    if compactness > compactness_threshold:
        # Отмечаем объект на исходном изображении (рисуем контур)
        cv2.drawContours(contur, [contour], -1, (0, 255, 0), 2)


# Отображение изображений рядом с помощью matplotlib
plt.figure(figsize=(15, 10))

plt.subplot(3, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(denoised_image, cmap='gray')
plt.title('Denoised Image')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(filtered_image, cmap='gray')
plt.title('Homomorphic Filtered Image')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(contour_image, cmap='gray')
plt.title('Contour segmentation')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(dilation_result, cmap='gray')
plt.title('Morphological processing')
plt.axis('off')


plt.subplot(3, 3, 7)
plt.imshow(contur, cmap='gray')
plt.title('Identification of objects by their compactness')
plt.axis('off')

plt.show()