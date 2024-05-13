import cv2
import numpy as np

# Загрузка изображения и преобразование в оттенки серого
image = cv2.imread('Uploads/etalon/granata.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (400, 400))
# Применение бинаризации для получения бинарного изображения
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

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
        cv2.drawContours(image, [contour], -1, (0, 255, 123), 5)

# Показ изображений
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
