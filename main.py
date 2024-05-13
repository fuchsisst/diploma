import cv2
import numpy as np

# RGB - Standart
# BGR - OpenCV format

# 1. Реализуем возможность загрузить и показать пользователю изображение.
img = cv2.imread('Uploads/etalon/granata.jpg')

img = cv2.resize(img, (400, 400))

new_img = np.zeros(img.shape, dtype='uint8')

# 2. Размытие изображения
img = cv2.GaussianBlur(img, (11, 11), 0)

# 3. Приведем картинку из формата RGB в формат серой картинки
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. Найдем углы и края изображения
img = cv2.Canny(img, 90, 240)

# 5. Увеличим толщину обводки
kernel = np.ones((2, 2), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)

# 6. Уменьшим толщину обводки
img = cv2.erode(img, kernel, iterations=1)

# 7. Находим контурі изображения
# В первой переменной хранится список со всеми позициями контуров
# Во второй переменной хранится иерархия самих объектов
con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(new_img, con, -1, (0, 0, 255), 1)
# print(con)

cv2.imshow('Picture to be recognized', new_img)

# Сохраняем контур изображения
# cv2.imwrite('Results/Etalon_contour/contours1.png', new_img)

# ---------------------------------------------------------------------------------
# Перебор контуров и извлечение характеристик
for contour in con:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    rect_area = w * h
    extent = area / rect_area

    # Проверка на достаточное количество точек для построения эллипса
    if len(contour) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        eccentricity = np.sqrt(1 - (MA / ma) ** 2) if ma != 0 else 0
    else:
        angle, eccentricity = 0, 0

    solidity = area / cv2.contourArea(cv2.convexHull(contour))
    HuMoments = cv2.HuMoments(M).flatten()

    print("Area:", area, "Perimeter:", perimeter, "Centroid: (", cx, ",", cy, ")",
          "Aspect Ratio:", aspect_ratio, "Extent:", extent, "Orientation:", angle,
          "Eccentricity:", eccentricity, "Solidity:", solidity, "Hu Moments:", HuMoments)
# ----------------------------------------------------------------------------------------


    
cv2.waitKey(0)
cv2.destroyAllWindows()