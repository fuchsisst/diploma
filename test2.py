import cv2
import numpy as np
from skimage.feature import hog
from scipy.spatial.distance import euclidean

def extract_hog_features(image):
    # Преобразование в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Вычисление HOG признаков
    features, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1), visualize=True)

    return features, hog_image


def compare_hog_features(features1, features2):
    # Вычисление евклидова расстояния между двумя векторами признаков
    distance = euclidean(features1, features2)
    return distance

# Загрузка изображений
image1 = cv2.imread('Uploads/etalon/granata.jpg')
image1  = cv2.resize(image1 , (400, 400))
image2 = cv2.imread('Uploads/Images/PFM-1_8.jpg')
image2 = cv2.resize(image2, (400, 400))

# Извлечение HOG признаков
features1, hog_image1 = extract_hog_features(image1)
features2, hog_image2 = extract_hog_features(image2)

# Сравнение признаков
distance = compare_hog_features(features1, features2)
print("Расстояние между изображениями:", distance)

# Определение схожести
threshold = 15  # Порог можно настроить в зависимости от задачи
print("Изображения схожи:", distance < threshold)
import matplotlib.pyplot as plt

# После извлечения признаков HOG и изображения HOG:
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(hog_image1, cmap='gray')
plt.title('HOG Image 1')
plt.subplot(122)
plt.imshow(hog_image2, cmap='gray')
plt.title('HOG Image 2')
plt.show()


# RGB - Standart
# BGR - OpenCV format

# 1. Реализуем возможность загрузить и показать пользователю изображение.
# img = cv2.imread('Uploads/Images/images.jpeg')

# img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))

# # 2. Размытие изображения
# img = cv2.GaussianBlur(img, (5, 5), 0)

# # 3. Приведем картинку из формата RGB в формат серой картинки
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # 4. Найдем углы и края изображения
# img = cv2.Canny(img, 120, 120)

# # 5. Увеличим толщину обводки
# kernel = np.ones((2, 2), np.uint8)
# img = cv2.dilate(img, kernel, iterations=1)

# # 6. Уменьшим толщину обводки
# img = cv2.erode(img, kernel, iterations=1)


# cv2.imshow('Picture to be recognized', img)

# cv2.waitKey(0)


# --------------------------------------------
# В будущем для обозначений найденных объектов

#photo = np.zeros((450, 450, 3), dtype='uint8')

#cv2.rectangle(photo, (50, 70), (100, 100), (119, 201, 105), thickness = 1)
#cv2.line(photo, (0, photo.shape[0]//2), (photo.shape[1], photo.shape[0]//2), (119, 201, 105), thickness = 3)
#cv2.circle(photo, (photo.shape[1]//2, photo.shape[0]//2), 50, (119, 201, 105), thickness = 1)
#cv2.putText(photo, 'Text', (100, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), thickness=1)
#cv2.imshow('Photo', photo)
#cv2.waitKey(0)

# -------------------------------------------

# img = cv2.imread('Uploads/Images/images.jpeg')

# new_img = np.zeros(img.shape, dtype='uint8')

#img = cv2.flip(img, 1)

# def rotate(img_param, angle):
#     heigth, width = img_param.shape[:2]
#     point = (width // 2, heigth // 2)
    
#     mat = cv2.getRotationMatrix2D(point, angle, 1)
#     return cv2.warpAffine(img_param, mat, (width, heigth))

# img = rotate(img, 90)

# def transform(img_param, x, y):
#         mat = np.float32([[1, 0, x], [0, 1, y]])
        
#         return cv2.warpAffine(img_param, mat, (img_param.shape[1], img_param.shape[0]))

# img = transform(img, 30, 10)

# НАЙДЕМ КОНТУРЫ ИЗОБРАЖЕНИЯ
# img = cv2.resize(img, (img.shape[1], img.shape[0]))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img, (5, 5), 0)
# img = cv2.Canny(img, 110, 160)

# 1. Находим контурі изображения
# В первой переменной хранится список со всеми позициями контуров
# Во второй переменной хранится иерархия самих объектов
# con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# cv2.drawContours(new_img, con, -1, (0, 0, 255), 1)
# print(con)
# cv2.imshow('Result', new_img)

# cv2.waitKey(0)

#-----------------------------------------
# Цветовые форматы
# img = cv2.imread('Uploads/Images/images.jpeg')

# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)

# r, g, b = cv2.split(img)

# img = cv2.merge([b, g, r])
# cv2.imshow('Result', b)

# cv2.waitKey(0)

# ---------------------------------------
# Побитовые операции

# img = cv2.imread('Uploads/Images/images.jpeg')
# photo = np.zeros(img.shape[:2], dtype='uint8')

# circle = cv2.circle(photo.copy(), (230, 120), 80, 255, -1)
# square = cv2.rectangle(photo.copy(), (25, 25), (250, 350), 255, -1)

# photo = cv2.bitwise_and(img, img, mask=square) 

# cv2.imshow("Result", photo)
# cv2.waitKey(0)


