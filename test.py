import cv2
import numpy as np
import os

# Эталонные характеристики изображения
reference_features = {
    "Area": 43361.0,
    "Perimeter": 1000.2884466648102,
    "Centroid": (211, 195),
    "Aspect Ratio": 1.0888888888888888,
    "Extent": 0.5462459057697153,
    "Orientation": 49.4031982421875,
    "Eccentricity": 0.8891642679384879,
    "Solidity": 0.8701611446689812,
    "Hu Moments": np.array([2.37865238e-01, 2.71729529e-02, 5.13667984e-04, 1.31447244e-04,
                             2.53728770e-08, 1.10352095e-05, 2.28660741e-08])
}

folder_path = 'Uploads/Images'
result_folder = 'Results/Test_contour'

# Проверка наличия директории для результатов, создание если не существует
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Получение списка файлов в папке
files = os.listdir(folder_path)

# Фильтрация списка файлов, чтобы оставить только изображения
image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
features_list = []

# Обработка каждого изображения
for index, filename in enumerate(image_files):
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (400, 400))
    new_img = np.zeros(img.shape, dtype='uint8')
    
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 90, 100)
    
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    
    con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(new_img, con, -1, (0, 0, 255), 1)

    # Сохранение каждого изображения с контурами
    save_path = os.path.join(result_folder, f'test_contour_{index}.png')
    cv2.imwrite(save_path, new_img)

    # Отображение обработанного изображения
    # cv2.imshow('Picture to be recognized', new_img)
    # cv2.waitKey(1)  # Ждем 1 мс или до нажатия клавиши

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
        if len(contour) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            eccentricity = np.sqrt(1 - (MA / ma) ** 2) if ma != 0 else 0
        else:
            angle, eccentricity = 0, 0

        convex_hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / convex_hull_area if convex_hull_area != 0 else 0  
        HuMoments = cv2.HuMoments(M).flatten()

        # Сохранение характеристик в список
        features_list.append({
            "Area": area,
            "Perimeter": perimeter,
            "Centroid": (cx, cy),
            "Aspect Ratio": aspect_ratio,
            "Extent": extent,
            "Orientation": angle,
            "Eccentricity": eccentricity,
            "Solidity": solidity,
            "Hu Moments": HuMoments
        })

# Функция для сравнения двух наборов признаков
def compare_features(feat1, feat2):
    # Сравнение Hu Moments с использованием корреляции
    hu_distance = cv2.matchShapes(feat1['Hu Moments'], feat2['Hu Moments'], 1, 0.0)

    # Вы можете добавить другие метрики сравнения, если это необходимо
    # Например, сравнение площадей
    area_difference = abs(feat1['Area'] - feat2['Area'])

    # Вы можете выбрать пороговое значение для этих метрик
    # Здесь пример с произвольными значениями
    is_similar = hu_distance < 0.1 and area_difference < 1000
    return is_similar

# Сравнение и вывод результатов
for features in features_list:
    is_similar = compare_features(reference_features, features)
    print("Этот контур похож на эталонный:", is_similar)

    # Вывод детальной информации, если контур похож
    if is_similar:
        print("Детали контура:")
        print("Площадь:", features["Area"])
        print("Периметр:", features["Perimeter"])
        print("Центроид:", features["Centroid"])
        print("Аспектное соотношение:", features["Aspect Ratio"])
        print("Протяженность:", features["Extent"])
        print("Ориентация:", features["Orientation"])
        print("Эксцентриситет:", features["Eccentricity"])
        print("Твердость:", features["Solidity"])
        print("Hu Moments:", features["Hu Moments"])



























# import cv2
# import numpy as np
# import os

# folder_path = 'Uploads/Images'
# result_folder = 'Results/Test_contour'

# # Проверка наличия директории для результатов, создание если не существует
# if not os.path.exists(result_folder):
#     os.makedirs(result_folder)

# # Получение списка файлов в папке
# files = os.listdir(folder_path)

# # Фильтрация списка файлов, чтобы оставить только изображения
# image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
# con_list = []
# # Обработка каждого изображения
# for index, filename in enumerate(image_files):
#     img_path = os.path.join(folder_path, filename)
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (400, 400))
#     new_img = np.zeros(img.shape, dtype='uint8')
    
#     img = cv2.GaussianBlur(img, (11, 11), 0)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.Canny(img, 90, 100)
    
#     kernel = np.ones((2, 2), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=1)
#     img = cv2.erode(img, kernel, iterations=1)
    
#     con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#     con_list.append(con)
#     cv2.drawContours(new_img, con, -1, (0, 0, 255), 1)

#     # Сохранение каждого изображения с контурами
#     save_path = os.path.join(result_folder, f'test_contour_{index}.png')
#     cv2.imwrite(save_path, new_img)

#     # Отображение обработанного изображения
#     cv2.imshow('Picture to be recognized', new_img)
#     cv2.waitKey(1)  # Ждем 1 мс или до нажатия клавиши

# cv2.destroyAllWindows()  # Закрыть все окна после цикла


# # ---------------------------------------------------------------------------------
# # Перебор контуров и извлечение характеристик
# for con in con_list:
#     for contour in con:
#         area = cv2.contourArea(contour)
#         perimeter = cv2.arcLength(contour, True)
#         M = cv2.moments(contour)
#         cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
#         cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
#         x, y, w, h = cv2.boundingRect(contour)
#         aspect_ratio = w / h
#         rect_area = w * h
#         extent = area / rect_area

#         # Проверка на достаточное количество точек для построения эллипса
#         if len(contour) >= 5:
#             (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
#             eccentricity = np.sqrt(1 - (MA / ma) ** 2) if ma != 0 else 0
#         else:
#             angle, eccentricity = 0, 0

#         solidity = area / cv2.contourArea(cv2.convexHull(contour))
#         HuMoments = cv2.HuMoments(M).flatten()

#         print("Area:", area, "Perimeter:", perimeter, "Centroid: (", cx, ",", cy, ")",
#               "Aspect Ratio:", aspect_ratio, "Extent:", extent, "Orientation:", angle,
#               "Eccentricity:", eccentricity, "Solidity:", solidity, "Hu Moments:", HuMoments)
    
# # # ----------------------------------------------------------------------------------------

# def compare_contours(features1, features2):
#     # Пример: Сравнение с использованием Евклидова расстояния
#     distance = np.linalg.norm(np.array(features1) - np.array(features2))
#     return distance



# def classify_contour(reference_features, test_features, threshold=10.0):
#     if compare_contours(reference_features, test_features) < threshold:
#         return "Similar"
#     else:
#         return "Not similar"


# reference = get_features('reference.jpg')  # Замените на функцию, которая возвращает признаки изображения
# test_images = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # Список тестовых изображений

# results = {}
# for test_image in test_images:
#     test_features = get_features(test_image)
#     result = classify_contour(reference, test_features)
#     results[test_image] = result

# print(results)

    
# cv2.waitKey(0)
# cv2.destroyAllWindows()