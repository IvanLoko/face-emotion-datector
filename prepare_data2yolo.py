import pandas as pd
import cv2
from random import randint
import numpy as np
import glob
from progress.bar import IncrementalBar


class Data2YOLO:

    def __init__(self):
        # Форируем словарь с классами
        classes = [i.split('\\')[-2] for i in glob.glob('./data/train/*/')]
        self.classes = dict(zip(classes, range(len(classes))))

        # Для создания обучающий bounding boxes будем использовать face detector из дополнительным материалов
        # https://colab.research.google.com/drive/1BNSiKr2HUey85HDEpPFMQNKCOIR4Q8Cq
        self.face_detector = cv2.CascadeClassifier('face detector/haarcascade_frontalface_default.xml')

        try:
            self.train_df = pd.read_csv('./data/yolo_df.csv', index_col=0)
            self.create_bboxes()
            self.filter_data()
        # В случае, если dataFrame уже создан
        except FileExistsError:
            self.train_df = pd.read_csv('./data/train.csv', index_col=0)

    def prepare_data(self):

        if 'bboxes' in self.train_df.columns:
            return

        bboxes = []

        for _, row in self.train_df.iterrows():
            bar = IncrementalBar('Countdown', max=self.train_df.shape[0])

            # Считываем изображение
            bgr_image = cv2.imread('data/' + row.image_path[1:], )
            # Преобразуем изображение под необходимый формат для детектора лиц библиотеки cv2
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            # Добавляем найденный bounding box
            bboxes.append(self.face_detector.detectMultiScale(gray_image))

            bar.next()

        self.train_df['bboxes'] = bboxes

    def create_bboxes(self):
        """преобразуем строку в числовной формат"""
        bboxes = []
        for bbox_str in self.train_df.bboxes:
            if bbox_str == '()':
                bboxes.append(None)
                continue
            bbox_list = bbox_str.replace('[', '').replace(']', '').replace('\n', '').split(' ')
            bbox = np.array([int(i) for i in bbox_list if i != ''])
            bbox = np.split(bbox, len(bbox) / 4)
            bboxes.append(bbox)
        self.train_df['bboxes'] = bboxes
        self.train_df.dropna()

    def filter_data(self):
        """Избавляемся от данных с отсутствующим bounding box"""
        self.train_df.dropna(inplace=True)
        self.train_df = self.train_df[self.train_df.bboxes.apply \
                                          (lambda x: False if len(x) == 1 else True) == False]

    def create_txt(self):
        """Для обучения YOLO необходимы .txt к соответствующему экземпляру выборки. Создаем к каждому фото файл .txt
        Формат координат bbox: cls x y width height
            где:
                cls - эмоция
                x - центр bbox по оси x
                y - центр bbox по оси y
                width - ширина bbox
                height - высота bbox
                все числа должны быть нормированы (от 0 до 1)
                """
        bar = IncrementalBar('Countdown', max=self.train_df.shape[0])

        for index, row in self.train_df.iterrows():
            file_name = 'data/training/' + row.emotion + '_' + row.image_path.split('/')[-1][:-3]

            with open(file_name + 'txt', 'w') as txt_file:
                x, y, w, h = row['bboxes'][0]
                img = cv2.imread('data/' + row['image_path'])
                width, height = img.shape[:2]
                x = (x + w / 2) / width
                y = (y + h / 2) / height
                txt_file.write(f'{self.classes[row["emotion"]]} {x} {y} {w / width} {h / height}')

            cv2.imwrite(file_name + 'jpg', img)
            bar.next()

    def show_example(self):
        """Показывает пример найденного лица"""
        row_index = randint(0, self.train_df.shape[0])
        image_path, _, bbox = self.train_df.iloc[row_index]
        if bbox is None:
            return
        x, y, w, h = bbox[0]

        bgr_image = cv2.imread('data/' + image_path)

        rgb_image_with_bounding_box = cv2.rectangle(bgr_image, (y, x), (y + h, x + w), (0, 255, 0), 3)
        rgb_image_with_bounding_box = cv2.putText(rgb_image_with_bounding_box, 'Face', (y, x - 10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        cv2.imshow('detected face', rgb_image_with_bounding_box)
        cv2.waitKey(0)


if __name__ == '__main__':
    data2yolo = Data2Yolo()

    for _ in range(5):
        data2yolo.show_example()
