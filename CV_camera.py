import cv2
from ultralytics import YOLO


class CVCamera:
    "детекции и классификация лиц с помощью подключенной камеры"""

    def __init__(self, model='./model/best.pt'):
        # Загружаем веса модели
        self.model = YOLO(model)
        self.classes = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy',
                        5: 'neutral', 6: 'sad', 7: 'surprise', 8: 'uncertain'}

    def prepare_frame(self, frame):
        """Инференс сети. Обработка изображения: во входной frame добавляется полученный bbox и confidanсe эмоции """
        result = self.model(frame)[0]
        if result.boxes.cls.nelement() != 0:
            for box in result.boxes:
                bbox = box.xyxy.numpy().astype('int32')
                confidence = box.conf.numpy()
                cls = box.cls.numpy()
                frame = cv2.rectangle(frame, bbox[0][:2], bbox[0][2:], (36, 255, 12))
                frame = cv2.putText(frame, '{} {:.2}'.format(self.classes[cls[0]], confidence[0]),
                                    (bbox[0][0], bbox[0][1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12))
        return frame

    def start(self, fps=12):
        # Запуск камеры
        cam = cv2.VideoCapture(0)
        if cam:
            print('camera started!')
        else:
            print('Invalid started')

        frame_count = 0
        # Считываем кадры до отключения камеры/прерывания программы
        while True:
            ret, frame = cam.read()
            # Чтобы полученное изображение не тормазило, считываем каждый 4 кадр,
            # так как средняя частота моей камеры = 4
            if frame_count % fps/3 == 0:
                image = self.prepare_frame(frame)

            if ret:
                cv2.imshow('camera!!!', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    CVCamera().start()
