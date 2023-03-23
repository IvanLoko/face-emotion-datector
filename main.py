from prepare_data2yolo import Data2YOLO


if __name__ == '__main__':

    # Для обучения YOLO необходимо подготовить датасет для обучения
    # Предобработка данных и формирование датасета реализованно в классе Data2Yolo
    data2yolo = Data2YOLO()
    data2yolo.prepare_data()
    data2yolo.create_txt()

    # Обучения сети YOLOv8n производилось в colab.researc
    # https://colab.research.google.com/drive/1XSsMeFpnAYMcpE9LAY1IogPDKVrAoB_c#scrollTo=4ZMkft-c-sCO&uniqifier=5
    # Результаты работы сети на тестовом датасете test_kaggle
    # https://colab.research.google.com/drive/1XSsMeFpnAYMcpE9LAY1IogPDKVrAoB_c#scrollTo=4ZMkft-c-sCO&uniqifier=5





