# Датасет LEGO® Colors

Этот датасет предназначен для обучения моделей распознавания цветов LEGO® кубиков. Включает в себя изображения различных цветовых вариантов популярных конструкторов LEGO®.

## Структура датасета

Датасет разделен на подкаталоги, каждый из которых соответствует отдельному цвету. Внутри каждого подкаталога находятся изображения LEGO® кубиков соответствующего цвета.

## Использование

Датасет можно загрузить с [Kaggle](https://www.kaggle.com/datasets/thijshavinga/lego-colors/data). Вы также можете найти txt-файл в папке `dataset`, который содержит ссылку на обработанный датасет. Python-скрипты для обработки датасета находятся в папке `processing`. Веса нейросети после обучения сохранены в папке `weights`.


## Пример Нейросети
Приведенный ниже код на Python представляет собой пример сверточной нейронной сети (CNN) для задачи классификации цветов LEGO® кубиков. Этот код может быть использован в качестве отправной точки для обучения модели на данном датасете.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LegoColorNet(nn.Module):
    def __init__(self, n_chans=32):
        super(LegoColorNet, self).__init__()
        self.n_chans = n_chans
        self.conv1 = nn.Conv2d(3, n_chans, kernel_size=3, padding=1, stride=1)
        self.conv1_batch_norm = nn.BatchNorm2d(num_features=n_chans)
        self.conv2 = nn.Conv2d(n_chans, n_chans // 2, kernel_size=3, padding=1, stride=1)
        self.conv2_batch_norm = nn.BatchNorm2d(num_features=n_chans // 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_chans * 2 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 57)  # 57 classes for LEGO® colors

    def forward(self, x):
        out = F.max_pool2d(F.relu(self.conv1_batch_norm(self.conv1(x))), 2)
        out = F.max_pool2d(F.relu(self.conv2_batch_norm(self.conv2(out))), 2)
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# Пример создания экземпляра модели
model = LegoColorNet().to(device=device)

# Пример определения потерь и оптимизатора (можно настроить в соответствии с задачей)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().to(device=device)
```

## Результаты обучения
<div align="center">
  <img src="https://github.com/Lapamore/Kaggle_competitions/blob/main/Lego%20Image%20Color%20Classifier/results/Обучение.png" alt="График обучения" width="400"/>
  <img src="https://github.com/Lapamore/Kaggle_competitions/blob/main/Lego%20Image%20Color%20Classifier/results/Потери.png" alt="График потерь" width="400"/>
</div>

## Результат инференса
![Predict](https://github.com/Lapamore/Kaggle_competitions/blob/main/Lego%20Image%20Color%20Classifier/results/Предсказания.png)
