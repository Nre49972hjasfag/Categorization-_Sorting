import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

dataset = {
    "Веб-разработка": [
        "Создать интернет-магазин одежды на WordPress под ключ.",
        "Разработать бэкенд на Django для корпоративного портала.",
        "Нужна верстка лендинга по готовому макету в Figma, адаптив.",
        "Исправить баги в скриптах JavaScript на сайте-визитке.",
        "Интегрировать платежную систему Stripe на сайт на React.",
        "Разработка веб-сервиса для бронирования отелей.",
        "Перенести существующий сайт компании на CMS 1с-Битрикс.",
        "Написать API для фронтенда на Node.js и Express.",
        "Оптимизировать скорость загрузки веб-страниц интернет-ресурса.",
        "Настроить личный кабинет пользователя для онлайн-школы."
    ],
    "Мобильное приложение": [
        "Разработать мобильное приложение для доставки еды на Flutter.",
        "Нужно написать нативное приложение под iOS на Swift.",
        "Создать MVP мобильного приложения для Android на Kotlin.",
        "Опубликовать готовое приложение в Google Play и App Store.",
        "Интегрировать push-уведомления и карты в мобильный софт.",
        "Исправить вылеты в мобильном приложении трекера привычек.",
        "Разработать дизайн и логику мобильного приложения для фитнеса.",
        "Сделать редизайн интерфейса мобильной игры на Unity.",
        "Добавить темную тему в существующее iOS приложение.",
        "Настроить внутриигровые покупки (In-App Purchases) для Android."
    ],
    "Дизайн": [
        "Разработать UI/UX макет веб-сайта в Figma, 5 страниц.",
        "Создать логотип и фирменный стиль для кофейни.",
        "Нужен дизайн баннеров для рекламной кампании.",
        "Оформить презентацию компании на 15 слайдов в PDF.",
        "Разработать дизайн упаковки для бренда натуральной косметики.",
        "Сделать 3D-модель продукта для карточки товара на маркетплейсе.",
        "Отрисовать иконки для интерфейса мобильной программы.",
        "Создать брендбук и гайдлайн по использованию шрифтов.",
        "Разработка инфографики для Wildberries и Ozon.",
        "Поправить шрифты и цветоввую гамму на готовом макете."
    ],
    "Аналитика": [
        "Провести UX-исследование и анализ путей пользователей (CJM).",
        "Собрать требования и составить  Т3 для команды разработки.",
        "Настроить сквозную аналитику в Google Analytics 4.",
        "Построить дашборд в Tableau на основе данных из CRM.",
        "Проанализировать метрики удержания пользователей (Retention).",
        "Провести конкурентный анализ рынка онлайн-образования.",
        "Выгрузить данные из базы PostgreSQL и сделать отчет в Excel.",
        "Проверить продуктовые гипотезы с помощью A/B тестирования.",
        "Составить карту процессов AS-IS и TO-BE для бизнеса.",
        "Анализ эффективности воронки продаж интернет-магазина."
    ],
    "Маркетинг": [
        "Настроить контекстную рекламу в Яндекс.Директ под ключ.",
        "Запустить таргет в социальных сетях для привлечения лидов.",
        "Разработать контент-план и стратегию продвижения в Telegram.",
        "Написать SEO-оптимизированные статьи для блога компании.",
        "Запустить email-рассылку по базе существующих клиентов.",
        "Разработать стратегию вывода нового продукта на рынок.",
        "Провести аудит текущей рекламной кампании в сети.",
        "Организовать инфлюенс-маркетинг, подбор блогеров для рекламы.",
        "Написать продающий текст для промо-страницы услуг.",
        "Увеличить конверсию сайта за счет маркетинговых акций."
    ]
}


# 1. Подготовка обучающих данных
texts, labels = [], []
for category, task_list in dataset.items():
    for task in task_list:
        texts.append(task.lower())
        labels.append(category)

# Векторизация текстов (символьные n-граммы от 3 до 5)
vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3, 5))
X_vectorized = vectorizer.fit_transform(texts).toarray()

# Кодирование текстовых меток в числа (0, 1, 2...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# Перевод в тензоры PyTorch
X_tensor = torch.tensor(X_vectorized, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)


# 2. Архитектура легкой нейросети
class LightweightClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LightweightClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 3. Инициализация и обучение
input_size = X_tensor.shape[1]
num_classes = len(label_encoder.classes_)

model = LightweightClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 40
model.train()

print("Процесс обучения модели:")
print("-" * 30)
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    # Контроль обучения: выводим лосс каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        print(f"Эпоха [{epoch+1}/{epochs}], Ошибка (Loss): {loss.item():.4f}")


# 4. Проверка на новых тестовых задачах
test_tasks = [
    "Сделать сайт-визитку для адвоката на Тильде.",
    "Написать приложение под айфон для учета калорий.",
    "Нарисовать красивую обложку для группы ВКонтакте.",
    "Нам нужно посчитать ROI и составить отчет по продажам за квартал.",
    "Настроить поток заявок из инстаграма через блогеров.",
    "Добавить корзину и оплату картами на веб сайт."
]

# Переводим модель в режим оценки
model.eval()
softmax = nn.Softmax(dim=1)

# Батчевая обработка тестовых данных
test_texts_cleaned = [task.lower() for task in test_tasks]
X_test_vectorized = vectorizer.transform(test_texts_cleaned).toarray()
X_test_tensor = torch.tensor(X_test_vectorized, dtype=torch.float32)

with torch.no_grad():
    logits = model(X_test_tensor)
    probabilities = softmax(logits)
    confidences, predicted_indices = torch.max(probabilities, dim=1)

# Декодируем сразу весь массив предсказанных индексов
predicted_labels = label_encoder.inverse_transform(predicted_indices.numpy())

print("\nРезультаты классификации нейросети:\n" + "="*50)
# Корректный вывод результатов предсказания
for task, label, conf in zip(test_tasks, predicted_labels, confidences):
    print(f"Текст: \"{task}\"")
    print(f"Робо-категория: {label} (Уверенность: {conf.item():.2%})")
    print("-" * 50)
