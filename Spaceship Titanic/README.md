# Соревнование Spaceship Titanic
## Оглавление
1. [Описание](#описание)
2. [Визуализация](#визуализация)
    1. [Визуализация](#визуализация)
    2. [Визуализация](#визуализация)
    3. [Визуализация](#визуализация)
4. [Анализ данных](#анализ-данных)
5. [Подготовка данных](#подготовка-данных)
6. [Выбор модели](#выбор-модели)
7. [Обучение модели](#обучение-модели)
8. [Предсказание](#предсказание)
9. [Оценка и результаты](#оценка-и-результаты)
10. [Заключение](#заключение)

## Описание
В 2912 году корабль "Spaceship Titanic" столкнулся с пространственно-временной аномалией, и половина пассажиров была перемещена в альтернативное измерение. Вам предстоит по записям из поврежденной компьютерной системы определить, какие пассажиры были затронуты аномалией.

## Анализ данных:
Мой путь в этом соревновании начался с внимательного анализа предоставленных данных. После осмотра выборок train и test стало ясно, что передо мной стоит задача работы с пятью категориальными и семью количественными признаками. Это первый шаг, который помог мне в разработке стратегии для создания модели. 

```python
train_data.head()
```

```python
train_data.info()
```

```python
train_data.describe()
```
```python
test_data.head()
```

Стоит отметить, что анализ показал наличие пропущенных значений в данных. Это выявление стало важным этапом, так как понимание, какие признаки и насколько подвержены пропущенным значениям, будет влиять на последующий этап обработки данных. 

## Визуализация:
Прежде чем приступать к подготовке данных, я решил визуализировать имеющуюся информацию, чтобы понять с чем мне предстоит работать.

### График распределения возрата:
<div style="text-align: center;">
    <img src="https://github.com/Lapamore/Kaggle_competitions/blob/main/Spaceship%20Titanic/img/hist.png" alt="График распределения возрата" width="550" height="400">
</div>
Заметно, что основная часть пассажиров космического корабля находится в возрастной группе от 13 до 38 лет.

### График Boxplot
<div style="text-align: center;">
    <img src="https://github.com/Lapamore/Kaggle_competitions/blob/main/Spaceship%20Titanic/img/boxplot.png" alt="График Boxplot" width="500" height="400">
</div>
Анализ графика Boxplot позволяет выявить наличие выбросов в данных столбца "Age". Это нюанс, который мы учтем при последующей обработке.

### График корреляции:
<div style="text-align: center;">
    <img src="https://github.com/Lapamore/Kaggle_competitions/blob/main/Spaceship%20Titanic/img/corr.png" alt="График корреляции" width="500" height="400">
</div>
Изучая матрицу корреляции, можно выделить переменные, между которыми наблюдается высокая степень взаимосвязи.

## Подготовка данных:
Перед тем, как приступать к обработке данных я создал 4 функции:
- FeatureEncoder - преобразует категориальные признаки в в числовой формат.
- drop_features - удаляет ненужные столбцы.
- fill_value - заполняет пропущенные данные.
- add_new_feature - добавляет новые столбцы.
  
```python
def FeatureEncoder(X):
    encoder = OneHotEncoder()
    matrix = encoder.fit_transform(X[['Embarked']]).toarray()

    column_names = ["C", "S", "Q", "N"]

    for i in range(len(matrix.T)):
        X[column_names[i]] = matrix.T[i]

    matrix = encoder.fit_transform(X[['Sex']]).toarray()  
    column_names = ["Female", "Male"]

    for i in range(len(matrix.T)):
        X[column_names[i]] = matrix.T[i]

    return X
```

```python
def drop_features(X):
    return X.drop(['Name','Cabin', 'PassengerId', 'Destination' 'HomePlanet', 'Deck'], axis=1 , errors='ignore')
```

```python
def fill_value(X):
    for col in X.columns:
        if X[col].isnull().sum() == 0:
            continue

        if X[col].dtype == object or X[col].dtype == bool:
            X[col] = X[col].fillna(X[col].mode()[0])

        else:
            X[col] = X[col].fillna(X[col].mean())
    return X
```

```python
def add_new_feature(X): 
    deck = X['Cabin'].map(lambda x: x.split("/") if isinstance(x, str) else [])
    
    X['Group'] = X['PassengerId'].map(lambda x: x.split("_")).apply(lambda x: int(x[1]))
    X['Deck'] = deck.apply(lambda x: x[0] if len(x) > 0 else None)
    X['Side'] = deck.apply(lambda x: x[2] if len(x) > 0 else None)
    return X
```
Расскажу побробнее, что я использовал в функциях:
* Одним из наиболее распространённых методов преобразования категориальных признаков является `OneHotEncoding`, который позволяет перевести переменные в числовой формат, при этом избегая ненужной упорядоченности или искажения значений. Именно этот метод я использовал в функции `FeatureEncoder`.

* Я добавил 3 новых столбца: 
    * Group - Этот столбец указывает на группу, с которой путешествует пассажир. Он поможет выявить связи между пассажирами и определить, какие группы могут иметь больший шанс выживания.
    * Deck - Введенный столбец "Deck" обозначает палубу, на которой расположен пассажир. Этот аспект может иметь значение для выживаемости, так как некоторые палубы могли быть ближе к спасательным средствам.
    * Side -  Столбец обозначает сторону корабля, на которой находится пассажир. В некоторых ситуациях определенные стороны могли быть более или менее подвержены опасности, что может повлиять на результат.

**Не добавленный столбец**\
В свете того, что пассажиры могли путешествовать семьями, я принял решение исследовать, как влияет фактор семейной принадлежности на предсказание модели. Попытка добавить четвертый столбец "family" в модель, к сожалению, не принесла ожидаемого улучшения результатов. Стоит отметить, что также приходилось иметь дело с пропущенными данными в столбце 'Name', которые нельзя было заполнить модой или другими методами. В итоге, было решено не включать эту фичу в функцию `add_new_feature`.

Реализация столбца "family":
```python
def AddFamily(X):
    family = X['Name'].apply(lambda x: x.split(" "))
    split_names_df = pd.DataFrame([i for i in family], columns=['First Name', 'Last Name'])
    split_names_df = split_names_df.groupby("Last Name", as_index=False).agg({"First Name": 'count'})
    split_names_df['Family'] = np.where(split_names_df['First Name'] > 1, 1, 0)
    
    family_list_values = []
    for col in X.Name:
        value_family = split_names_df['Family'].loc[split_names_df['Last Name'] == col.split(" ")[1]]
        family_list_values.append(value_family.iloc[0])
        
    X['Family'] = family_list_values
    
    return X
```

## Обработке и предобработка данных для обучения модели:
Следующим этапом было применение всех вышеописанных функцих к train_data.

```python
train_data = add_new_feature(train_data)
train_data = FeatureEncoder(train_data)
train_data = drop_features(train_data)
```
Кроме того, важно отметить, что обработка пропущенных значений в столбце "Age" была выполнена отдельно в связи с наличием выбросов:

```python
mean_age = train_data[train_data['Age'] < 61]['Age'].mean()
train_data['Age'] = train_data['Age'].fillna(mean_age)
```
Дополнительные шаги обработки включали:
```python
train_data['Side'] = train_data['Side'].map({"P": 1, "S":2})

bool_features = ['CryoSleep', 'VIP', 'Transported']
for col in bool_features:
    train_data[col] = train_data[col].map({False:0, True:1})
```
После этого я получаю train_data с полной обработкой

С учетом сделанных исключений, я сформировал переменную X, включающую в себя dataframe без столбца Transported, который содержит правильные ответы. Вектор правильных ответов (Transported) я поместил в переменную y. Это предоставило мне четкое разделение между входными признаками и целевой переменной, что является фундаментом для дальнейшего построения и обучения модели.

```python
X = train_data.drop("Transported", axis=1)
y = train_data['Transported']
```

### Разделение на тренировочную и тестовую выборки
После всех этапов предобработки данных, следующим шагом было разделение данных на тренировочную и тестовую выборки. Это позволяет оценить качество обучения модели на отложенных данных и проверить, насколько успешно модель будет работать на новых наборах данных.

Для этого я использовал функцию `train_test_split` из библиотеки `scikit-learn`, которая случайным образом разбивает данные на две части: одну для обучения модели (тренировочную выборку) и другую для оценки её производительности (тестовую выборку).

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3, random_state=17)
```

## Выбор модели:

Я решил использовать два классификатора - RandomForestClassifier и XGBoost. После проведения экспериментов, XGBoost оказался более эффективным, поэтому я выбрал его в качестве основной модели. Реализацию RandomForestClassifier Вы можете посмотреть в файле 'RandomForestClassifier_model'.
 
## Обучение модели
Для настройки параметров модели и достижения наилучшей производительности, я применил метод `GridSearchCV`. Этот метод позволяет обучать модель на кросс-валидации, автоматически перебирая различные комбинации заданных параметров.

В результате, переменная с наилучшими параметрами модели будет создана автоматически. Это позволяет упростить и ускорить процесс поиска наиболее оптимальных параметров для данной выборки данных.

```python
clf = xgb.XGBClassifier(objective='binary:logistic', random_state=17)

param_grid = {
    'n_estimators': [100, 200, 300],             # Number of trees
    'learning_rate': [0.01, 0.1, 0.2],           # Learning rate (gradient descent step)
    'max_depth': [3, 6, 9],                      # Maximum tree depth
    'min_child_weight': [1, 3, 5],               # Minimum sum of child weights in a node
    'subsample': [0.8, 1.0],                     # Fraction of subsample to train each tree
    'colsample_bytree': [0.8, 1.0],              # Fraction of features when building each tree
    'gamma': [0, 0.1, 0.2],                      # Minimum reduction in loss function to make splitting
    'reg_alpha': [0, 0.1, 0.5],                  # L1 regularization on tree leaf weights
    'reg_lambda': [0, 0.1, 0.5],                 # L2 regularization on tree leaf weights
    'scale_pos_weight': [1, 2, 3]                # Balancing class weights in case of unbalanced data
}
grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```
```python
grid_search.score(X_test, y_test) # accuracy = 0.8006
```
К сожалению, поиск лучшей модели занял много времени (около 10 часов). Но результат стоит этого времени.
## Предсказание
Пройдя через все этапы обработки (для test_data провелась вся та же процедура, что и для train_data) и настройки модели, я перешел к фазе предсказания. Для этого использовал обученную модель с наилучшими параметрами.

```python
predicted_values = best_model.predict(test_data)
```

После выполнения предсказаний, я создал dataframe, где каждому человеку предсказано, какие пассажиры были затронуты аномалией, а какие нет.

```python
predicted_values = pd.Series(predicted_values).map({1:True, 0:False})
predict_data = pd.DataFrame({"PassengerId":passenger_id,"Transported":predicted_values})
predict_data.to_csv("Predict_proba Xgboost.csv", index=False)
```

## Оценка и результаты
Следующим этапом работы стала загрузка `Predict_proba Xgboost.csv` на платформу Kaggle, где я оценил точность своей модели предсказания на тестовой выборке.

![Accuracy](https://github.com/Lapamore/Kaggle_competitions/blob/main/Spaceship%20Titanic/img/XgBoost.png)

После загрузки предсказаний на платформу Kaggle, я получил оценку точности моей модели на основе метрики, предоставленной в задаче. Это позволило мне понять, насколько успешно моя модель обобщает данные и делает предсказания на новых данных.

![Место в таблице](https://github.com/Lapamore/Kaggle_competitions/blob/main/Spaceship%20Titanic/img/XgBoost%20table.jpg)

Этот этап закрывает мой проект и дает понимание того, насколько успешно моя модель справляется с задачей предсказания.

## Заключение
Это соревнование стало для меня вторым в мире Kaggle. С каждой попыткой я старался перегнать свой предыдущий резульат. Я считаю, что достиг хорошего результата для начала своего пути. Спасибо за прочтение!
