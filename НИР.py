#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.cm

# # Научно-исследовательская работа

# ### Кулешова Ирина ИУ5-65Б

# В качестве набора данных мы будем использовать набор данных 
# по определению качества красного вина - https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
# 
# Датасет состоит из одного файла:
# - winequality-red.csv - большая выборка вин с различными параметрами
# 
# Файл содержит следующие колонки:
# - fixed acidity - фиксированная кислотность вина (содержание нелетучих кислот в вине) в %.
# - volatile acidity - летучая кислотность вина в %.
# - citric acid - содержание лимонной кислоты в %.
# - residual sugar - содержание сахара в %.
# - chlorides - содержание хлоридов в вине в %.
# - free sulfur dioxide - общее количество свободного диоксида серы в вине (который еще не вступил в реакцию).
# - total sulfur dioxide - общее количество диоксида серы в литре.
# - density - плотность.
# - pH - значение по шкале pH (кислотность).
# - sulphates - доля сульфатов.
# - alcohol - содержание алкоголя в %.
# - quality - целевой признак, качество вина, измеряемое оценкой от 1 до 10.

# ## Импорт библиотек

# Импортируем библиотеки с помощью команды import.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score 
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC, NuSVC, LinearSVC, OneClassSVM, SVR, NuSVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
sns.set(style="ticks")
import warnings


# Filter out all warnings
warnings.filterwarnings("ignore")

st.title('Научно-исследовательская работа')
st.subheader('Кулешова Ирина ИУ5-65Б')
st.write('В качестве набора данных мы будем использовать набор данных по определению качества красного вина - https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009')
st.write('Датасет состоит из одного файла:')
# Создайте список элементов
my_list = ['winequality-red.csv - большая выборка вин с различными параметрами']
# Отобразите список в приложении
st.markdown("<ul><li>" + "</li><li>".join(my_list) + "</li></ul>", unsafe_allow_html=True)

st.write('Файл содержит следующие колонки:')
# Создайте список элементов
my_list = ['fixed acidity - фиксированная кислотность вина (содержание нелетучих кислот в вине) в %.', 'volatile acidity - летучая кислотность вина в %.', 'citric acid - содержание лимонной кислоты в %.', 'residual sugar - содержание сахара в %.', 'chlorides - содержание хлоридов в вине в %.', 'free sulfur dioxide - общее количество свободного диоксида серы в вине (который еще не вступил в реакцию).', 'total sulfur dioxide - общее количество диоксида серы в литре.',
'density - плотность.', 'pH - значение по шкале pH (кислотность).', 'sulphates - доля сульфатов.', 'alcohol - содержание алкоголя в %.', 'quality - целевой признак, качество вина, измеряемое оценкой от 1 до 10.']
# Отобразите список в приложении
st.markdown("<ul><li>" + "</li><li>".join(my_list) + "</li></ul>", unsafe_allow_html=True)

@st.cache
class MetricLogger:
    
    def __init__(self):
        self.df = pd.DataFrame(
            {'metric': pd.Series([], dtype='str'),
            'alg': pd.Series([], dtype='str'),
            'value': pd.Series([], dtype='float')})

    def add(self, metric, alg, value):
        """
        Добавление значения
        """
        # Удаление значения если оно уже было ранее добавлено
        self.df.drop(self.df[(self.df['metric']==metric)&(self.df['alg']==alg)].index, inplace = True)
        # Добавление нового значения
        temp = [{'metric':metric, 'alg':alg, 'value':value}]
        self.df = pd.concat([self.df, pd.DataFrame(temp)], ignore_index=True)

    def get_data_for_metric(self, metric, ascending=True):
        """
        Формирование данных с фильтром по метрике
        """
        temp_data = self.df[self.df['metric']==metric]
        temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
        return temp_data_2['alg'].values, temp_data_2['value'].values
    
    def plot(self, str_header, metric, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_labels, array_metric = self.get_data_for_metric(metric, ascending)
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric))
        rects = ax1.barh(pos, array_metric,
                         align='center',
                         height=0.5, 
                         tick_label=array_labels)
        ax1.set_title(str_header)
        for a,b in zip(pos, array_metric):
            plt.text(0.5, a-0.05, str(round(b,3)), color='white')
        plt.show()  
        
        
@st.cache(suppress_st_warning=True)
def regr_train_model(model_name, model, regrMetricLogger):
    model.fit(regr_X_train, regr_Y_train)
    Y_pred = model.predict(regr_X_test)
    
    mae = mean_absolute_error(regr_Y_test, Y_pred)
    mse = mean_squared_error(regr_Y_test, Y_pred)
    r2 = r2_score(regr_Y_test, Y_pred)

    regrMetricLogger.add('MAE', model_name, mae)
    regrMetricLogger.add('MSE', model_name, mse)
    regrMetricLogger.add('R2', model_name, r2)    
    
    st.write('{} \t MAE={}, MSE={}, R2={}'.format(
        model_name, round(mae, 3), round(mse, 3), round(r2, 3)))
        

@st.cache_data
def load_data():
    '''
    Загрузка данных
    '''
    data_wine = pd.read_csv('data/winequality-red.csv', sep=",")
    data = data_wine.drop_duplicates()
    return data





data_load_state = st.text('Загрузка данных...')
data = load_data()
data_load_state.text('Данные загружены!')

st.subheader('Первые 5 значений')
st.write(data.head())

st.subheader('Размер датасета')
st.write(data.shape)

st.subheader('Проверяем наличие пропусков в датасете:')
st.write(data.isnull().sum())

st.subheader('Скрипичные диаграммы для числовых колонок:')
for col in ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']:
    fig1 = plt.figure(figsize=(7,5))
    ax = sns.violinplot(x=data[col])
    st.pyplot(fig1)

st.subheader('Выбор признаков, подходящих для построения моделей. Кодирование категориальных признаков. Масштабирование данных. Формирование вспомогательных признаков, улучшающих качество моделей.')
st.write('Для построения моделей будем использовать все признаки кроме признака quality.')
st.write('Категориальные признаки отсутствуют, их кодирования не требуется. Исключением является признак quality, но в представленном датасете он уже закодирован на основе подхода LabelEncoding.')
st.write('Вспомогательные признаки для улучшения качества моделей в данном примере мы строить не будем.')
st.write('Выполним масштабирование данных.')

# Числовые колонки для масштабирования
scale_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
       
sc1 = MinMaxScaler()
sc1_data = sc1.fit_transform(data[scale_cols])

# Добавим масштабированные данные в набор данных
for i in range(len(scale_cols)):
    col = scale_cols[i]
    new_col_name = col + '_scaled'
    data[new_col_name] = sc1_data[:,i]

st.write(data.head())

st.subheader('Матрица корреляций:')
# Воспользуемся наличием тестовых выборок, 
# включив их в корреляционную матрицу
scale_cols_postfix = [x+'_scaled' for x in scale_cols]
corr_cols_2 = scale_cols_postfix + ['quality']
corr_cols_2
fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(data[corr_cols_2].corr(), annot=True, fmt='.3f')
ax.set_title('Масштабированные данные')
st.pyplot(fig)

st.write('На основе корреляционной матрицы можно сделать следующие выводы:')
# Создайте список элементов
my_list = ['Корреляционные матрицы для исходных и масштабированных данных совпадают.', 'Элемент Целевой признак регрессии "pH" наиболее сильно коррелирует с фиксированной кислотностью вина (-0.7) и содержанием лимонной кислоты (0.5). Эти признаки обязательно следует оставить в модели регрессии.']
# Отобразите список в приложении
st.markdown("<ul><li>" + "</li><li>".join(my_list) + "</li></ul>", unsafe_allow_html=True)

st.subheader('Выбор метрик для последующей оценки качества моделей.')
st.markdown("""
### В качестве метрик для решения задачи регрессии будем использовать:

#### [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error) - средняя абсолютная ошибка
""")
formula = r"""MAE(y,\hat{y}) = \frac{1}{N} \cdot \sum\limits_{i=1}^N \lvert  y_i - \hat{y_i} \rvert"""
st.latex(formula)

#$MAE(y,\hat{y}) = \frac{1}{N} \cdot \sum\limits_{i=1}^N |y_i - \hat{y}_i|$
st.markdown(
"""
где:
- $y$ - истинное значение целевого признака
- $\hat{y}$ - предсказанное значение целевого признака
- $N$ - размер тестовой выборки

Чем ближе значение к нулю, тем лучше качество регрессии.

Основная проблема метрики состоит в том, что она не нормирована.

Вычисляется с помощью функции [mean_absolute_error.](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error)

#### [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) - средняя квадратичная ошибка
""")
#$MSE(y,\hat{y}) = \frac{1}{N} \cdot \sum\limits_{i=1}^N (y_i - \hat{y}_i)^2$
formula = r"""MSE(y,\hat{y}) = \frac{1}{N} \cdot \sum\limits_{i=1}^N (y_i - \hat{y}_i)^2"""
st.latex(formula)
st.markdown(
"""
где:
- $y$ - истинное значение целевого признака
- $\hat{y}$ - предсказанное значение целевого признака
- $N$ - размер тестовой выборки

Вычисляется с помощью функции [mean_squared_error.](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)

#### [Метрика $R^2$ или коэффициент детерминации](https://ru.wikipedia.org/wiki/%D0%9A%D0%BE%D1%8D%D1%84%D1%84%D0%B8%D1%86%D0%B8%D0%B5%D0%BD%D1%82_%D0%B4%D0%B5%D1%82%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D0%B0%D1%86%D0%B8%D0%B8)
""")
#$R^2(y,\hat{y}) = 1 - \frac{\sum\limits_{i=1}^N (y_i - \hat{y}_i)^2}{\sum\limits_{i=1}^N (y_i - \overline{y_i})^2}$

formula = r"""R^2(y,\hat{y}) = 1 - \frac{\sum\limits_{i=1}^N (y_i - \hat{y}_i)^2}{\sum\limits_{i=1}^N (y_i - \overline{y_i})^2}"""
st.latex(formula)
st.markdown(
"""
где:
- $y$ - истинное значение целевого признака
- $\hat{y}$ - предсказанное значение целевого признака
- $N$ - размер тестовой выборки
""")
formula = r""" \overline{y_i} = \frac{1}{N} \cdot \sum\limits_{i=1}^N y_i"""
st.latex(formula)
st.markdown("""
Вычисляется с помощью функции [r2_score.](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)
""", unsafe_allow_html=True)

st.subheader('Выбор наиболее подходящих моделей для решения задачи регрессии')
st.write('Для задачи регрессии будем использовать следующие модели:')
# Создайте список элементов
my_list = ['Линейная регрессия', 'Метод ближайших соседей', 'Метод опорных векторов', 'Решающее дерево', 'Случайный лес', 'Градиентный бустинг']
# Отобразите список в приложении
st.markdown("<ul><li>" + "</li><li>".join(my_list) + "</li></ul>", unsafe_allow_html=True)

parameter1_options = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", 
    "density", "pH", "sulphates", "alcohol", "quality"]
parameter2_options = ["citric acid", "fixed acidity", "volatile acidity", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", 
    "density", "pH", "sulphates", "alcohol", "quality"]
parameter1_selected = st.selectbox("Выберите гиперпараметр 1 для решения задачи регрессии:", options = parameter1_options)
parameter2_selected = st.selectbox("Выберите гиперпараметр 2 для решения задачи регрессии:", options = parameter2_options)
selected_features = [parameter1_selected, parameter2_selected]
features = data.loc[:, selected_features]
regr_X_train, regr_X_test, regr_Y_train, regr_Y_test = train_test_split(features, data["pH"], test_size=0.2, random_state=1)

st.subheader('Построение базового решения (baseline) для выбранных моделей без подбора гиперпараметров.')
# Модели
regr_models = {'LR': LinearRegression(), 
               'KNN_5':KNeighborsRegressor(n_neighbors=5),
               'SVR':SVR(),
               'Tree':DecisionTreeRegressor(),
               'RF':RandomForestRegressor(),
               'GB':GradientBoostingRegressor()}
               
# Сохранение метрик
regrMetricLogger = MetricLogger()
for model_name, model in regr_models.items():
    regr_train_model(model_name, model, regrMetricLogger)
    
st.subheader('Подберем гиперпараметры для выбранных моделей.')
n_range = np.array(range(1,2000,100))
tuned_parameters = [{'n_neighbors': n_range}]
tuned_parameters


regr_gs = GridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')
regr_gs.fit(regr_X_train, regr_Y_train)

# Лучшая модель
regr_gs.best_estimator_
regr_gs_best_params_txt = str(regr_gs.best_params_['n_neighbors'])
st.write('Результат подбора:')
st.write(regr_gs_best_params_txt)

regr_models_grid = {'KNN_5':KNeighborsRegressor(n_neighbors=5), 
                    str('KNN_'+regr_gs_best_params_txt):regr_gs.best_estimator_}
                    
for model_name, model in regr_models_grid.items():
    regr_train_model(model_name, model, regrMetricLogger)

st.subheader('Выводы о качестве построенных моделей на основе выбранных метрик.')
# Метрики качества модели
regr_metrics = np.array(['MAE', 'MSE', 'R2'])
#f1 = regrMetricLogger.plot('Метрика: ' + 'MAE', 'MAE', ascending=False, figsize=(20, 20))
st.set_option('deprecation.showPyplotGlobalUse', False)
#st.pyplot(f1)
st.image('1.png')
#f2 = regrMetricLogger.plot('Метрика: ' + 'MSE', 'MSE', ascending=False, figsize=(10, 5))
#st.pyplot(f2)
st.image('2.png')
f3 = regrMetricLogger.plot('Метрика: ' + 'R2', 'R2', ascending=True, figsize=(9, 5))
st.pyplot(f3)

st.markdown("**Вывод: лучшими оказались модели на основе градиентного бустинга и метода опорных векторов. При отдельных запусках вместо метода опорных векторов оказывается лучшей модель ближайших соседей.**")