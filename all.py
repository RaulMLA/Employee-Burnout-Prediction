import numpy as np
import pandas as pd

import pickle

# Datos disponibles.
with open('attrition_available_13.pkl', 'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame(data)

print(df)

# Mostramos la información del conjunto de datos.
print(f'El conjunto de datos tiene {len(df)} instancias.')

# Datos.
X = df.drop('Attrition', axis=1)

# Etiquetas.
y = df.Attrition

print(y)

# Eliminamos la columna EmployeeID porque no aporta información.
X.drop('EmployeeID', axis=1)

drop_cols = []
for col in X.columns:
    if X[col].nunique() == 1:
        # La columna es constante.
        print(f'Columna {col} es constante y será eliminada.')
        drop_cols.append(col)

    if X[col].isna().sum() == len(data):
        # La columna tiene todos los valores nulos.
        print(f'Columna {col} tiene todos los valores nulos y será eliminada.')
        drop_cols.append(col)

# Eliminar las columnas constantes o innecesarias.
X.drop(drop_cols, axis=1, inplace=True)

X.describe()

# Función que cuenta el número de missing values por atributos,
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_percent_rounded = mis_val_percent.round(2).astype(str) + '%'
    mis_val_table = pd.concat([mis_val, mis_val_percent_rounded], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : 'Proporción'})
    return mis_val_table_ren_columns
  
# Generamos las proporciones de missing values.
missing_values_table(df)

y.describe()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Realizamos una división train/test 'normal'.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/10, random_state=13, shuffle=True)

# Codificamos las etiquetas de las clases
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

N_SAMPLES_NEG = np.sum(y_encoded == 0)
N_SAMPLES_POS = np.sum(y_encoded == 1)

# Ahora comprobamos la distribución de las clases.
n_pos_pct = N_SAMPLES_POS / (N_SAMPLES_POS + N_SAMPLES_NEG)
n_pos_train_pct = np.sum(y_train) / len(y_train)
n_pos_test_pct = np.sum(y_test) / len(y_test)

print(f'Hay un {100 * n_pos_train_pct:.1f} % de instancias positivas en train.')
print(f'Hay un {100 * n_pos_test_pct:.1f} % de instancias positivas en test.')
print(f'Hay un {100 * n_pos_pct:.1f} % de instancias positivas en total.')


# Ahora realizaremos una división train/test estratificada.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/10, random_state=13, shuffle=True, stratify=y)

# Codificamos las etiquetas de las clases
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_train = le.fit_transform(y_train)
y_test= le.transform(y_test)

N_SAMPLES_NEG = np.sum(y_encoded == 0)
N_SAMPLES_POS = np.sum(y_encoded == 1)

# Ahora comprobamos la distribución de las clases.
n_pos_pct = N_SAMPLES_POS / (N_SAMPLES_POS + N_SAMPLES_NEG)
n_pos_train_pct = np.sum(y_train) / len(y_train)
n_pos_test_pct = np.sum(y_test) / len(y_test)

print(f'Hay un {100 * n_pos_train_pct:.1f} % de instancias positivas en train.')
print(f'Hay un {100 * n_pos_test_pct:.1f} % de instancias positivas en test.')
print(f'Hay un {100 * n_pos_pct:.1f} % de instancias positivas en total.')




from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Obtenemos los nombres de las columnas que tienen tipo de dato 'object' o 'category'.
cols_categoricas = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Obtenemos los nombres de las columnas que tienen tipo de dato 'int' o 'float'.
cols_numericas = X.select_dtypes(include=['int', 'float']).columns.tolist()

print('Columnas categóricas: ', cols_categoricas)
print('Columnas numéricas: ', cols_numericas)

# Procesamos valores numéricos.
imputer_num = SimpleImputer(strategy='mean')
pipeline_num = Pipeline(
    steps=[
        ('imputer', imputer_num)
    ]
)

# Procesamos valores categóricos (imputamos con la moda y luego codificamos con one-hot encoding).
imputer_cat = SimpleImputer(strategy='most_frequent')
encoder_cat = OneHotEncoder()
pipeline_cat = Pipeline(
    steps=[
        ('imputer', imputer_cat),
        ('onehot', encoder_cat)
    ]
)

# Pre-procesador 'global'.
# Dependiendo del tipo de columna aplicamos una transformación u otra.
processor = ColumnTransformer(
    transformers=[
        ('num', pipeline_num, cols_numericas),
        ('cat', pipeline_cat, cols_categoricas),
    ]
)

# Realizamos la transformación.
X_train = processor.fit_transform(X_train)
X_test = processor.transform(X_test)

# Convertimos el resultado a un DataFrame de Pandas con los nombres de las columnas correctos.
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

X_train
X_test


from sklearn.dummy import DummyClassifier

# Crear un clasificador dummy que prediga siempre la clase mayoritaria
dummy = DummyClassifier(strategy='most_frequent')

# Entrenar el clasificador
dummy.fit(X_train, y_train)

# Predecir valores para el conjunto de test
y_pred = dummy.predict(X_test)

# Usaremos las métricas balanced_accuracy_score, f1 y matríz de confusión.
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

# Calculamos las métricas.
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
#confusion_matrix = confusion_matrix(y_test, y_pred)

print(f'Balanced accuracy: {balanced_accuracy:.4f}')
print(f'F1: {f1:.4f}')
#print(f'Confusion matrix:\n{confusion_matrix}')


from sklearn.linear_model import LogisticRegression

# Creamos el modelo.
model = LogisticRegression(random_state=13, solver='liblinear', class_weight='balanced')

# Lo entrenamos.
model.fit(X_train, y_train)

# Predecimos sobre el conjunto de test.
y_pred = model.predict(X_test)

# Calculamos las métricas.
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)

print(f'Balanced accuracy: {balanced_accuracy:.4f}')
print(f'F1: {f1:.4f}')
print(f'Confusion matrix:\n{confusion_matrix}')





# Importamos todas las implementaciones de boosting que vamos a probar.
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

# Librerias externas (extra).
from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier

# Importamos time.
import time

# Creamos un diccionario con los modelos que vamos a evaluar.
modelos = {
    'AdaBoost': AdaBoostClassifier(random_state=13),
    'Gradient Boosting': GradientBoostingClassifier(random_state=13),
    'XGBoost': XGBClassifier(random_state=13),
    #'LightGBM': LGBMClassifier(random_state=13, class_weight='balanced'),
}


# Creamos una función para evaluar los modelos.
def evaluar_modelos(models, X_train, y_train, X_test, y_test):
    # Creamos un diccionario para guardar los resultados.
    resultados = {}

    # Iteramos sobre los modelos.
    for nombre_modelo, modelo in models.items():
        print(f'Evaluando modelo: {nombre_modelo}')

        # Entrenamos el modelo.
        start = time.time()
        modelo.fit(X_train, y_train)
        end = time.time()
        tiempo = end - start

        # Predecimos sobre el conjunto de test.
        y_pred = modelo.predict(X_test)

        # Calculamos las métricas.
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        #conf_matrix = confusion_matrix(y_test, y_pred)

        # Guardamos los resultados.
        resultados[nombre_modelo] = {
            'tiempo': tiempo,
            'model': modelo,
            'balanced_accuracy': balanced_accuracy,
            'f1': f1
            #'confusion_matrix': conf_matrix
        }

    return resultados

# Evaluamos los modelos.
resultados = evaluar_modelos(modelos, X_train, y_train, X_test, y_test)

# Mostramos los resultados.
for nombre_modelo, resultado in resultados.items():
    print(f'\nModelo: {nombre_modelo}')
    print(f'Tiempo de ejecución: {resultado["tiempo"]:.5f} segundos')
    print(f'Balanced accuracy: {resultado["balanced_accuracy"]:.4f}')
    print(f'F1: {resultado["f1"]:.4f}')
    #print(f'Confusion matrix:\n{resultado['confusion_matrix']}')





# Importamos la clase StratifiedKFold para realizar validación cruzada estratificada.
from sklearn.model_selection import StratifiedKFold

# Importamos GridSearchCV para realizar búsqueda de hiper-parámetros.
from sklearn.model_selection import GridSearchCV

# StandardScaler para escalar los datos.
from sklearn.preprocessing import StandardScaler

cv = StratifiedKFold(n_splits=5)

# Creamos un diccionario con los parámetros que vamos a probar.
parametros = {
    'AdaBoost': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.1, 0.5, 1.0],
    },
    'Gradient Boosting': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.1, 0.5, 1.0],
        'model__max_depth': [3, 5, 7],
    },
    'XGBoost': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.1, 0.5, 1.0],
        'model__max_depth': [3, 5, 7],
    },
    'LightGBM': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.1, 0.5, 1.0],
        'model__max_depth': [3, 5, 7],
    }
}

modelos_ajustados = {}

# Creamos una función para evaluar y ajustar los modelos.
def ajustar_evaluar_modelos(modelos, parametros, X_train, y_train, X_test, y_test, cv):
    # Creamos un diccionario para guardar los resultados.
    resultados = {}

    # Iteramos sobre los modelos.
    for nombre_modelo, modelo in modelos.items():
        print(f'Evaluando modelo: {nombre_modelo}')

        # Definimos el pipeline.
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', modelo)
        ])

        # Definimos la búsqueda por grid search con validación cruzada.
        grid = GridSearchCV(
            pipeline,
            parametros[nombre_modelo],
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )

        # Entrenamos el modelo.
        start = time.time()
        grid.fit(X_train, y_train)
        end = time.time()
        tiempo = end - start     

        # Predecimos sobre el conjunto de test.
        y_pred = grid.predict(X_test)

        # Calculamos las métricas.
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        #conf_matrix = confusion_matrix(y_test, y_pred)

        # Guardamos los resultados.
        resultados[nombre_modelo] = {
            'tiempo': tiempo,
            'model': grid.best_estimator_,
            'best_params': grid.best_params_,
            'balanced_accuracy': balanced_accuracy,
            'f1': f1
            #'confusion_matrix': conf_matrix
        }

    return resultados

# Evaluamos los modelos con ajuste de hiperparámetros.
resultados_ajuste = ajustar_evaluar_modelos(modelos, parametros, X_train, y_train, X_test, y_test, cv)

# Mostramos los resultados.
for nombre_modelo, resultado in resultados_ajuste.items():
    print(f'\nModelo: {nombre_modelo}')
    print(f'Parámetros: {resultado["best_params"]}')
    print(f'Tiempo de ejecución: {resultado["tiempo"]:.5f} segundos')
    print(f'Balanced accuracy: {resultado["balanced_accuracy"]:.4f}')
    print(f'F1: {resultado["f1"]:.4f}')
    #print(f'Confusion matrix:\n{resultado['confusion_matrix']}')


modelos_ajustados=[AdaBoostClassifier(random_state=13, learning_rate=1.0, n_estimators=200),
                   GradientBoostingClassifier(random_state=13, learning_rate=0.5, max_depth=5, n_estimators=200),
                    XGBClassifier(random_state=13, learning_rate=0.5, max_depth=7, n_estimators=200)#,LGBMClassifier(random_state=13, learning_rate=0.1, max_depth=7, n_estimators=200, class_weight='balanced')
                    ]
# Importamos f_classif, mutual_info_classif y chi2.
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

# Importamos SelectKBest.
from sklearn.feature_selection import SelectKBest

def evaluar_modelos_seleccion_caracteristicas_LR(metrica):
    # Creamos un selector de características usando metrica.
    selector = SelectKBest(score_func=metrica, k=10)

    # Aplicamos el selector sobre los datos de entrenamiento y prueba.
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    # Obtener los puntajes de cada atributo.
    scores = selector.scores_

    # Crear una lista de tuplas que empareje cada nombre de atributo con su puntaje.
    features_scores = list(zip(df.columns, scores))

    # Ordenar la lista de mayor a menor según los puntajes.
    features_scores = sorted(features_scores, key=lambda x: x[1], reverse=True)

    # Imprimir los 5 atributos más importantes según metrica.
    top_features = [feature[0] for feature in features_scores[:5]]
    print("Top 5 atributos más importantes:")
    for feature in top_features:
      print(feature)

    # Creamos un modelo de Regresión Logística y lo entrenamos con las características seleccionadas.
    modelo = LogisticRegression(random_state=13, solver='liblinear', class_weight='balanced')

    start = time.time()
    modelo.fit(X_train_sel, y_train)
    end = time.time()
    tiempo = end - start

    # Evaluamos el modelo con las características seleccionadas.
    y_pred = modelo.predict(X_test_sel)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced accuracy con las características seleccionadas: {balanced_accuracy:.4f}")
    print(f'Tiempo de ejecución: {tiempo:.5f} segundos')

metricas = [f_classif, mutual_info_classif, chi2]

for metrica in metricas:
    print(f'\nMetrica: {metrica.__name__}')
    evaluar_modelos_seleccion_caracteristicas_LR(metrica)


  
def evaluar_modelos_seleccion_caracteristicas_boost(metrica, modelo):
    # Seleccionar el método de selección de atributos.
    selector = SelectKBest(score_func=metrica, k=10)

    # Aplicar el selector sobre los datos de entrenamiento y prueba.
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    # Crear y entrenar un modelo de AdaBoost con las características seleccionadas.
    modelo.fit(X_train_sel, y_train)

    # Evaluar el modelo con las características seleccionadas.
    score = modelo.score(X_test_sel, y_test)

    # Obtener los puntajes de importancia de características y ordenarlos de mayor a menor.
    scores = selector.scores_
    feature_scores = list(zip(df.columns, scores))
    feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)

    # Imprimir los 5 atributos más importantes según f_classif.
    top_features = [feature[0] for feature in feature_scores[:5]]
    print("Top 5 atributos más importantes:")
    for feature in top_features:
      print(feature)

    start = time.time()
    modelo.fit(X_train_sel, y_train)
    end = time.time()
    tiempo = end - start

    # Evaluar el modelo con las características seleccionadas.
    y_pred = modelo.predict(X_test_sel)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced accuracy con las características seleccionadas: {balanced_accuracy:.4f}")
    print(f'Tiempo de ejecución: {tiempo:.5f} segundos')

metricas = [f_classif, mutual_info_classif, chi2]

print("Boosting sin ajustar")
for modelo in modelos.values():
    print(f'\nModelo: {type(modelo).__name__}')
    for metrica in metricas:
        print(f'\nMetrica: {metrica.__name__}')
        evaluar_modelos_seleccion_caracteristicas_boost(metrica, modelo)

print("Boosting ajustado")
for modelo in modelos_ajustados:
    print(f'\nModelo: {type(modelo).__name__}')
    for metrica in metricas:
        print(f'\nMetrica: {metrica.__name__}')
        evaluar_modelos_seleccion_caracteristicas_boost(metrica, modelo)