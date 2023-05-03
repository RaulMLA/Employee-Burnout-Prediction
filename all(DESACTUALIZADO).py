import numpy as np
import pandas as pd

import time

import pickle
from rich import print


from sklearn.preprocessing import MinMaxScaler


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
start = time.time()
dummy.fit(X_train, y_train)
end = time.time()
tiempo_dummy = end - start

# Predecir valores para el conjunto de test
y_pred = dummy.predict(X_test)

# Usaremos las métricas balanced_accuracy_score, f1 y matríz de confusión.
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

# Calculamos las métricas.
balanced_accuracy_dummy = balanced_accuracy_score(y_test, y_pred)
f1_dummy = f1_score(y_test, y_pred)
confusion_matrix_dummy = confusion_matrix(y_test, y_pred)

print('\n[bold yellow]Dummy classifier (most frequent class)[/bold yellow]')
print(f'Tiempo de entrenamiento: {tiempo_dummy:.4f} segundos')
print(f'Balanced accuracy: {balanced_accuracy_dummy:.4f}')
print(f'F1: {f1_dummy:.4f}')
print(f'Confusion matrix:\n{confusion_matrix_dummy}')





from sklearn.linear_model import LogisticRegression

# Creamos el modelo.
model = LogisticRegression(random_state=13, solver='liblinear', class_weight='balanced')

# Lo entrenamos.
start = time.time()
model.fit(X_train, y_train)
end = time.time()
tiempo = end - start

# Predecimos sobre el conjunto de test.
y_pred = model.predict(X_test)

# Calculamos las métricas.
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print('\n[bold yellow]Logistic regression[/bold yellow]')
print(f'Tiempo de entrenamiento: {tiempo:.4f} segundos.')
print(f'Balanced accuracy: {balanced_accuracy:.4f}')
print(f'F1: {f1:.4f}')
print(f'Confusion matrix:\n{conf_matrix}')





# Speedup de LogisticRegression vs DummyClassifier.
print('\n[bold yellow]Ratios entre LogisticRegression y DummyClassifier[/bold yellow]')
print(f'Ratio Tiempo de entrenamiento LogisticRegression / dummy: {tiempo_dummy / tiempo:.4f}')
print(f'Ratio Balanced accuracy LogisticRegression / dummy: {balanced_accuracy_dummy / balanced_accuracy:.4f}')
print(f'Ratio F1 LogisticRegression / dummy: {f1_dummy / f1:.4f}')
print(f'Ratio Confusion matrix LogisticRegression / dummy:\n{confusion_matrix_dummy / conf_matrix}')





# Importamos todas las implementaciones de boosting que vamos a probar.
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

# Librerias externas (extra).
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Importamos time.
import time

# calcular los pesos de muestra personalizados para el conjunto de entrenamiento
class_counts = np.bincount(y_train)
sample_weights = np.zeros(len(y_train))
for i, count in enumerate(class_counts):
    sample_weights[y_train == i] = class_counts.sum() / (len(class_counts) * count)

# Calcular los pesos de las muestras
weights = np.where(y_train == 0, 1 / np.sum(y_train == 0), 1 / np.sum(y_train == 1))


# Calcular la proporción de clases
print(np.sum(y_train == 0))
proporcion = np.sum(y_train == 0) / np.sum(y_train == 1)


# Creamos un diccionario con los modelos que vamos a evaluar.
modelos = {
    'AdaBoost': AdaBoostClassifier(random_state=13),
    'Gradient Boosting': GradientBoostingClassifier(random_state=13),
    'XGBoost': XGBClassifier(random_state=13),
    'LightGBM': LGBMClassifier(random_state=13, class_weight='balanced'),
}


# Creamos una función para evaluar los modelos.
def evaluar_modelos(models, X_train, y_train, X_test, y_test):
    # Creamos un diccionario para guardar los resultados.
    resultados = {}

    # Iteramos sobre los modelos.
    for nombre_modelo, modelo in models.items():
        #print(f'Evaluando modelo: {nombre_modelo}')

        # Entrenamos el modelo.
        if nombre_modelo == 'AdaBoost':
            start = time.time()
            modelo.fit(X_train, y_train, sample_weight=sample_weights)
            end = time.time()
        elif nombre_modelo == 'Gradient Boosting':
            start = time.time()
            modelo.fit(X_train, y_train, sample_weight=weights)
            end = time.time()
        else:
            start = time.time()
            modelo.fit(X_train, y_train)
            end = time.time()

        tiempo = end - start

        # Predecimos sobre el conjunto de test.
        y_pred = modelo.predict(X_test)

        # Calculamos las métricas.
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Guardamos los resultados.
        resultados[nombre_modelo] = {
            'tiempo': tiempo,
            'model': modelo,
            'balanced_accuracy': balanced_accuracy,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }

    return resultados

print('\n[bold yellow]Boosting sin ajuste de hiper-parámetros[/bold yellow]')

# Evaluamos los modelos.
resultados = evaluar_modelos(modelos, X_train, y_train, X_test, y_test)

# Mostramos los resultados.
for nombre_modelo, resultado in resultados.items():
    print(f'[blue]Modelo: {nombre_modelo}[/blue]')
    print(f'Tiempo de ejecución: {resultado["tiempo"]:.5f} segundos')
    print(f'Balanced accuracy: {resultado["balanced_accuracy"]:.4f}')
    print(f'F1: {resultado["f1"]:.4f}')
    print(f'Confusion matrix:\n{resultado["confusion_matrix"]}\n')





# Speedup del modelo vs dummy.
for nombre_modelo, resultado in resultados.items():
    print(f'\nRatio Tiempo de entrenamiento {nombre_modelo} / dummy: {resultado["tiempo"] / tiempo_dummy:.4f}')
    print(f'Ratio Balanced accuracy {nombre_modelo} / dummy: {resultado["balanced_accuracy"] / balanced_accuracy_dummy:.4f}')


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
        #print(f'Evaluando modelo: {nombre_modelo}')

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
        if nombre_modelo == 'AdaBoost':
            start = time.time()
            grid.fit(X_train, y_train, model__sample_weight=sample_weights)
            end = time.time()
        elif nombre_modelo == 'Gradient Boosting':
            start = time.time()
            grid.fit(X_train, y_train, model__sample_weight=weights)
            end = time.time()
        else:
            start = time.time()
            grid.fit(X_train, y_train)
            end = time.time()

        tiempo = end - start     

        # Predecimos sobre el conjunto de test.
        y_pred = grid.predict(X_test)

        # Calculamos las métricas.
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Guardamos los resultados.
        resultados[nombre_modelo] = {
            'tiempo': tiempo,
            'model': grid.best_estimator_,
            'best_params': grid.best_params_,
            'balanced_accuracy': balanced_accuracy,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }

    return resultados

print('\n[bold yellow]Boosting con ajuste de hiper-parámetros[/bold yellow]')

# Evaluamos los modelos con ajuste de hiperparámetros.
resultados_ajuste = ajustar_evaluar_modelos(modelos, parametros, X_train, y_train, X_test, y_test, cv)

# Mostramos los resultados.
for nombre_modelo, resultado in resultados_ajuste.items():
    print(f'[blue]Modelo: {nombre_modelo}[/blue]')
    print(f'Parámetros: {resultado["best_params"]}')
    print(f'Tiempo de ejecución: {resultado["tiempo"]:.5f} segundos')
    print(f'Balanced accuracy: {resultado["balanced_accuracy"]:.4f}')
    print(f'F1: {resultado["f1"]:.4f}')
    print(f'Confusion matrix:\n{resultado["confusion_matrix"]}\n')



# Speedup del modelo vs dummy.
for nombre_modelo, resultado in resultados_ajuste.items():
    print(f'\nRatio Tiempo de entrenamiento {nombre_modelo} / dummy: {resultado["tiempo"] / tiempo_dummy:.4f}')
    print(f'Ratio Balanced accuracy {nombre_modelo} / dummy: {resultado["balanced_accuracy"] / balanced_accuracy_dummy:.4f}')

# Speedup del modelo ajustado y sin ajustar.
for nombre_modelo, resultado in resultados_ajuste.items():
    print(f'\nRatio Tiempo de entrenamiento {nombre_modelo} / {nombre_modelo} sin ajustar: {resultado["tiempo"] / resultados[nombre_modelo]["tiempo"]:.4f}')
    print(f'Ratio Balanced accuracy {nombre_modelo} / {nombre_modelo} sin ajustar: {resultado["balanced_accuracy"] / resultados[nombre_modelo]["balanced_accuracy"]:.4f}')
    print(f'Ratio F1 {nombre_modelo} / {nombre_modelo} sin ajustar: {resultado["f1"] / resultados[nombre_modelo]["f1"]:.4f}')
    print(f'Ratio Confusion matrix {nombre_modelo} / {nombre_modelo} sin ajustar:\n{resultado["confusion_matrix"] / resultados[nombre_modelo]["confusion_matrix"]}\n')



modelos_ajustados=[AdaBoostClassifier(random_state=13, learning_rate=1.0, n_estimators=200),
                   GradientBoostingClassifier(random_state=13, learning_rate=0.5, max_depth=5, n_estimators=200),
                    XGBClassifier(random_state=13, learning_rate=0.5, max_depth=7, n_estimators=200),
                    LGBMClassifier(random_state=13, learning_rate=0.1, max_depth=7, n_estimators=200, class_weight='balanced')
                  ]






# Importamos f_classif, mutual_info_classif y chi2.
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

# Importamos SelectKBest.
from sklearn.feature_selection import SelectKBest


selector = SelectKBest(score_func=f_classif, k=25)
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
print("[green]Top 5 atributos más importantes:[/green]")
for feature in top_features:
    print(feature)


# Creamos un modelo de Regresión Logística y lo entrenamos con las características seleccionadas.
modelo = LogisticRegression(random_state=13, solver='liblinear', class_weight='balanced')

start = time.time()
modelo.fit(X_train_sel, y_train)
end = time.time()
tiempo_r = end - start

# Evaluamos el modelo con las características seleccionadas.
y_pred = modelo.predict(X_test_sel)
balanced_accuracy_r = balanced_accuracy_score(y_test, y_pred)
print(f"\n\nBalanced accuracy con las características seleccionadas: {balanced_accuracy:.4f}")
print(f'Tiempo de ejecución: {tiempo:.5f} segundos')

#ratio de tiempo de entrenamiento
print(f'\nRatio Tiempo de entrenamiento sin reducir / reducido: {tiempo / tiempo_r:.4f}')

#rattio de balanced accuracy
print(f'\nRatio Balanced accuracy reducido / sin reducir: {balanced_accuracy_r / balanced_accuracy:.4f}')

print('\n[bold yellow]Mejora de resultados LogisticRegression[/bold yellow]')

metricas = [f_classif, mutual_info_classif, chi2]


modelos_ajustados=[AdaBoostClassifier(random_state=13, learning_rate=0.5, n_estimators=200),
                   GradientBoostingClassifier(random_state=13, learning_rate=0.5, max_depth=5, n_estimators=200),
                    XGBClassifier(random_state=13, scale_pos_weight=proporcion),
                    LGBMClassifier(random_state=13, learning_rate=0.1, max_depth=7, n_estimators=200, class_weight='balanced')
                  ]



def evaluar_modelos_seleccion_caracteristicas(modelo, metricas):

    metrica_seleccionada = "f_classif"
    k = 30
    tuneado_seleccion = None
        
    for metrica in metricas:
        selector = SelectKBest(score_func=metrica)
        
        pipe_selector = Pipeline([
            ('scale', MinMaxScaler()),
            ('select', selector),
            ('model', modelo)
        ])

        param_grid = {"select__k": list(range(1, 46))}
        inner = StratifiedKFold(n_splits=5)
        tuneado = GridSearchCV(pipe_selector, param_grid, cv=inner, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
        
        if modelo.__class__.__name__ == "AdaBoostClassifier":
            tuneado.fit(X_train, y_train, model__sample_weight=sample_weights)
        elif modelo.__class__.__name__ == "GradientBoostingClassifier":
            tuneado.fit(X_train, y_train, model__sample_weight=weights)
        else:
            tuneado.fit(X_train, y_train)

        if tuneado.best_params_['select__k'] < k:
            k = tuneado.best_params_['select__k']
            metrica_seleccionada = metrica.__name__
            tuneado_seleccion = tuneado

    print(modelo.__class__.__name__)
    print(f"Mejor métrica: {metrica_seleccionada}")
    print(f"Numero de atributos seleccionados: {k}")

    import matplotlib.pyplot as plt
    plt.plot(tuneado_seleccion.cv_results_['param_select__k'].data, tuneado_seleccion.cv_results_['mean_test_score'])
    plt.ylabel('Balanced accuracy')
    plt.xlabel('Number of features')
    plt.show()

print('\n[bold yellow]Mejora de resultados LogisticRegression[/bold yellow]')

metricas = [f_classif, mutual_info_classif, chi2]

for modelo in modelos_ajustados:
    evaluar_modelos_seleccion_caracteristicas(modelo, metricas)





  
def atributos_mas_importantes_boosting(modelo, metrica, k):
    selector = SelectKBest(score_func=metrica, k=k)

    # Aplicar el selector sobre los datos de entrenamiento y prueba.
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    # Crear y entrenar un modelo con las características seleccionadas.

    if modelo.__class__.__name__ == "AdaBoostClassifier":
        modelo.fit(X_train, y_train, sample_weight=sample_weights)
    elif modelo.__class__.__name__ == "GradientBoostingClassifier":
        modelo.fit(X_train_sel, y_train, sample_weight=weights)
    else:
        modelo.fit(X_train_sel, y_train)

    # Evaluar el modelo con las características seleccionadas.
    score = modelo.score(X_test_sel, y_test)

    # Obtener los puntajes de importancia de características y ordenarlos de mayor a menor.
    scores = selector.scores_
    feature_scores = list(zip(X_train.columns, scores))
    feature_scores = list(zip(df.columns, scores))
    feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)

    # Imprimir los 5 atributos más importantes según f_classif.
    top_features = [feature[0] for feature in feature_scores[:5]]
    print("[green]Top 5 atributos más importantes:[/green]")
    for feature in top_features:
      print(feature)

    start = time.time()
    modelo.fit(X_train_sel, y_train)
    end = time.time()
    tiempo = end - start

    # Evaluar el modelo con las características seleccionadas.
    y_pred = modelo.predict(X_test_sel)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print(f"\n\nBalanced accuracy con las características seleccionadas: {balanced_accuracy:.4f}")
    print(f'Tiempo de ejecución: {tiempo:.5f} segundos')

    return balanced_accuracy, tiempo


ada = atributos_mas_importantes_boosting(AdaBoostClassifier(random_state=13, learning_rate=0.5, n_estimators=200), f_classif, 30)
gb = atributos_mas_importantes_boosting(GradientBoostingClassifier(random_state=13, learning_rate=0.5, max_depth=5, n_estimators=200), f_classif, 30)
xgb = atributos_mas_importantes_boosting(XGBClassifier(random_state=13, scale_pos_weight=proporcion), f_classif, 30)
lgbm = atributos_mas_importantes_boosting(LGBMClassifier(random_state=13, learning_rate=0.1, max_depth=7, n_estimators=200, class_weight='balanced'), f_classif, 30)


#impresion de resultados
print("AdaBoostClassifier")
print(f"Balanced accuracy: {ada[0]:.4f}")
print(f"Tiempo de ejecución: {ada[1]:.5f} segundos")

print("\nGradientBoostingClassifier")
print(f"Balanced accuracy: {gb[0]:.4f}")
print(f"Tiempo de ejecución: {gb[1]:.5f} segundos")

print("\nXGBClassifier")
print(f"Balanced accuracy: {xgb[0]:.4f}")
print(f"Tiempo de ejecución: {xgb[1]:.5f} segundos")

print("\nLGBMClassifier")
print(f"Balanced accuracy: {lgbm[0]:.4f}")
print(f"Tiempo de ejecución: {lgbm[1]:.5f} segundos")

#ratios de precision y tiempo sin y reducido
print("\n\n[bold yellow]Ratios de precision y tiempo sin y reducido[/bold yellow]")
#usar los resultados de resultados_ajuste
print("AdaBoostClassifier")
#ratio
print(f"Ratio: {resultados_ajuste['AdaBoost']['tiempo']/ada[0]:.4f}")
print(f"Ratio balance accuracy sin reducir / reducido: {resultados_ajuste['AdaBoost']['balanced_accuracy']/ada[1]:.4f}")

print("\nGradientBoostingClassifier")
#ratio
print(f"Ratio: {resultados_ajuste['GradientBoosting']['tiempo']/gb[0]:.4f}")
print(f"Ratio: {resultados_ajuste['GradientBoosting']['balanced_accuracy']/gb[1]:.4f}")

print("\nXGBClassifier")
#ratio
print(f"Ratio: {resultados_ajuste['XGB']['tiempo']/xgb[0]:.4f}")
print(f"Ratio: {resultados_ajuste['XGB']['balanced_accuracy']/xgb[1]:.4f}")

print("\nLGBMClassifier")
#ratio
print(f"Ratio: {resultados_ajuste['LGBM']['tiempo']/lgbm[0]:.4f}")
print(f"Ratio: {resultados_ajuste['LGBM']['balanced_accuracy']/lgbm[1]:.4f}")


