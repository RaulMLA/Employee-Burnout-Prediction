import numpy as np
import pandas as pd

import pickle

# Datos disponibles.
with open('attrition_available_13.pkl', 'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame(data)

print(df)

# Mostramos la información del conjunto de datos.
print(f"El conjunto de datos tiene {len(df)} instancias.")

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

# Realizamos una división train/test "normal".
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/10, random_state=13, shuffle=True)

# Codificamos las etiquetas de las clases
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

N_SAMPLES_NEG = np.sum(y_encoded == 0)
N_SAMPLES_POS = np.sum(y_encoded == 1)

# Ahora comprobamos la distribución de las clases.
n_pos_pct = N_SAMPLES_POS / (N_SAMPLES_POS + N_SAMPLES_NEG)
n_pos_train_pct = np.sum(y_train_encoded) / len(y_train_encoded)
n_pos_test_pct = np.sum(y_test_encoded) / len(y_test_encoded)

print(f"Hay un {100 * n_pos_train_pct:.1f} % de instancias positivas en train.")
print(f"Hay un {100 * n_pos_test_pct:.1f} % de instancias positivas en test.")
print(f"Hay un {100 * n_pos_pct:.1f} % de instancias positivas en total.")


# Ahora realizaremos una división train/test estratificada.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/10, random_state=13, shuffle=True, stratify=y)

# Codificamos las etiquetas de las clases
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

N_SAMPLES_NEG = np.sum(y_encoded == 0)
N_SAMPLES_POS = np.sum(y_encoded == 1)

# Ahora comprobamos la distribución de las clases.
n_pos_pct = N_SAMPLES_POS / (N_SAMPLES_POS + N_SAMPLES_NEG)
n_pos_train_pct = np.sum(y_train_encoded) / len(y_train_encoded)
n_pos_test_pct = np.sum(y_test_encoded) / len(y_test_encoded)

print(f"Hay un {100 * n_pos_train_pct:.1f} % de instancias positivas en train.")
print(f"Hay un {100 * n_pos_test_pct:.1f} % de instancias positivas en test.")
print(f"Hay un {100 * n_pos_pct:.1f} % de instancias positivas en total.")




from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Obtenemos los nombres de las columnas que tienen tipo de dato 'object' o 'category'.
cols_categoricas = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Obtenemos los nombres de las columnas que tienen tipo de dato 'int' o 'float'.
cols_numericas = X.select_dtypes(include=['int', 'float']).columns.tolist()

print("Columnas categóricas: ", cols_categoricas)
print("Columnas numéricas: ", cols_numericas)

# Procesamos valores numéricos.
imputer_num = SimpleImputer(strategy='mean')
pipeline_num = Pipeline(
    steps=[
        ("imputer", imputer_num)
    ]
)

# Procesamos valores categóricos (imputamos con la moda y luego codificamos con one-hot encoding).
imputer_cat = SimpleImputer(strategy='most_frequent')
encoder_cat = OneHotEncoder()
pipeline_cat = Pipeline(
    steps=[
        ("imputer", imputer_cat),
        ("onehot", encoder_cat)
    ]
)

# Pre-procesador "global".
# Dependiendo del tipo de columna aplicamos una transformación u otra.
processor = ColumnTransformer(
    transformers=[
        ("num", pipeline_num, cols_numericas),
        ("cat", pipeline_cat, cols_categoricas),
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


from sklearn.linear_model import LogisticRegression

# Creamos el modelo.
model = LogisticRegression(random_state=13, solver='liblinear')

# Lo entrenamos.
model.fit(X_train, y_train_encoded)

# Predecimos sobre el conjunto de test.
y_pred = model.predict(X_test)

# Usaremos las métricas balanced_accuracy_score, f1 y matríz de confusión.
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

# Calculamos las métricas.
balanced_accuracy = balanced_accuracy_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred)
confusion_matrix = confusion_matrix(y_test_encoded, y_pred)

print(f"Balanced accuracy: {balanced_accuracy:.4f}")
print(f"F1: {f1:.4f}")
print(f"Confusion matrix:\n{confusion_matrix}")

