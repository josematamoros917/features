import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#cargar los datos
features_data = pd.read_csv('./csv_cleaned/features-data-set-cleaned.csv')
sales_data = pd.read_csv('./csv_cleaned/sales-data-set-cleaned.csv')
stores_data = pd.read_csv('./csv_cleaned/stores-data-set-cleaned.csv')
#cambiar formato de fechas a un formato que entienda pandas, indicando el dia primero
features_data['Date'] = pd.to_datetime(features_data['Date'], dayfirst=True)
sales_data['Date'] = pd.to_datetime(sales_data['Date'], dayfirst=True)
#crear graficos de dstribucion
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = int(np.ceil(nCol / nGraphPerRow))
    plt.figure(figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80)
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist(bins=50)
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()

def plotCorrelationMatrix(df, graphWidth):
    df = df.dropna(axis='columns')
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(figsize=(graphWidth, graphWidth), dpi=80)
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title('Correlation Matrix', fontsize=15)
    plt.show()

def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include=[np.number])
    df = df.dropna(axis='columns')
    df = df[[col for col in df if df[col].nunique() > 1]]  # Corregido el filtro de columnas
    columnNames = list(df)
    if len(columnNames) > 10:
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

print("Exploratory Data Analysis")
plotPerColumnDistribution(features_data, 10, 5)
plotPerColumnDistribution(sales_data, 10, 5)
plotPerColumnDistribution(stores_data, 10, 5)

plotCorrelationMatrix(features_data, 8)
plotCorrelationMatrix(sales_data, 8)
plotCorrelationMatrix(stores_data, 8)

plotScatterMatrix(features_data, 9, 10)
plotScatterMatrix(sales_data, 9, 10)
plotScatterMatrix(stores_data, 6, 15)

# Unir los datos para análisis
merged_data = sales_data.merge(features_data, on=['Store', 'Date', 'IsHoliday'], how='left')
merged_data = merged_data.merge(stores_data, on='Store', how='left')

# Limitar a 500 filas para el entrenamiento del modelo
merged_data_sample = merged_data.sample(n=500, random_state=42)

# Añadir la característica 'Temperature_Fuel_Price'
merged_data_sample['Temperature_Fuel_Price'] = merged_data_sample['Temperature'] * merged_data_sample['Fuel_Price']

# Añadir las columnas dummy para 'Type'
merged_data_sample['Type_A'] = (merged_data_sample['Type'] == 'A').astype(int)
merged_data_sample['Type_B'] = (merged_data_sample['Type'] == 'B').astype(int)
merged_data_sample['Type_C'] = (merged_data_sample['Type'] == 'C').astype(int)

# Convertir la columna 'IsHoliday' a numérico
merged_data_sample['IsHoliday'] = merged_data_sample['IsHoliday'].astype(int)

# Crear características adicionales
merged_data_sample['WeekOfYear'] = merged_data_sample['Date'].dt.isocalendar().week

# Dividir los datos en conjuntos de entrenamiento y prueba
X = merged_data_sample.drop(['Weekly_Sales', 'Date', 'Type'], axis=1)  # Eliminar la columna original 'Type'
y = merged_data_sample['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajuste de Hiperparámetros con GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42, n_jobs=-1), 
                           param_grid=param_grid, 
                           cv=5, 
                           scoring='neg_mean_squared_error',
                           n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Hacer predicciones y evaluar el modelo
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio: {mse}')

# Validación cruzada para evaluar el modelo de manera más robusta
scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
mse_cv = -scores.mean()  # Error cuadrático medio promedio de la validación cruzada
print(f'Error cuadrático medio promedio (validación cruzada): {mse_cv}')

# Analizar los efectos de los descuentos en semanas festivas
holiday_weeks = merged_data_sample[merged_data_sample['IsHoliday'] == 1]['WeekOfYear'].unique()
merged_data_sample['IsHolidayWeek'] = merged_data_sample['WeekOfYear'].isin(holiday_weeks).astype(int)

# Modelo de efectos de descuentos en semanas festivas
features = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'IsHolidayWeek']
X = merged_data_sample[features]
y = merged_data_sample['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio en semanas festivas: {mse}')

# Recomendaciones basadas en los insights obtenidos
print("Recomendaciones:")
print("- Ajustar los descuentos durante las semanas festivas para maximizar las ventas.")
print("- Optimizar la asignación de inventario según las previsiones de ventas.")
print("- Considerar factores adicionales como el clima y el precio del combustible en las estrategias de marketing.")

