import pandas as pd

# Cargar datos desde el archivo CSV
features_data = pd.read_csv('./csv/features-data-set.csv')

# Convertir las fechas al formato datetime
features_data['Date'] = pd.to_datetime(features_data['Date'], dayfirst=True)

# Convertir 'IsHoliday' de TRUE/FALSE a 1/0
features_data['IsHoliday'] = features_data['IsHoliday'].replace({True: 1, False: 0}).infer_objects()

# Reemplazar los valores 'NA' con la mediana de cada columna
numerical_columns = ['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']

# Calcular la mediana de cada columna numérica y reemplazar los valores 'NA'
for column in numerical_columns:
    median = features_data[column].median()
    features_data[column].fillna(median, inplace=True)

# Asegurar que los datos numéricos tienen el formato adecuado (dos decimales)
features_data[numerical_columns] = features_data[numerical_columns].apply(lambda x: x.round(2))

# Guardar los datos limpios en un nuevo archivo CSV
cleaned_data = './csv_cleaned/features-data-set-cleaned.csv'
features_data.to_csv(cleaned_data, index=False)

print("Datos limpiados y guardados en", cleaned_data)




