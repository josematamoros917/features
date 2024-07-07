import pandas as pd

# Cargar el archivo CSV 'sales-data-set.csv'
sales_data = pd.read_csv('./csv/sales-data-set.csv')

# Formatear 'Weekly_Sales' a dos decimales
sales_data['Weekly_Sales'] = sales_data['Weekly_Sales'].round(2)

# Convertir 'IsHoliday' de TRUE/FALSE a 1/0
sales_data['IsHoliday'] = sales_data['IsHoliday'].replace({True: 1, False: 0}).astype(int)

# Guardar los datos modificados de vuelta al archivo CSV
sales_data.to_csv('./csv_cleaned/sales-data-set-cleaned.csv', index=False)

print("Los datos han sido limpiados y guardados exitosamente en 'sales-data-set-cleaned.csv'.")
