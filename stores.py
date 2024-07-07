# import
# pandas as pd

# # Cargar el archivo CSV 'stores-data-set.csv'
# stores_data = pd.read_csv('./stores-data-set.csv')

# # Convertir la columna 'Type' en variables dummy
# stores_data = pd.get_dummies(stores_data, columns=['Type'])

# # Guardar los datos modificados de vuelta al archivo CSV
# stores_data.to_csv('./stores-data-set-cleaned.csv', index=False)

# print("Los datos han sido limpiados y guardados exitosamente en 'stores-data-set-cleaned.csv'.")
import pandas as pd

# Cargar el archivo CSV 'stores-data-set.csv'
stores_data = pd.read_csv('./csv/stores-data-set.csv')

# Convertir la columna 'Type' a valores numéricos (0 o 1)
stores_data['Type_A'] = (stores_data['Type'] == 'A').astype(int)
stores_data['Type_B'] = (stores_data['Type'] == 'B').astype(int)
stores_data['Type_C'] = (stores_data['Type'] == 'C').astype(int)

# Eliminar la columna original 'Type'
stores_data.drop(columns=['Type'], inplace=True)

# Guardar los datos modificados de vuelta al archivo CSV
stores_data.to_csv('./csv_cleaned/stores-data-set-cleaned.csv', index=False)

print("Los datos han sido limpiados y guardados exitosamente en 'stores-data-set-cleaned.csv'.")
