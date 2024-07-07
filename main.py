import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name("key.json", scope)
client = gspread.authorize(creds)

spreadsheet = client.open('Python Features')
sheet = spreadsheet.worksheet('features')

existing_data = pd.DataFrame(sheet.get_all_records())
# print(existing_data.columns)

for column in ['MarkDown1', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
    existing_data[column] = pd.to_numeric(existing_data[column], errors='coerce')
    
for column in ['MarkDown1', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
    existing_data[column] = existing_data[column].fillna(existing_data[column].median())

existing_data['CPI'] = pd.to_numeric(existing_data['CPI'], errors='coerce')
existing_data['CPI'] = existing_data['CPI'].fillna(existing_data['CPI'].mean())

existing_data['Unemployment'] = pd.to_numeric(existing_data['Unemployment'], errors='coerce')
existing_data['Unemployment'] = existing_data['Unemployment'].fillna(existing_data['Unemployment'].mean())

clean_data_list = existing_data.to_dict(orient = 'records')
update_cells = []

#Insertar osnombres de las columnas com primera fila
update_cells.append(gspread.models.Cell(1,1,"Store"))
update_cells.append(gspread.models.Cell(1,2,"Date"))
update_cells.append(gspread.models.Cell(1,3,"Temperature"))
update_cells.append(gspread.models.Cell(1,4,"Fuel_Price"))
update_cells.append(gspread.models.Cell(1,5,"MarkDown1"))
update_cells.append(gspread.models.Cell(1,6,"MarkDown2"))
update_cells.append(gspread.models.Cell(1,7,"MarkDown3"))
update_cells.append(gspread.models.Cell(1,8,"MarkDown4"))
update_cells.append(gspread.models.Cell(1,9,"MarkDown5"))
update_cells.append(gspread.models.Cell(1,"CPI"))
update_cells.append(gspread.models.Cell(1,"Unemployment"))

for i, entry in enumerate(clean_data_list):
    for key, value in entry.items():
        update_cells.append(gspread.models.Cell(i+2, existing_data.columns.get_loc(key)+1, value))
sheet.clear() #borrar contenido existente en la hoja de trabajo
sheet.update_cell(update_cells) #actualizar datos

print("Los datos limpios han sido guardados exitosamente en Google Sheets.")

# Los otros archivos estan trabajando desde un csv directo en la raiz de este proyecto, esto se
# debe a que nuestro equipo tiene una capacidad limitada y otras tareas mas complejas
# exigen menos recursos al trabajar desde el csv, pero en otras condicines puede ser mas conveniente
# trabajar todo desde la API del proyecto en google sheets u otras plataformas.