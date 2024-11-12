import os
import pandas as pd
import kagglehub
import sqlite3

# Extract, Download from kaggle
path = kagglehub.dataset_download("fbi-us/california-crime")
print("Saving Path:", path)

# Read
crime_path = os.path.join(path, 'ca_offenses_by_city.csv')
enforcement_path = os.path.join(path, 'ca_law_enforcement_by_city.csv')
c_data = pd.read_csv(crime_path)
e_data = pd.read_csv(enforcement_path)

# Transform
c_data = c_data.drop(columns=['Rape (legacy definition)', 'Population'])
c_data['City'] = c_data['City'].str.rstrip('3')
e_data.columns = e_data.columns.str.replace('\r', ' ', regex=False)
e_data.columns = e_data.columns.str.replace(r'\s+', ' ', regex=True)
merged_data = pd.merge(c_data, e_data, on='City', how='inner')
merged_data = merged_data.replace(r',(?=\d)', '', regex=True)

merged_data = merged_data.astype({
    'City': 'string',
    'Violent crime': 'int32',
    'Murder and nonnegligent manslaughter': 'int32',
    'Rape (revised definition)': 'int32',
    'Robbery': 'int32',
    'Aggravated assault': 'int32',
    'Property crime': 'int32',
    'Burglary': 'int32',
    'Larceny-theft': 'int32',
    'Motor vehicle theft': 'int32',
    'Arson': 'int32',
    'Population': 'int32',
    'Total law enforcement employees': 'int32',
    'Total officers': 'int32',
    'Total civilians': 'int32'
})

print(merged_data.info())

# Load

# Writes data into a database
conn = sqlite3.connect('data/project3.db')
merged_data.to_sql('project3', conn, if_exists='replace', index=False)

# commit and close connection
conn.commit()
conn.close()

print("Database is created successfully.")