import yaml
from sqlalchemy import create_engine
import pandas as pd

# Load the credentials
def loads_credentials():
    with open('credentials.yaml', 'r') as file:
     credentials = yaml.safe_load(file)
    return credentials

# Connecting to the remote database
# use dictionary methods inside this function to access each component of the credentials
class RDSDatabaseConnector:
    def __init__(self, credentials):
        self.host = credentials['RDS_HOST']
        self.password = credentials['RDS_PASSWORD']
        self.user = credentials['RDS_USER']
        self.database = credentials['RDS_DATABASE']
        self.port = credentials['RDS_PORT']
        self.engine = self._create_sqlalchemy_engine()

    def _create_sqlalchemy_engine(self):
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = self.host
        USER = self.user
        PASSWORD = self.password
        DATABASE = self.database
        PORT = self.port
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
        return engine

    # table named 'customer activity' will be returned as a dataframe.
    # with self.engine.connect() as cnx syntax ensures that the connection (cnx) is closed automatically after fetching data.
    def extract_data(self):
        with self.engine.connect() as cnx:
            df = pd.read_sql_table('customer_activity', cnx)
        return df

    # save data in csv format to your local machine
    # pass df and file_path as parameters to allow for flexibility with different df and filepaths
    def _save_date(self, df, file_path):
        df.to_csv(file_path, index=False)


    # create a function which will load the data from your local machine into a Pandas DataFrame
    def _load_data(self, file_path):
        df = pd.read_csv('customer_activity_data.csv')
        return df

# Example Usage and calling the methods
 # Step 1: Load credentials
credentials = loads_credentials()

 # Step 2: Create an instance of RDSDatabaseConnector
connector = RDSDatabaseConnector(credentials)

 # Step 3: Create the engine
connector._create_sqlalchemy_engine()

# Step 4: Extract data from the 'customer_activity' table
df = connector.extract_data()
print("Data extracted successfully")
print(df.head())

# Step 5: Save the DataFrame to a CSV file
file_path = 'customer_activity_data.csv'
connector._save_date(df, file_path )
print(f"Data saved to {file_path}")

# Step 6: Load the Dataframe from your local machine into a Pandas dataframe
loaded_df = connector._load_data(file_path)
print("Data loaded from CSV successfully")
print(loaded_df.head())