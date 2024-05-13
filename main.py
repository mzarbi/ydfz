import random
from datetime import datetime

import numpy as np
import pandas as pd

from ydfz.mfile import MappedDataFrame, PagedFile
from ydfz.serializers import MetadataSerializer


def generate_banking_data_frame(rows):
    """Generate a DataFrame that simulates banking data with specified number of rows."""
    data = {
        "Transaction_ID": np.arange(1, rows + 1),  # Unique ID for each transaction
        "Account_Number": np.random.randint(100000, 999999, size=rows),  # Simulated account number
        "Transaction_Date": pd.date_range(start="2023-01-01", periods=rows, freq="D"),  # Daily frequency
        "Transaction_Amount": np.random.uniform(-1000, 1000, size=rows).round(2),  # Transaction amounts
        "Account_Balance": np.random.uniform(0, 10000, size=rows).round(2),  # Simulated account balance
        "Customer_Age": np.random.randint(18, 70, size=rows),  # Age of the customer
        "Branch_Code": np.random.choice(['B001', 'B002', 'B003', 'B004'], size=rows),  # Branch codes
        "Transaction_Type": np.random.choice(['Deposit', 'Withdrawal', 'Payment', 'Transfer'], size=rows),
        # Types of transactions
        "Currency": np.random.choice(['USD', 'EUR', 'GBP'], size=rows),  # Transaction currency
        "Is_Fraud": np.random.choice([True, False], size=rows, p=[0.05, 0.95])  # Flag for fraudulent transactions
    }
    return pd.DataFrame(data)


def generate_banking_data_frame_with_nulls(rows, null_percentage=0.1):
    """Generate a DataFrame that simulates banking data with specified number of rows, including null values."""
    data = {
        "Transaction_ID": np.arange(1, rows + 1),  # Unique ID for each transaction
        "Account_Number": np.random.randint(100000, 999999, size=rows),  # Simulated account number
        "Transaction_Date": pd.date_range(start=f"20{random.randint(10, 30)}-01-01", periods=rows, freq="s"),
        # Daily frequency
        "Transaction_Amount": np.random.uniform(-1000, 1000, size=rows).round(2),  # Transaction amounts
        "Account_Balance": np.random.uniform(0, 10000, size=rows).round(2),  # Simulated account balance
        "Customer_Age": np.random.randint(18, 70, size=rows),  # Age of the customer
        "Branch_Code": np.random.choice(['B001', 'B002', 'B003', 'B004'], size=rows),  # Branch codes
        "Transaction_Type": np.random.choice(['Deposit', 'Withdrawal', 'Payment', 'Transfer'], size=rows),
        # Types of transactions
        "Currency": np.random.choice(['USD', 'EUR', 'GBP'], size=rows),  # Transaction currency
        "Is_Fraud": np.random.choice([True, False], size=rows, p=[0.05, 0.95])  # Flag for fraudulent transactions
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Introduce nulls into each column based on the null_percentage
    for column in df.columns:
        indices = np.random.choice(df.index, size=int(null_percentage * rows), replace=False)
        df.loc[indices, column] = np.nan

    return df


from tqdm import trange

shards = 10  # Number of data shards (iterations)
rows_per_shard = 100  # Number of rows per DataFrame

metadata = {'creator': 'John Doe', 'version': 1.0}
with MappedDataFrame('example.dat', 'write', file_metadata=metadata, footer_serializer=MetadataSerializer) as f:
    for shard in trange(shards):
        df = generate_banking_data_frame_with_nulls(rows_per_shard)
        df.to_parquet("data.parquet")
        for col in df.columns:
            column_data = df[col]
            dtype = str(column_data.dtype)

            f.write_series(column_data, dtype, compression="snappy", series_metadata={
                'shard': shard,
                'name': col
            }, compute_stats=True)

# Reading must also happen within a context where the file is open
with MappedDataFrame('example.dat', 'read', footer_serializer=MetadataSerializer) as file:
    relevant_partitions = file.query('output.dat',{
        "operation": "AND",
        "rules": [
            {
                "field": "Transaction_ID",
                "operator": "==",
                "value": 4
            }
        ],
        "groups": [

        ]
    })

with PagedFile('output.dat', 'read') as file:
    for tmp in range(file.page_count):
        print(file.read_page(tmp))