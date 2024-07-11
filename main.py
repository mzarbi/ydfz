import random
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
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


def generate_banking_data_frame_with_nulls(rows, test_indices=None):
    """Generate a DataFrame that simulates banking data with specified number of rows, including null values."""
    if test_indices is None:
        test_indices = {}

    data = {
        "Transaction_ID": np.arange(1, rows + 1),
        "Account_Number": np.random.randint(100000, 999999, size=rows),
        "Transaction_Date": pd.date_range(start=f"20{random.randint(10, 30)}-01-01", periods=rows, freq="s"),
        "Transaction_Amount": np.random.uniform(-1000, 1000, size=rows).round(2),
        "Account_Balance": np.random.uniform(0, 10000, size=rows).round(2),
        "Customer_Age": np.random.randint(18, 70, size=rows),
        "Branch_Code": np.random.choice(['B001', 'B002', 'B003', 'B004'], size=rows),
        "Transaction_Type": np.random.choice(['Deposit', 'Withdrawal', 'Payment', 'Transfer'], size=rows),
        "Currency": np.random.choice(['USD', 'EUR', 'GBP'], size=rows),
        "Is_Fraud": np.random.choice([True, False], size=rows, p=[0.05, 0.95])
    }

    df = pd.DataFrame(data)

    # Introduce specific test values at known indices
    for column, idx_value_pairs in test_indices.items():
        for idx, value in idx_value_pairs:
            if idx < rows:
                df.at[idx, column] = value

    # Introduce nulls
    for column in df.columns:
        indices = np.random.choice(df.index, size=int(0.1 * rows), replace=False)
        df.loc[indices, column] = np.nan

    return df


from tqdm import trange

shards = 1  # Number of data shards (iterations)
rows_per_shard = 100000  # Number of rows per DataFrame

test_values = {
    'Transaction_ID': [(50, 12345), (100, 67890)],  # Inject Transaction_ID 12345 at index 50, etc.
}

metadata = {'creator': 'John Doe', 'version': 1.0}
with MappedDataFrame('example.dat', 'write', file_metadata=metadata, footer_serializer=MetadataSerializer) as f:
    for shard in trange(shards):
        if shard == 7:
            df = generate_banking_data_frame_with_nulls(rows_per_shard, test_values)
        else:
            df = generate_banking_data_frame_with_nulls(rows_per_shard)
        for col in df.columns:
            column_data = df[col]

            if col == "Transaction_ID":
                dtype = "int"
            if col == "Account_Number":
                dtype = "int"
            if col == "Customer_Age":
                dtype = "int"
            else:
                dtype = str(column_data.dtype)
            f.write_series(column_data, dtype, compression="snappy", series_metadata={
                'shard': shard,
                'name': col
            }, compute_stats=True)

# Reading must also happen within a context where the file is open
with MappedDataFrame('example.dat', 'read', footer_serializer=MetadataSerializer) as file:
    query = {
        "operation": "AND",
        "rules": [
            {
                "field": "Transaction_ID",
                "operator": "==",
                "value": 12345
            }
        ],
        "groups": [

        ]
    }
    relevant_shards = file.predicate_pushdown(query)
    print(len(file.partitions))
    print(relevant_shards)
    for i in relevant_shards:
        print(file.read_shard(i))
    relevant_partitions = file.query('output.dat', query)

with PagedFile('output.dat', 'read') as file:
    for tmp in range(file.page_count):
        print(file.read_page(tmp))
