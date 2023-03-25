import pandas as pd
import numpy as np
import multiprocessing as mp
import ctypes

# Define the ctypes data structure that matches the structure of your DataFrame
class SharedDataFrame(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("index", ctypes.POINTER(ctypes.c_char_p)),
        ("columns", ctypes.POINTER(ctypes.c_char_p)),
        ("nrows", ctypes.c_int),
        ("ncols", ctypes.c_int),
    ]


# Create a DataFrame with some random data
df = pd.DataFrame(np.random.randn(10, 5), columns=list("ABCDE"))

# Create a shared memory block and cast it to the SharedDataFrame ctype
data_buffer = mp.shared_memory.SharedMemory(create=True, size=df.values.nbytes)
data_array = np.ndarray(df.shape, dtype=df.values.dtype, buffer=data_buffer.buf)
data_array[:] = df.values[:]
shared_df = SharedDataFrame(
    data=ctypes.cast(data_buffer.buf, ctypes.POINTER(ctypes.c_double)),
    index=ctypes.cast(id(df.index.values), ctypes.POINTER(ctypes.c_char_p)),
    columns=ctypes.cast(id(df.columns.values), ctypes.POINTER(ctypes.c_char_p)),
    nrows=ctypes.c_int(df.shape[0]),
    ncols=ctypes.c_int(df.shape[1]),
)

# Define a function that processes the DataFrame
def process_dataframe(sd):
    # Access the data directly from the shared memory block
    df = pd.DataFrame(
        np.ctypeslib.as_array(sd.data, shape=(sd.nrows.value, sd.ncols.value)),
        index=np.ctypeslib.as_array(sd.index, shape=(sd.nrows.value,)),
        columns=np.ctypeslib.as_array(sd.columns, shape=(sd.ncols.value,)),
    )
    # Add a new column to the DataFrame
    df["F"] = df["A"] + df["B"] + df["C"] + df["D"] + df["E"]
    # Update the shared DataFrame
    sd.data.contents[:] = np.ascontiguousarray(df.values).ravel()
    return sd


# Process the DataFrame in parallel
with mp.Pool() as pool:
    result = pool.map(process_dataframe, (shared_df,))
# Retrieve the updated DataFrame from shared memory
df.values[:] = np.ctypeslib.as_array(
    shared_df.data, shape=(shared_df.nrows.value, shared_df.ncols.value)
)
df.index = np.ctypeslib.as_array(shared_df.index, shape=(shared_df.nrows.value,))
df.columns = np.ctypeslib.as_array(shared_df.columns, shape=(shared_df.ncols.value,))
