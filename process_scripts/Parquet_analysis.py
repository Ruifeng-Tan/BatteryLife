import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import numpy as np

# Load the parquet file (here is an example of PD/cell 113)
loadpath="dataset/raw_data/Farasis/PA-b1/001_single_parquet_with_soc_and_corrections.parquet"

# Load the quantities you'd like to use
data=pq.read_table(loadpath).to_pandas()
current = np.array(data["current (A)"])
voltage = np.array(data["voltage (V)"])
step = np.array(data["step_no"])
rpt = np.array(data["rpt"])
soc = np.array(data["soc"])

# You can also save the data as other format, like .csv
savepath="dataset/processed/"
data.to_csv(savepath+'XXX.csv',sep=',', index=False, mode='w', encoding='utf-8')

# You can also visualize them
plt.plot(voltage, linewidth=1.2)
plt.plot(current, linewidth=1.2)
plt.show()
