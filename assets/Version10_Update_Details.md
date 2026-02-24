# Version 10 of BatteryLife_Processed_Dataset update details

Here are the updated details of this version 10 compared to version 9:

1. **Fixing the discharging column data of the datasets `XJTU`, `ZN-coin`, and `CALB`.** In the previous versions, we did not split the raw capacity column data into the charge capacity column and the discharge capacity column.
2. **Updating the life labels of the datasets `XJTU`, `ZN-coin`, and `CALB`.** Due to the inappropriate capacity processing in the previous versions (**see 1**), our life label calculation could be inaccurate, so we recalculate the life labels for these three datasets.
3. **Fixing the wrong preprocessing operation for the dataset  `ZN-coin`.** In the previous versions,  we did not delete the first 9 formation cycles for some batteries.
4. **Fixing the time normalization script.** To standardize the time formats across different battery datasets, we applied a time normalization process to the raw data. Prior to processing, significant discrepancies existed in time-recording standards among various datasets: some (e.g., `RWTH` and `ISU_ILCC`) used milliseconds or nanoseconds, and their time baselines were inconsistent,certain datasets reset the time to zero at the start of each charge-discharge cycle, whereas others recorded global cumulative time. Additionally, some data contained anomalies such as abrupt timestamp drops, unexpected zero-resets, negative values, and excessively long gaps caused by test interruptions. To address these issues, our processing pipeline unified all underlying time records into global cumulative time in seconds, ensuring a monotonically increasing timeline across all cycles. Specifically, the preprocessing script automatically identified and converted non-second-level timestamps, eliminated negative anomalies, and repaired intra-cycle reset breakpoints by calculating time differences to stitch the data seamlessly. Finally, rest periods or interruption gaps exceeding 5 minutes within a single cycle were uniformly compressed to 1 second, thereby guaranteeing the continuity of the time series.




All updates have been released on both Zenodo and Huggingface websites.
