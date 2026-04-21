# Version 11 of BatteryLife_Processed_Dataset update details

Here are the update details of Version 11 compared with Version 10:

1. The charge capacity of the `CALB` dataset has been fixed. Refer to [Issue #21](https://github.com/Ruifeng-Tan/BatteryLife/issues/21).
2. The dataloader has been fixed to correctly handle the loading of Zn-ion batteries. Legacy code related to the OX dataset has also been cleaned up. Refer to [Issue #24](https://github.com/Ruifeng-Tan/BatteryLife/issues/24).
3. The seen/unseen category of `MATR_b3c0.pkl` has been revised from `unseen` to `seen`. Refer to [Issue #26](https://github.com/Ruifeng-Tan/BatteryLife/issues/26).
4. Sorting by `system_time` has been removed for the `XJTU` dataset to avoid random ordering of data points with identical `system_time` timestamps. Refer to [Issue #22](https://github.com/Ruifeng-Tan/BatteryLife/issues/22).
5. `CALCE` dataset: Adding the `charge_protocol` and `discharge_protocol` attribute to each cell.
6. `RWTH` dataset: Adding the `rate_in_C` to each cell and update preprocess_RWTH script to fix the multi charge and discharge record in one cycle issue. Refer to [Issue #22](https://github.com/Ruifeng-Tan/BatteryLife/issues/22).
7. (Label Update) The life labels for `CALB` dataset are corrected: We observed that labels of some `CALB` batteries were computed wrongly in v10. The new life label calculation script has been updated in Github repository.