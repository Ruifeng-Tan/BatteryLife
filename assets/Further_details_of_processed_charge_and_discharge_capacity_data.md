# Further details of processed charge and discharge capacity data

Due to the inconsistency of charge and discharge capacity recording standards across different datasets from various battery testing equipment, we have written this note to provide developers with the details of each recording standard for charge and discharge capacity for each raw sub-dataset in our BatteryLife.

- **Specific note:** If developers want to use the charge and discharge capacity data, they need to check this file carefully and may need further preprocessing for charge and discharge capacity data in order to suit their specific task.



## **Recording standard table for each sub-dataset**

| Index | Dataset  | Protocol type                    | Charge value        | Discharge value     | Save format |
| ----- | -------- | -------------------------------- | ------------------- | ------------------- | ----------- |
| 1     | CALCE    | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 1    |
| 2     | HNEI     | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 2    |
| 3     | HUST     | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 1    |
| 4     | UL_PUR   | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 2    |
| 5     | ISU_ILCC | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 2    |
| 6     | MATR     | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 1    |
| 7     | MICH     | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 2    |
| 8     | MICH_EXP | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 2    |
| 9     | RWTH     | **Discharge first, then charge** | 0-->positive number | 0-->positive number | Format 1    |
| 10    | SNL      | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 1    |
| 11    | Stanford | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 2    |
| 12    | Tongji   | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 2    |
| 13    | UL_PUR   | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 2    |
| 14    | XJTU     | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 2    |
| 15    | ZN-coin  | **Discharge first, then charge** | 0-->positive number | 0-->positive number | Format 2    |
| 16    | NA-ion   | Charge first, then discharge     | 0-->positive number | 0-->positive number | Format 2    |
| 17    | CALB     | **Discharge first, then charge** | 0-->positive number | 0-->positive number | Format 2    |



## **Detailed explanations of different save formats**

- ### **Format 1**

  - The raw charge capacity and discharge capacity are recorded in different columns. 
  - During charge:
    - If the `protocol type` is **charge first, then discharge**, the charge capacity starts from 0 to a positive number, and the discharge capacity is maintained at 0.
    - If the `protocol type` is **discharge first, then charge**, the discharge capacity will be maintained at the last discharge capacity point during charge, and the charge capacity starts from 0 to a positive number.
  - During discharge:
    - If the `protocol type` is **charge first, then discharge**, the charge capacity will be maintained at the last charge capacity point during discharge, and the discharge capacity starts from 0 to a positive number.
    - If the `protocol type` is **discharge first, then charge**, the discharge capacity starts from 0 to a positive number, and the charge capacity is maintained at 0.
  - **How did we process?**
    - We simply copy the raw data of charge and discharge capacity into our uniform data format of `charge_capacity_in_Ah` and `discharge_capacity_in_Ah` columns.



- ### **Format 2**

  - The raw charge capacity and discharge capacity columns are recorded in different columns. 
  - During charge:
    - If the `protocol type` is **charge first, then discharge**, the charge capacity starts from 0 to a positive number, and the discharge capacity is maintained at 0.
    - If the `protocol type` is **discharge first, then charge**, the discharge capacity will be set at 0 during charge, and the charge capacity starts from 0 to a positive number.
  - During discharge:
    - If the `protocol type` is **charge first, then discharge**, the charge capacity will be set at 0 during discharge, and the discharge capacity starts from 0 to a positive number.
    - If the `protocol type` is **discharge first, then charge**, the discharge capacity starts from 0 to a positive number, and the charge capacity is maintained at 0.
  - **How did we process?**
    - We simply copy the raw data of charge and discharge capacity into our uniform data format of `charge_capacity_in_Ah` and `discharge_capacity_in_Ah` columns.