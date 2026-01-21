import os
from tqdm import tqdm
import json

HUST_train_files = ['HUST_1-6.pkl', 'HUST_2-2.pkl', 'HUST_1-3.pkl', 'HUST_6-3.pkl', 'HUST_1-2.pkl', 'HUST_3-7.pkl',
                    'HUST_3-2.pkl', 'HUST_10-6.pkl', 'HUST_3-6.pkl', 'HUST_5-1.pkl', 'HUST_10-5.pkl', 'HUST_6-2.pkl',
                    'HUST_6-1.pkl', 'HUST_8-1.pkl', 'HUST_10-7.pkl', 'HUST_1-4.pkl', 'HUST_5-4.pkl', 'HUST_1-5.pkl',
                    'HUST_6-6.pkl', 'HUST_5-6.pkl', 'HUST_6-4.pkl', 'HUST_9-2.pkl', 'HUST_10-4.pkl', 'HUST_5-3.pkl',
                    'HUST_7-7.pkl', 'HUST_3-1.pkl', 'HUST_4-1.pkl', 'HUST_4-4.pkl', 'HUST_4-6.pkl', 'HUST_8-8.pkl',
                    'HUST_2-4.pkl', 'HUST_9-8.pkl', 'HUST_9-5.pkl', 'HUST_3-3.pkl', 'HUST_1-7.pkl', 'HUST_4-5.pkl',
                    'HUST_9-6.pkl', 'HUST_1-1.pkl', 'HUST_4-3.pkl', 'HUST_2-5.pkl', 'HUST_4-7.pkl', 'HUST_7-2.pkl',
                    'HUST_8-4.pkl', 'HUST_3-5.pkl', 'HUST_2-6.pkl', 'HUST_8-6.pkl', 'HUST_7-5.pkl']
HUST_val_files = ['HUST_6-8.pkl', 'HUST_3-8.pkl', 'HUST_2-3.pkl', 'HUST_9-1.pkl', 'HUST_10-8.pkl', 'HUST_7-4.pkl',
                  'HUST_2-8.pkl', 'HUST_8-3.pkl', 'HUST_5-2.pkl', 'HUST_10-1.pkl', 'HUST_5-5.pkl', 'HUST_5-7.pkl',
                  'HUST_7-8.pkl', 'HUST_7-6.pkl', 'HUST_1-8.pkl']
HUST_test_files = ['HUST_9-4.pkl', 'HUST_10-2.pkl', 'HUST_10-3.pkl', 'HUST_8-5.pkl', 'HUST_7-3.pkl', 'HUST_7-1.pkl',
                   'HUST_9-3.pkl', 'HUST_4-2.pkl', 'HUST_8-7.pkl', 'HUST_9-7.pkl', 'HUST_6-5.pkl', 'HUST_3-4.pkl',
                   'HUST_8-2.pkl', 'HUST_4-8.pkl', 'HUST_2-7.pkl']

MATR_train_files = ['MATR_b1c5.pkl', 'MATR_b3c6.pkl', 'MATR_b1c7.pkl', 'MATR_b4c18.pkl', 'MATR_b3c44.pkl',
                    'MATR_b3c18.pkl', 'MATR_b2c12.pkl', 'MATR_b2c26.pkl', 'MATR_b4c26.pkl', 'MATR_b3c40.pkl',
                    'MATR_b1c9.pkl', 'MATR_b2c18.pkl', 'MATR_b3c17.pkl', 'MATR_b3c21.pkl', 'MATR_b2c6.pkl',
                    'MATR_b3c45.pkl', 'MATR_b1c23.pkl', 'MATR_b2c4.pkl', 'MATR_b1c35.pkl', 'MATR_b1c19.pkl',
                    'MATR_b2c39.pkl', 'MATR_b3c26.pkl', 'MATR_b2c28.pkl', 'MATR_b2c33.pkl', 'MATR_b3c39.pkl',
                    'MATR_b2c29.pkl', 'MATR_b3c31.pkl', 'MATR_b4c42.pkl', 'MATR_b4c7.pkl', 'MATR_b2c38.pkl',
                    'MATR_b1c41.pkl', 'MATR_b3c33.pkl', 'MATR_b4c40.pkl', 'MATR_b1c17.pkl', 'MATR_b3c1.pkl',
                    'MATR_b4c10.pkl', 'MATR_b1c33.pkl', 'MATR_b2c10.pkl', 'MATR_b3c13.pkl', 'MATR_b4c37.pkl',
                    'MATR_b4c23.pkl', 'MATR_b4c15.pkl', 'MATR_b2c0.pkl', 'MATR_b2c19.pkl', 'MATR_b4c1.pkl',
                    'MATR_b3c8.pkl', 'MATR_b1c15.pkl', 'MATR_b4c24.pkl', 'MATR_b3c15.pkl', 'MATR_b1c3.pkl',
                    'MATR_b1c16.pkl', 'MATR_b3c3.pkl', 'MATR_b4c20.pkl', 'MATR_b4c30.pkl', 'MATR_b4c25.pkl',
                    'MATR_b4c9.pkl', 'MATR_b1c20.pkl', 'MATR_b3c14.pkl', 'MATR_b2c5.pkl', 'MATR_b3c22.pkl',
                    'MATR_b3c16.pkl', 'MATR_b4c43.pkl', 'MATR_b4c19.pkl', 'MATR_b2c31.pkl', 'MATR_b2c21.pkl',
                    'MATR_b4c12.pkl', 'MATR_b2c36.pkl', 'MATR_b1c21.pkl', 'MATR_b2c3.pkl', 'MATR_b2c37.pkl',
                    'MATR_b4c4.pkl', 'MATR_b2c44.pkl', 'MATR_b4c34.pkl', 'MATR_b4c29.pkl', 'MATR_b3c7.pkl',
                    'MATR_b4c21.pkl', 'MATR_b2c1.pkl', 'MATR_b1c31.pkl', 'MATR_b2c14.pkl', 'MATR_b1c26.pkl',
                    'MATR_b4c38.pkl', 'MATR_b1c42.pkl', 'MATR_b2c17.pkl', 'MATR_b3c28.pkl', 'MATR_b3c10.pkl',
                    'MATR_b3c36.pkl', 'MATR_b3c24.pkl', 'MATR_b1c6.pkl', 'MATR_b2c34.pkl', 'MATR_b3c9.pkl',
                    'MATR_b4c14.pkl', 'MATR_b2c24.pkl', 'MATR_b2c30.pkl', 'MATR_b3c4.pkl', 'MATR_b4c11.pkl',
                    'MATR_b2c41.pkl', 'MATR_b4c8.pkl', 'MATR_b3c25.pkl', 'MATR_b1c38.pkl', 'MATR_b3c27.pkl',
                    'MATR_b1c18.pkl', 'MATR_b2c32.pkl']
MATR_val_files = ['MATR_b1c32.pkl', 'MATR_b1c36.pkl', 'MATR_b4c16.pkl', 'MATR_b1c25.pkl', 'MATR_b4c5.pkl',
                  'MATR_b3c11.pkl', 'MATR_b3c38.pkl', 'MATR_b4c39.pkl', 'MATR_b1c43.pkl', 'MATR_b1c11.pkl',
                  'MATR_b2c45.pkl', 'MATR_b4c3.pkl', 'MATR_b2c46.pkl', 'MATR_b2c40.pkl', 'MATR_b2c2.pkl',
                  'MATR_b2c47.pkl', 'MATR_b2c22.pkl', 'MATR_b3c35.pkl', 'MATR_b2c35.pkl', 'MATR_b1c2.pkl',
                  'MATR_b1c28.pkl', 'MATR_b4c28.pkl', 'MATR_b1c30.pkl', 'MATR_b2c25.pkl', 'MATR_b4c33.pkl',
                  'MATR_b4c31.pkl', 'MATR_b2c13.pkl', 'MATR_b1c27.pkl', 'MATR_b3c34.pkl', 'MATR_b4c32.pkl',
                  'MATR_b4c13.pkl', 'MATR_b1c14.pkl', 'MATR_b1c34.pkl', 'MATR_b4c22.pkl']
MATR_test_files = ['MATR_b2c20.pkl', 'MATR_b1c1.pkl', 'MATR_b1c0.pkl', 'MATR_b1c24.pkl', 'MATR_b1c39.pkl',
                   'MATR_b1c45.pkl', 'MATR_b3c30.pkl', 'MATR_b3c12.pkl', 'MATR_b1c37.pkl', 'MATR_b1c4.pkl',
                   'MATR_b1c44.pkl', 'MATR_b3c5.pkl', 'MATR_b3c19.pkl', 'MATR_b1c29.pkl', 'MATR_b4c36.pkl',
                   'MATR_b4c44.pkl', 'MATR_b4c2.pkl', 'MATR_b1c40.pkl', 'MATR_b3c41.pkl', 'MATR_b2c23.pkl',
                   'MATR_b3c29.pkl', 'MATR_b4c6.pkl', 'MATR_b3c20.pkl', 'MATR_b4c35.pkl', 'MATR_b3c0.pkl',
                   'MATR_b4c41.pkl', 'MATR_b4c27.pkl', 'MATR_b4c17.pkl', 'MATR_b2c27.pkl', 'MATR_b2c43.pkl',
                   'MATR_b2c11.pkl', 'MATR_b2c42.pkl', 'MATR_b4c0.pkl']

# BatteryML MIX100 does not have SNL batteries in the testing set. We resplit it.
SNL_train_files = ['SNL_18650_NCA_25C_0-100_0.5-0.5C_b.pkl', 'SNL_18650_NCA_35C_0-100_0.5-1C_b.pkl',
                   'SNL_18650_NMC_35C_0-100_0.5-2C_b.pkl', 'SNL_18650_NMC_25C_0-100_0.5-3C_b.pkl',
                   'SNL_18650_NMC_25C_0-100_0.5-1C_a.pkl', 'SNL_18650_LFP_25C_0-100_0.5-3C_c.pkl',
                   'SNL_18650_NCA_25C_0-100_0.5-2C_b.pkl', 'SNL_18650_NMC_15C_0-100_0.5-2C_b.pkl',
                   'SNL_18650_LFP_35C_0-100_0.5-1C_d.pkl', 'SNL_18650_NCA_35C_0-100_0.5-1C_a.pkl',
                   'SNL_18650_LFP_35C_0-100_0.5-2C_b.pkl', 'SNL_18650_NMC_25C_0-100_0.5-1C_c.pkl',
                   'SNL_18650_NMC_25C_0-100_0.5-3C_a.pkl', 'SNL_18650_NCA_35C_0-100_0.5-1C_d.pkl',
                   'SNL_18650_NMC_25C_0-100_0.5-1C_d.pkl', 'SNL_18650_LFP_25C_0-100_0.5-3C_a.pkl',
                   'SNL_18650_NCA_15C_0-100_0.5-2C_a.pkl', 'SNL_18650_NMC_25C_0-100_0.5-2C_a.pkl',
                   'SNL_18650_NCA_35C_0-100_0.5-1C_c.pkl', 'SNL_18650_NCA_15C_0-100_0.5-1C_b.pkl',
                   'SNL_18650_NCA_35C_0-100_0.5-2C_a.pkl', 'SNL_18650_LFP_25C_0-100_0.5-3C_d.pkl',
                   'SNL_18650_NCA_15C_0-100_0.5-1C_a.pkl', 'SNL_18650_NMC_35C_0-100_0.5-2C_a.pkl',
                   'SNL_18650_NMC_15C_0-100_0.5-2C_a.pkl', 'SNL_18650_NCA_25C_20-80_0.5-0.5C_c.pkl',
                   'SNL_18650_NMC_25C_0-100_0.5-2C_b.pkl', 'SNL_18650_NMC_25C_0-100_0.5-0.5C_b.pkl',
                   'SNL_18650_NMC_35C_0-100_0.5-1C_a.pkl', 'SNL_18650_LFP_35C_0-100_0.5-1C_b.pkl']
SNL_val_files = ['SNL_18650_NMC_25C_0-100_0.5-1C_b.pkl', 'SNL_18650_NMC_15C_0-100_0.5-1C_b.pkl',
                 'SNL_18650_NCA_25C_20-80_0.5-0.5C_d.pkl', 'SNL_18650_NCA_35C_0-100_0.5-2C_b.pkl',
                 'SNL_18650_NMC_25C_0-100_0.5-3C_d.pkl', 'SNL_18650_NCA_25C_0-100_0.5-1C_d.pkl',
                 'SNL_18650_NCA_25C_0-100_0.5-1C_a.pkl', 'SNL_18650_NCA_25C_0-100_0.5-1C_b.pkl',
                 'SNL_18650_NMC_35C_0-100_0.5-1C_d.pkl', 'SNL_18650_NCA_25C_0-100_0.5-0.5C_a.pkl']
SNL_test_files = ['SNL_18650_NMC_15C_0-100_0.5-1C_a.pkl', 'SNL_18650_NMC_35C_0-100_0.5-1C_c.pkl',
                  'SNL_18650_NCA_25C_0-100_0.5-2C_a.pkl', 'SNL_18650_NMC_25C_0-100_0.5-3C_c.pkl',
                  'SNL_18650_LFP_25C_0-100_0.5-3C_b.pkl', 'SNL_18650_NCA_15C_0-100_0.5-2C_b.pkl',
                  'SNL_18650_NMC_35C_0-100_0.5-1C_b.pkl', 'SNL_18650_LFP_35C_0-100_0.5-1C_c.pkl',
                  'SNL_18650_LFP_35C_0-100_0.5-2C_a.pkl', 'SNL_18650_NCA_25C_0-100_0.5-1C_c.pkl']

RWTH_train_files = ['RWTH_016.pkl', 'RWTH_045.pkl', 'RWTH_009.pkl', 'RWTH_039.pkl', 'RWTH_046.pkl', 'RWTH_019.pkl',
                    'RWTH_037.pkl', 'RWTH_013.pkl', 'RWTH_003.pkl', 'RWTH_044.pkl', 'RWTH_026.pkl', 'RWTH_006.pkl',
                    'RWTH_031.pkl', 'RWTH_036.pkl', 'RWTH_048.pkl', 'RWTH_033.pkl', 'RWTH_021.pkl', 'RWTH_012.pkl',
                    'RWTH_034.pkl', 'RWTH_018.pkl', 'RWTH_022.pkl', 'RWTH_030.pkl', 'RWTH_028.pkl', 'RWTH_011.pkl',
                    'RWTH_040.pkl', 'RWTH_041.pkl', 'RWTH_042.pkl', 'RWTH_025.pkl', 'RWTH_047.pkl', 'RWTH_004.pkl']
RWTH_val_files = ['RWTH_007.pkl', 'RWTH_032.pkl', 'RWTH_024.pkl', 'RWTH_002.pkl', 'RWTH_029.pkl', 'RWTH_010.pkl',
                  'RWTH_027.pkl', 'RWTH_014.pkl', 'RWTH_049.pkl']
RWTH_test_files = ['RWTH_038.pkl', 'RWTH_008.pkl', 'RWTH_035.pkl', 'RWTH_017.pkl', 'RWTH_015.pkl', 'RWTH_023.pkl',
                   'RWTH_020.pkl', 'RWTH_005.pkl', 'RWTH_043.pkl']

MICH_train_files = ['MICH_BLForm2_pouch_NMC_45C_0-100_1-1C_b.pkl', 'MICH_BLForm1_pouch_NMC_45C_0-100_1-1C_a.pkl',
                    'MICH_MCForm32_pouch_NMC_45C_0-100_1-1C_b.pkl', 'MICH_MCForm27_pouch_NMC_25C_0-100_1-1C_g.pkl',
                    'MICH_MCForm22_pouch_NMC_25C_0-100_1-1C_b.pkl', 'MICH_MCForm36_pouch_NMC_45C_0-100_1-1C_f.pkl',
                    'MICH_MCForm29_pouch_NMC_25C_0-100_1-1C_i.pkl', 'MICH_MCForm39_pouch_NMC_45C_0-100_1-1C_i.pkl',
                    'MICH_BLForm5_pouch_NMC_45C_0-100_1-1C_e.pkl', 'MICH_BLForm16_pouch_NMC_25C_0-100_1-1C_f.pkl',
                    'MICH_MCForm30_pouch_NMC_25C_0-100_1-1C_j.pkl', 'MICH_BLForm14_pouch_NMC_25C_0-100_1-1C_d.pkl',
                    'MICH_MCForm38_pouch_NMC_45C_0-100_1-1C_h.pkl', 'MICH_BLForm8_pouch_NMC_45C_0-100_1-1C_h.pkl',
                    'MICH_BLForm17_pouch_NMC_25C_0-100_1-1C_g.pkl', 'MICH_MCForm23_pouch_NMC_25C_0-100_1-1C_c.pkl',
                    'MICH_BLForm18_pouch_NMC_25C_0-100_1-1C_h.pkl', 'MICH_BLForm10_pouch_NMC_25C_0-100_1-1C_j.pkl',
                    'MICH_MCForm35_pouch_NMC_45C_0-100_1-1C_e.pkl', 'MICH_MCForm25_pouch_NMC_25C_0-100_1-1C_e.pkl',
                    'MICH_MCForm26_pouch_NMC_25C_0-100_1-1C_f.pkl', 'MICH_MCForm37_pouch_NMC_45C_0-100_1-1C_g.pkl',
                    'MICH_BLForm15_pouch_NMC_25C_0-100_1-1C_e.pkl', 'MICH_MCForm31_pouch_NMC_45C_0-100_1-1C_a.pkl']
MICH_val_files = ['MICH_BLForm3_pouch_NMC_45C_0-100_1-1C_c.pkl', 'MICH_BLForm4_pouch_NMC_45C_0-100_1-1C_d.pkl',
                  'MICH_MCForm24_pouch_NMC_25C_0-100_1-1C_d.pkl', 'MICH_BLForm12_pouch_NMC_25C_0-100_1-1C_b.pkl',
                  'MICH_BLForm13_pouch_NMC_25C_0-100_1-1C_c.pkl', 'MICH_BLForm19_pouch_NMC_25C_0-100_1-1C_i.pkl',
                  'MICH_BLForm11_pouch_NMC_25C_0-100_1-1C_a.pkl', 'MICH_BLForm9_pouch_NMC_45C_0-100_1-1C_i.pkl']
MICH_test_files = ['MICH_BLForm20_pouch_NMC_25C_0-100_1-1C_j.pkl', 'MICH_MCForm21_pouch_NMC_25C_0-100_1-1C_a.pkl',
                   'MICH_MCForm34_pouch_NMC_45C_0-100_1-1C_d.pkl', 'MICH_BLForm6_pouch_NMC_45C_0-100_1-1C_f.pkl',
                   'MICH_MCForm40_pouch_NMC_45C_0-100_1-1C_j.pkl', 'MICH_BLForm7_pouch_NMC_45C_0-100_1-1C_g.pkl',
                   'MICH_MCForm28_pouch_NMC_25C_0-100_1-1C_h.pkl', 'MICH_MCForm33_pouch_NMC_45C_0-100_1-1C_c.pkl']

MICH_EXP_train_files = ['MICH_11C_pouch_NMC_-5C_0-100_0.2-1.5C.pkl', 'MICH_02C_pouch_NMC_-5C_0-100_0.2-0.2C.pkl',
                        'MICH_18H_pouch_NMC_45C_50-100_0.2-1.5C.pkl', 'MICH_04R_pouch_NMC_25C_0-100_1.5-1.5C.pkl',
                        'MICH_03H_pouch_NMC_45C_0-100_0.2-0.2C.pkl', 'MICH_08C_pouch_NMC_-5C_0-100_2-2C.pkl',
                        'MICH_17C_pouch_NMC_-5C_50-100_0.2-1.5C.pkl', 'MICH_10R_pouch_NMC_25C_0-100_0.2-1.5C.pkl',
                        'MICH_15H_pouch_NMC_45C_50-100_0.2-0.2C.pkl', 'MICH_13R_pouch_NMC_25C_50-100_0.2-0.2C.pkl',
                        'MICH_14C_pouch_NMC_-5C_50-100_0.2-0.2C.pkl', 'MICH_12H_pouch_NMC_45C_0-100_0.2-1.5C.pkl']
MICH_EXP_val_files = ['MICH_06H_pouch_NMC_45C_0-100_1.5-1.5C.pkl', 'MICH_07R_pouch_NMC_25C_0-100_2-2C.pkl',
                      'MICH_05C_pouch_NMC_-5C_0-100_1.5-1.5C.pkl']
MICH_EXP_test_files = ['MICH_09H_pouch_NMC_45C_0-100_2-2C.pkl', 'MICH_16R_pouch_NMC_25C_50-100_0.2-1.5C.pkl',
                       'MICH_01R_pouch_NMC_25C_0-100_0.2-0.2C.pkl']

UL_PUR_train_files = ['UL-PUR_N10-NA7_18650_NCA_23C_0-100_0.5-0.5C_g.pkl',
                      'UL-PUR_N15-NA10_18650_NCA_23C_0-100_0.5-0.5C_j.pkl']
UL_PUR_val_files = []
UL_PUR_test_files = []

CALCE_train_files = ['CALCE_CS2_36.pkl', 'CALCE_CS2_37.pkl', 'CALCE_CX2_33.pkl', 'CALCE_CS2_34.pkl', 'CALCE_CX2_37.pkl',
                     'CALCE_CS2_33.pkl', 'CALCE_CS2_35.pkl', 'CALCE_CS2_38.pkl', 'CALCE_CX2_36.pkl']
CALCE_val_files = ['CALCE_CX2_35.pkl', 'CALCE_CX2_34.pkl']
CALCE_test_files = ['CALCE_CX2_16.pkl', 'CALCE_CX2_38.pkl']

HNEI_train_files = ['HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_o.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_p.pkl',
                    'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_e.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_l.pkl',
                    'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_b.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_t.pkl',
                    'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_c.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_a.pkl',
                    'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_g.pkl']
HNEI_val_files = ['HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_n.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_s.pkl',
                  'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_j.pkl']
HNEI_test_files = ['HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_f.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_d.pkl']

Tongji_train_files = ['Tongji1_CY35-05_1--2.pkl', 'Tongji1_CY45-05_1--21.pkl', 'Tongji1_CY45-05_1--28.pkl',
                      'Tongji2_CY45-05_1--11.pkl', 'Tongji1_CY25-1_1--5.pkl', 'Tongji1_CY25-1_1--9.pkl',
                      'Tongji1_CY45-05_1--19.pkl', 'Tongji3_CY25-05_4--1.pkl', 'Tongji1_CY25-05_1--2.pkl',
                      'Tongji3_CY25-05_2--1.pkl', 'Tongji1_CY45-05_1--5.pkl', 'Tongji1_CY45-05_1--13.pkl',
                      'Tongji1_CY45-05_1--11.pkl', 'Tongji1_CY35-05_1--1.pkl', 'Tongji1_CY45-05_1--17.pkl',
                      'Tongji2_CY25-05_1--13.pkl', 'Tongji1_CY25-1_1--6.pkl', 'Tongji2_CY45-05_1--22.pkl',
                      'Tongji1_CY45-05_1--27.pkl', 'Tongji2_CY45-05_1--27.pkl', 'Tongji1_CY25-05_1--17.pkl',
                      'Tongji2_CY45-05_1--9.pkl', 'Tongji3_CY25-05_2--2.pkl', 'Tongji1_CY45-05_1--18.pkl',
                      'Tongji3_CY25-05_4--2.pkl', 'Tongji1_CY45-05_1--20.pkl', 'Tongji1_CY45-05_1--1.pkl',
                      'Tongji2_CY45-05_1--23.pkl', 'Tongji2_CY25-05_1--5.pkl', 'Tongji2_CY45-05_1--25.pkl',
                      'Tongji1_CY25-05_1--4.pkl', 'Tongji1_CY25-05_1--12.pkl', 'Tongji1_CY25-05_1--16.pkl',
                      'Tongji1_CY25-05_1--13.pkl', 'Tongji2_CY25-05_1--17.pkl', 'Tongji1_CY45-05_1--7.pkl',
                      'Tongji1_CY25-025_1--5.pkl', 'Tongji2_CY25-05_1--12.pkl', 'Tongji1_CY45-05_1--6.pkl',
                      'Tongji3_CY25-05_1--2.pkl', 'Tongji3_CY25-05_2--3.pkl', 'Tongji2_CY45-05_1--16.pkl',
                      'Tongji1_CY25-05_1--15.pkl', 'Tongji1_CY25-025_1--3.pkl', 'Tongji2_CY45-05_1--7.pkl',
                      'Tongji2_CY45-05_1--20.pkl', 'Tongji1_CY45-05_1--23.pkl', 'Tongji2_CY25-05_1--15.pkl',
                      'Tongji2_CY35-05_1--4.pkl', 'Tongji1_CY45-05_1--12.pkl', 'Tongji2_CY45-05_1--15.pkl',
                      'Tongji1_CY25-05_1--5.pkl', 'Tongji2_CY35-05_1--3.pkl', 'Tongji1_CY25-05_1--14.pkl',
                      'Tongji1_CY45-05_1--26.pkl', 'Tongji1_CY45-05_1--22.pkl', 'Tongji2_CY45-05_1--24.pkl',
                      'Tongji1_CY25-05_1--18.pkl', 'Tongji1_CY45-05_1--9.pkl', 'Tongji1_CY45-05_1--24.pkl',
                      'Tongji1_CY25-1_1--4.pkl', 'Tongji2_CY45-05_1--26.pkl', 'Tongji3_CY25-05_1--3.pkl',
                      'Tongji1_CY45-05_1--15.pkl', 'Tongji1_CY25-025_1--2.pkl', 'Tongji2_CY45-05_1--14.pkl']
Tongji_val_files = ['Tongji1_CY25-1_1--3.pkl', 'Tongji1_CY25-05_1--3.pkl', 'Tongji1_CY45-05_1--10.pkl',
                    'Tongji1_CY45-05_1--14.pkl', 'Tongji2_CY45-05_1--21.pkl', 'Tongji2_CY45-05_1--8.pkl',
                    'Tongji2_CY25-05_1--8.pkl', 'Tongji2_CY45-05_1--18.pkl', 'Tongji1_CY25-1_1--8.pkl',
                    'Tongji1_CY45-05_1--8.pkl', 'Tongji2_CY45-05_1--10.pkl', 'Tongji1_CY45-05_1--16.pkl',
                    'Tongji2_CY45-05_1--1.pkl', 'Tongji1_CY25-05_1--10.pkl', 'Tongji1_CY25-025_1--4.pkl',
                    'Tongji2_CY25-05_1--16.pkl', 'Tongji1_CY25-05_1--11.pkl', 'Tongji2_CY45-05_1--12.pkl',
                    'Tongji2_CY25-05_1--10.pkl', 'Tongji1_CY25-025_1--1.pkl', 'Tongji1_CY25-05_1--7.pkl']
Tongji_test_files = ['Tongji1_CY45-05_1--2.pkl', 'Tongji1_CY45-05_1--25.pkl', 'Tongji2_CY25-05_1--9.pkl',
                     'Tongji1_CY25-1_1--1.pkl', 'Tongji1_CY25-05_1--19.pkl', 'Tongji1_CY25-1_1--2.pkl',
                     'Tongji2_CY45-05_1--17.pkl', 'Tongji1_CY25-025_1--7.pkl', 'Tongji2_CY45-05_1--19.pkl',
                     'Tongji2_CY45-05_1--13.pkl', 'Tongji1_CY25-025_1--6.pkl', 'Tongji2_CY35-05_1--2.pkl',
                     'Tongji2_CY45-05_1--2.pkl', 'Tongji1_CY25-05_1--1.pkl', 'Tongji3_CY25-05_4--3.pkl',
                     'Tongji3_CY25-05_1--1.pkl', 'Tongji1_CY25-1_1--7.pkl', 'Tongji2_CY45-05_1--28.pkl',
                     'Tongji2_CY35-05_1--1.pkl', 'Tongji2_CY25-05_1--2.pkl', 'Tongji1_CY25-05_1--6.pkl']

Stanford_train_files = ['Stanford_Nova_Regular_100.pkl', 'Stanford_Nova_Regular_102.pkl', 'Stanford_Nova_Regular_103.pkl', 'Stanford_Nova_Regular_104.pkl', 'Stanford_Nova_Regular_107.pkl', 'Stanford_Nova_Regular_109.pkl', 'Stanford_Nova_Regular_113.pkl', 'Stanford_Nova_Regular_115.pkl', 'Stanford_Nova_Regular_116.pkl', 'Stanford_Nova_Regular_118.pkl', 'Stanford_Nova_Regular_119.pkl', 'Stanford_Nova_Regular_121.pkl', 'Stanford_Nova_Regular_122.pkl', 'Stanford_Nova_Regular_123.pkl', 'Stanford_Nova_Regular_124.pkl', 'Stanford_Nova_Regular_125.pkl', 'Stanford_Nova_Regular_126.pkl', 'Stanford_Nova_Regular_131.pkl', 'Stanford_Nova_Regular_134.pkl', 'Stanford_Nova_Regular_135.pkl', 'Stanford_Nova_Regular_136.pkl', 'Stanford_Nova_Regular_137.pkl', 'Stanford_Nova_Regular_140.pkl', 'Stanford_Nova_Regular_141.pkl', 'Stanford_Nova_Regular_142.pkl', 'Stanford_Nova_Regular_148.pkl', 'Stanford_Nova_Regular_149.pkl', 'Stanford_Nova_Regular_150.pkl', 'Stanford_Nova_Regular_151.pkl', 'Stanford_Nova_Regular_152.pkl', 'Stanford_Nova_Regular_154.pkl', 'Stanford_Nova_Regular_156.pkl', 'Stanford_Nova_Regular_158.pkl', 'Stanford_Nova_Regular_159.pkl', 'Stanford_Nova_Regular_161.pkl', 'Stanford_Nova_Regular_163.pkl', 'Stanford_Nova_Regular_166.pkl', 'Stanford_Nova_Regular_169.pkl', 'Stanford_Nova_Regular_170.pkl', 'Stanford_Nova_Regular_171.pkl', 'Stanford_Nova_Regular_173.pkl', 'Stanford_Nova_Regular_181.pkl', 'Stanford_Nova_Regular_182.pkl', 'Stanford_Nova_Regular_183.pkl', 'Stanford_Nova_Regular_184.pkl', 'Stanford_Nova_Regular_187.pkl', 'Stanford_Nova_Regular_188.pkl', 'Stanford_Nova_Regular_189.pkl', 'Stanford_Nova_Regular_190.pkl', 'Stanford_Nova_Regular_192.pkl', 'Stanford_Nova_Regular_193.pkl', 'Stanford_Nova_Regular_194.pkl', 'Stanford_Nova_Regular_198.pkl', 'Stanford_Nova_Regular_200.pkl', 'Stanford_Nova_Regular_202.pkl', 'Stanford_Nova_Regular_203.pkl', 'Stanford_Nova_Regular_204.pkl', 'Stanford_Nova_Regular_205.pkl', 'Stanford_Nova_Regular_206.pkl', 'Stanford_Nova_Regular_208.pkl', 'Stanford_Nova_Regular_210.pkl', 'Stanford_Nova_Regular_212.pkl', 'Stanford_Nova_Regular_213.pkl', 'Stanford_Nova_Regular_214.pkl', 'Stanford_Nova_Regular_215.pkl', 'Stanford_Nova_Regular_221.pkl', 'Stanford_Nova_Regular_222.pkl', 'Stanford_Nova_Regular_223.pkl', 'Stanford_Nova_Regular_224.pkl', 'Stanford_Nova_Regular_225.pkl', 'Stanford_Nova_Regular_227.pkl', 'Stanford_Nova_Regular_229.pkl', 'Stanford_Nova_Regular_230.pkl', 'Stanford_Nova_Regular_269.pkl', 'Stanford_Nova_Regular_270.pkl', 'Stanford_Nova_Regular_271.pkl', 'Stanford_Nova_Regular_272.pkl', 'Stanford_Nova_Regular_274.pkl', 'Stanford_Nova_Regular_275.pkl', 'Stanford_Nova_Regular_277.pkl', 'Stanford_Nova_Regular_278.pkl', 'Stanford_Nova_Regular_280.pkl', 'Stanford_Nova_Regular_282.pkl', 'Stanford_Nova_Regular_283.pkl', 'Stanford_Nova_Regular_286.pkl', 'Stanford_Nova_Regular_287.pkl', 'Stanford_Nova_Regular_290.pkl', 'Stanford_Nova_Regular_291.pkl', 'Stanford_Nova_Regular_293.pkl', 'Stanford_Nova_Regular_295.pkl', 'Stanford_Nova_Regular_299.pkl', 'Stanford_Nova_Regular_300.pkl', 'Stanford_Nova_Regular_301.pkl', 'Stanford_Nova_Regular_303.pkl', 'Stanford_Nova_Regular_304.pkl', 'Stanford_Nova_Regular_305.pkl', 'Stanford_Nova_Regular_306.pkl', 'Stanford_Nova_Regular_309.pkl', 'Stanford_Nova_Regular_310.pkl', 'Stanford_Nova_Regular_311.pkl', 'Stanford_Nova_Regular_312.pkl', 'Stanford_Nova_Regular_314.pkl', 'Stanford_Nova_Regular_315.pkl', 'Stanford_Nova_Regular_316.pkl', 'Stanford_Nova_Regular_317.pkl', 'Stanford_Nova_Regular_320.pkl', 'Stanford_Nova_Regular_322.pkl', 'Stanford_Nova_Regular_323.pkl', 'Stanford_Nova_Regular_324.pkl']
Stanford_val_files = ['Stanford_Nova_Regular_294.pkl', 'Stanford_Nova_Regular_288.pkl', 'Stanford_Nova_Regular_296.pkl', 'Stanford_Nova_Regular_139.pkl', 'Stanford_Nova_Regular_319.pkl', 'Stanford_Nova_Regular_177.pkl', 'Stanford_Nova_Regular_326.pkl', 'Stanford_Nova_Regular_297.pkl', 'Stanford_Nova_Regular_196.pkl', 'Stanford_Nova_Regular_146.pkl', 'Stanford_Nova_Regular_195.pkl', 'Stanford_Nova_Regular_101.pkl', 'Stanford_Nova_Regular_105.pkl', 'Stanford_Nova_Regular_179.pkl', 'Stanford_Nova_Regular_165.pkl', 'Stanford_Nova_Regular_157.pkl', 'Stanford_Nova_Regular_186.pkl', 'Stanford_Nova_Regular_273.pkl', 'Stanford_Nova_Regular_172.pkl', 'Stanford_Nova_Regular_199.pkl', 'Stanford_Nova_Regular_211.pkl', 'Stanford_Nova_Regular_162.pkl', 'Stanford_Nova_Regular_216.pkl', 'Stanford_Nova_Regular_276.pkl', 'Stanford_Nova_Regular_143.pkl', 'Stanford_Nova_Regular_106.pkl', 'Stanford_Nova_Regular_147.pkl', 'Stanford_Nova_Regular_110.pkl', 'Stanford_Nova_Regular_112.pkl', 'Stanford_Nova_Regular_128.pkl', 'Stanford_Nova_Regular_289.pkl', 'Stanford_Nova_Regular_302.pkl', 'Stanford_Nova_Regular_178.pkl', 'Stanford_Nova_Regular_217.pkl', 'Stanford_Nova_Regular_138.pkl', 'Stanford_Nova_Regular_325.pkl']
Stanford_test_files = ['Stanford_Nova_Regular_209.pkl', 'Stanford_Nova_Regular_307.pkl', 'Stanford_Nova_Regular_284.pkl', 'Stanford_Nova_Regular_175.pkl', 'Stanford_Nova_Regular_168.pkl', 'Stanford_Nova_Regular_308.pkl', 'Stanford_Nova_Regular_108.pkl', 'Stanford_Nova_Regular_220.pkl', 'Stanford_Nova_Regular_228.pkl', 'Stanford_Nova_Regular_292.pkl', 'Stanford_Nova_Regular_117.pkl', 'Stanford_Nova_Regular_185.pkl', 'Stanford_Nova_Regular_174.pkl', 'Stanford_Nova_Regular_180.pkl', 'Stanford_Nova_Regular_318.pkl', 'Stanford_Nova_Regular_120.pkl', 'Stanford_Nova_Regular_144.pkl', 'Stanford_Nova_Regular_145.pkl', 'Stanford_Nova_Regular_114.pkl', 'Stanford_Nova_Regular_127.pkl', 'Stanford_Nova_Regular_285.pkl', 'Stanford_Nova_Regular_313.pkl', 'Stanford_Nova_Regular_321.pkl', 'Stanford_Nova_Regular_281.pkl', 'Stanford_Nova_Regular_226.pkl', 'Stanford_Nova_Regular_130.pkl', 'Stanford_Nova_Regular_155.pkl', 'Stanford_Nova_Regular_219.pkl', 'Stanford_Nova_Regular_176.pkl', 'Stanford_Nova_Regular_279.pkl', 'Stanford_Nova_Regular_191.pkl', 'Stanford_Nova_Regular_201.pkl', 'Stanford_Nova_Regular_207.pkl', 'Stanford_Nova_Regular_129.pkl', 'Stanford_Nova_Regular_160.pkl', 'Stanford_Nova_Regular_167.pkl']

ISU_ILCC_train_files = ['ISU-ILCC_G23C3.pkl', 'ISU-ILCC_G64C4.pkl', 'ISU-ILCC_G50C1.pkl', 'ISU-ILCC_G36C2.pkl',
                        'ISU-ILCC_G7C4.pkl', 'ISU-ILCC_G16C2.pkl', 'ISU-ILCC_G13C2.pkl', 'ISU-ILCC_G29C3.pkl',
                        'ISU-ILCC_G41C3.pkl', 'ISU-ILCC_G39C4.pkl', 'ISU-ILCC_G36C4.pkl', 'ISU-ILCC_G28C3.pkl',
                        'ISU-ILCC_G18C1.pkl', 'ISU-ILCC_G58C1.pkl', 'ISU-ILCC_G8C2.pkl', 'ISU-ILCC_G16C3.pkl',
                        'ISU-ILCC_G27C2.pkl', 'ISU-ILCC_G22C1.pkl', 'ISU-ILCC_G43C1.pkl', 'ISU-ILCC_G53C2.pkl',
                        'ISU-ILCC_G32C4.pkl', 'ISU-ILCC_G44C3.pkl', 'ISU-ILCC_G38C1.pkl', 'ISU-ILCC_G61C3.pkl',
                        'ISU-ILCC_G43C2.pkl', 'ISU-ILCC_G20C2.pkl', 'ISU-ILCC_G17C1.pkl', 'ISU-ILCC_G33C4.pkl',
                        'ISU-ILCC_G60C1.pkl', 'ISU-ILCC_G27C3.pkl', 'ISU-ILCC_G45C3.pkl', 'ISU-ILCC_G18C4.pkl',
                        'ISU-ILCC_G50C4.pkl', 'ISU-ILCC_G1C1.pkl', 'ISU-ILCC_G19C4.pkl', 'ISU-ILCC_G49C4.pkl',
                        'ISU-ILCC_G3C3.pkl', 'ISU-ILCC_G19C2.pkl', 'ISU-ILCC_G52C4.pkl', 'ISU-ILCC_G31C1.pkl',
                        'ISU-ILCC_G47C2.pkl', 'ISU-ILCC_G43C4.pkl', 'ISU-ILCC_G47C1.pkl', 'ISU-ILCC_G40C2.pkl',
                        'ISU-ILCC_G20C4.pkl', 'ISU-ILCC_G6C2.pkl', 'ISU-ILCC_G32C3.pkl', 'ISU-ILCC_G41C4.pkl',
                        'ISU-ILCC_G60C2.pkl', 'ISU-ILCC_G64C3.pkl', 'ISU-ILCC_G9C2.pkl', 'ISU-ILCC_G49C3.pkl',
                        'ISU-ILCC_G46C1.pkl', 'ISU-ILCC_G30C1.pkl', 'ISU-ILCC_G29C4.pkl', 'ISU-ILCC_G50C2.pkl',
                        'ISU-ILCC_G51C1.pkl', 'ISU-ILCC_G55C2.pkl', 'ISU-ILCC_G17C4.pkl', 'ISU-ILCC_G50C3.pkl',
                        'ISU-ILCC_G25C1.pkl', 'ISU-ILCC_G4C4.pkl', 'ISU-ILCC_G13C4.pkl', 'ISU-ILCC_G46C3.pkl',
                        'ISU-ILCC_G38C2.pkl', 'ISU-ILCC_G34C1.pkl', 'ISU-ILCC_G45C4.pkl', 'ISU-ILCC_G27C4.pkl',
                        'ISU-ILCC_G35C1.pkl', 'ISU-ILCC_G21C4.pkl', 'ISU-ILCC_G24C1.pkl', 'ISU-ILCC_G40C4.pkl',
                        'ISU-ILCC_G42C3.pkl', 'ISU-ILCC_G39C1.pkl', 'ISU-ILCC_G14C1.pkl', 'ISU-ILCC_G61C2.pkl',
                        'ISU-ILCC_G63C3.pkl', 'ISU-ILCC_G23C4.pkl', 'ISU-ILCC_G36C1.pkl', 'ISU-ILCC_G57C2.pkl',
                        'ISU-ILCC_G55C1.pkl', 'ISU-ILCC_G4C1.pkl', 'ISU-ILCC_G7C3.pkl', 'ISU-ILCC_G12C4.pkl',
                        'ISU-ILCC_G34C2.pkl', 'ISU-ILCC_G7C1.pkl', 'ISU-ILCC_G51C4.pkl', 'ISU-ILCC_G4C2.pkl',
                        'ISU-ILCC_G44C1.pkl', 'ISU-ILCC_G46C4.pkl', 'ISU-ILCC_G42C2.pkl', 'ISU-ILCC_G28C2.pkl',
                        'ISU-ILCC_G12C3.pkl', 'ISU-ILCC_G51C2.pkl', 'ISU-ILCC_G62C4.pkl', 'ISU-ILCC_G30C2.pkl',
                        'ISU-ILCC_G28C1.pkl', 'ISU-ILCC_G56C2.pkl', 'ISU-ILCC_G23C2.pkl', 'ISU-ILCC_G52C1.pkl',
                        'ISU-ILCC_G37C3.pkl', 'ISU-ILCC_G34C3.pkl', 'ISU-ILCC_G57C3.pkl', 'ISU-ILCC_G32C2.pkl',
                        'ISU-ILCC_G38C3.pkl', 'ISU-ILCC_G18C3.pkl', 'ISU-ILCC_G10C3.pkl', 'ISU-ILCC_G40C3.pkl',
                        'ISU-ILCC_G14C3.pkl', 'ISU-ILCC_G1C4.pkl', 'ISU-ILCC_G29C2.pkl', 'ISU-ILCC_G17C3.pkl',
                        'ISU-ILCC_G45C1.pkl', 'ISU-ILCC_G62C3.pkl', 'ISU-ILCC_G31C3.pkl', 'ISU-ILCC_G25C3.pkl',
                        'ISU-ILCC_G44C4.pkl', 'ISU-ILCC_G4C3.pkl', 'ISU-ILCC_G44C2.pkl', 'ISU-ILCC_G62C1.pkl',
                        'ISU-ILCC_G30C3.pkl', 'ISU-ILCC_G56C3.pkl', 'ISU-ILCC_G47C4.pkl', 'ISU-ILCC_G25C2.pkl',
                        'ISU-ILCC_G42C1.pkl', 'ISU-ILCC_G20C1.pkl', 'ISU-ILCC_G59C1.pkl', 'ISU-ILCC_G38C4.pkl',
                        'ISU-ILCC_G9C3.pkl', 'ISU-ILCC_G35C3.pkl', 'ISU-ILCC_G8C4.pkl', 'ISU-ILCC_G53C1.pkl',
                        'ISU-ILCC_G30C4.pkl', 'ISU-ILCC_G20C3.pkl', 'ISU-ILCC_G9C1.pkl', 'ISU-ILCC_G21C1.pkl',
                        'ISU-ILCC_G3C2.pkl', 'ISU-ILCC_G5C1.pkl', 'ISU-ILCC_G1C3.pkl', 'ISU-ILCC_G62C2.pkl',
                        'ISU-ILCC_G58C3.pkl', 'ISU-ILCC_G21C2.pkl', 'ISU-ILCC_G28C4.pkl', 'ISU-ILCC_G24C2.pkl']
ISU_ILCC_val_files = ['ISU-ILCC_G57C1.pkl', 'ISU-ILCC_G31C4.pkl', 'ISU-ILCC_G6C1.pkl', 'ISU-ILCC_G39C3.pkl',
                      'ISU-ILCC_G7C2.pkl', 'ISU-ILCC_G16C4.pkl', 'ISU-ILCC_G10C4.pkl', 'ISU-ILCC_G10C2.pkl',
                      'ISU-ILCC_G64C1.pkl', 'ISU-ILCC_G35C2.pkl', 'ISU-ILCC_G12C1.pkl', 'ISU-ILCC_G49C2.pkl',
                      'ISU-ILCC_G54C2.pkl', 'ISU-ILCC_G19C3.pkl', 'ISU-ILCC_G33C1.pkl', 'ISU-ILCC_G63C1.pkl',
                      'ISU-ILCC_G34C4.pkl', 'ISU-ILCC_G54C1.pkl', 'ISU-ILCC_G6C4.pkl', 'ISU-ILCC_G45C2.pkl',
                      'ISU-ILCC_G13C3.pkl', 'ISU-ILCC_G59C3.pkl', 'ISU-ILCC_G48C2.pkl', 'ISU-ILCC_G48C1.pkl',
                      'ISU-ILCC_G60C4.pkl', 'ISU-ILCC_G55C4.pkl', 'ISU-ILCC_G46C2.pkl', 'ISU-ILCC_G49C1.pkl',
                      'ISU-ILCC_G10C1.pkl', 'ISU-ILCC_G31C2.pkl', 'ISU-ILCC_G57C4.pkl', 'ISU-ILCC_G22C4.pkl',
                      'ISU-ILCC_G64C2.pkl', 'ISU-ILCC_G32C1.pkl', 'ISU-ILCC_G52C2.pkl', 'ISU-ILCC_G17C2.pkl',
                      'ISU-ILCC_G18C2.pkl', 'ISU-ILCC_G59C2.pkl', 'ISU-ILCC_G2C1.pkl', 'ISU-ILCC_G47C3.pkl',
                      'ISU-ILCC_G48C4.pkl', 'ISU-ILCC_G27C1.pkl', 'ISU-ILCC_G19C1.pkl', 'ISU-ILCC_G36C3.pkl',
                      'ISU-ILCC_G21C3.pkl', 'ISU-ILCC_G53C4.pkl', 'ISU-ILCC_G63C2.pkl', 'ISU-ILCC_G29C1.pkl']
ISU_ILCC_test_files = ['ISU-ILCC_G24C4.pkl', 'ISU-ILCC_G54C3.pkl', 'ISU-ILCC_G14C4.pkl', 'ISU-ILCC_G35C4.pkl',
                       'ISU-ILCC_G24C3.pkl', 'ISU-ILCC_G41C2.pkl', 'ISU-ILCC_G56C4.pkl', 'ISU-ILCC_G5C3.pkl',
                       'ISU-ILCC_G41C1.pkl', 'ISU-ILCC_G6C3.pkl', 'ISU-ILCC_G2C4.pkl', 'ISU-ILCC_G53C3.pkl',
                       'ISU-ILCC_G59C4.pkl', 'ISU-ILCC_G48C3.pkl', 'ISU-ILCC_G40C1.pkl', 'ISU-ILCC_G8C3.pkl',
                       'ISU-ILCC_G43C3.pkl', 'ISU-ILCC_G39C2.pkl', 'ISU-ILCC_G2C2.pkl', 'ISU-ILCC_G2C3.pkl',
                       'ISU-ILCC_G51C3.pkl', 'ISU-ILCC_G63C4.pkl', 'ISU-ILCC_G33C3.pkl', 'ISU-ILCC_G12C2.pkl',
                       'ISU-ILCC_G22C2.pkl', 'ISU-ILCC_G52C3.pkl', 'ISU-ILCC_G5C4.pkl', 'ISU-ILCC_G3C1.pkl',
                       'ISU-ILCC_G22C3.pkl', 'ISU-ILCC_G55C3.pkl', 'ISU-ILCC_G54C4.pkl', 'ISU-ILCC_G5C2.pkl',
                       'ISU-ILCC_G61C4.pkl', 'ISU-ILCC_G16C1.pkl', 'ISU-ILCC_G37C1.pkl', 'ISU-ILCC_G58C4.pkl',
                       'ISU-ILCC_G23C1.pkl', 'ISU-ILCC_G37C4.pkl', 'ISU-ILCC_G58C2.pkl', 'ISU-ILCC_G8C1.pkl',
                       'ISU-ILCC_G14C2.pkl', 'ISU-ILCC_G37C2.pkl', 'ISU-ILCC_G56C1.pkl', 'ISU-ILCC_G33C2.pkl',
                       'ISU-ILCC_G3C4.pkl', 'ISU-ILCC_G60C3.pkl', 'ISU-ILCC_G13C1.pkl', 'ISU-ILCC_G1C2.pkl']

XJTU_train_files = ['XJTU_2C_battery-6.pkl', 'XJTU_2C_battery-2.pkl', 'XJTU_3C_battery-2.pkl', 'XJTU_3C_battery-3.pkl',
                    'XJTU_3C_battery-10.pkl', 'XJTU_3C_battery-13.pkl', 'XJTU_3C_battery-7.pkl',
                    'XJTU_3C_battery-15.pkl', 'XJTU_3C_battery-4.pkl', 'XJTU_3C_battery-5.pkl',
                    'XJTU_3C_battery-11.pkl', 'XJTU_3C_battery-6.pkl', 'XJTU_2C_battery-1.pkl', 'XJTU_2C_battery-8.pkl',
                    'XJTU_2C_battery-4.pkl']
XJTU_val_files = ['XJTU_3C_battery-14.pkl', 'XJTU_3C_battery-8.pkl', 'XJTU_2C_battery-3.pkl', 'XJTU_2C_battery-7.pkl']
XJTU_test_files = ['XJTU_3C_battery-12.pkl', 'XJTU_3C_battery-1.pkl', 'XJTU_2C_battery-5.pkl', 'XJTU_3C_battery-9.pkl']

# ZNcoin_train_files = ['ZN-coin_202_20231213213655_03_3.pkl', 'ZN-coin_202_20231213213655_03_4.pkl',
#                       'ZN-coin_202_20231213213655_03_5.pkl', 'ZN-coin_204-1_20231205230212_07_1.pkl',
#                       'ZN-coin_204-3_20231205230221_07_3.pkl', 'ZN-coin_205-2_20231205230234_07_5.pkl',
#                       'ZN-coin_402-2_20231209225727_01_2.pkl', 'ZN-coin_403-1_20231209225922_01_4.pkl',
#                       'ZN-coin_404-3_20231209231250_08_1.pkl', 'ZN-coin_405-1_20231209231331_08_2.pkl',
#                       'ZN-coin_405-2_20231209231413_08_3.pkl', 'ZN-coin_405-3_20231209231450_08_4.pkl',
#                       'ZN-coin_407-1_20231209231725_08_8.pkl', 'ZN-coin_407-3_20231209231841_02_2.pkl',
#                       'ZN-coin_408-1_20231209231918_02_3.pkl', 'ZN-coin_408-2_20231209231947_02_4.pkl',
#                       'ZN-coin_408-3_20231209232028_05_1.pkl', 'ZN-coin_409-1_20231209232338_05_2.pkl',
#                       'ZN-coin_409-2_20231209232422_05_3.pkl', 'ZN-coin_409-3_20231209232500_05_4.pkl',
#                       'ZN-coin_410-1_20231209232559_09_1.pkl', 'ZN-coin_410-3_20231209232707_09_3.pkl',
#                       'ZN-coin_412-3_20231209233120_06_1.pkl', 'ZN-coin_414-2_20231209233354_06_6.pkl',
#                       'ZN-coin_415-2_20231209233606_10_1.pkl', 'ZN-coin_416-3_20231209233856_10_5.pkl',
#                       'ZN-coin_417-3_20231209234058_10_8.pkl', 'ZN-coin_418-1_20231209234141_11_1.pkl',
#                       'ZN-coin_418-3_20231209234252_11_3.pkl', 'ZN-coin_420-3_20231205230017_01_3.pkl',
#                       'ZN-coin_422-3_20231205230049_02_1.pkl', 'ZN-coin_423-1_20231205230055_02_2.pkl',
#                       'ZN-coin_425-2_20231205230124_03_1.pkl', 'ZN-coin_428-1_20231212185048_01_2.pkl',
#                       'ZN-coin_429-2_20231212185157_01_8.pkl', 'ZN-coin_430-1_20231212185250_02_6.pkl',
#                       'ZN-coin_432-2_20231227204437_01_2.pkl', 'ZN-coin_433-1_20231227204534_01_4.pkl',
#                       'ZN-coin_433-2_20231227204539_01_5.pkl', 'ZN-coin_434-1_20231227204606_01_7.pkl',
#                       'ZN-coin_434-2_20231227204612_01_8.pkl', 'ZN-coin_434-3_20231227204618_03_1.pkl',
#                       'ZN-coin_435-2_20231227204630_03_3.pkl', 'ZN-coin_435-3_20231227204635_03_4.pkl',
#                       'ZN-coin_436-3_20231227204657_03_7.pkl', 'ZN-coin_437-1_20231227204706_03_8.pkl',
#                       'ZN-coin_437-3_20231227204717_04_2.pkl', 'ZN-coin_438-1_20231227204743_04_3.pkl',
#                       'ZN-coin_438-2_20231227204748_04_4.pkl', 'ZN-coin_439-2_20231227204810_04_7.pkl',
#                       'ZN-coin_439-3_20231227204817_04_8.pkl', 'ZN-coin_440-2_20231227204832_08_2.pkl',
#                       'ZN-coin_442-1_20240104212418_09_1.pkl', 'ZN-coin_442-3_20240104212433_09_3.pkl',
#                       'ZN-coin_443-2_20240104212500_09_5.pkl', 'ZN-coin_445-1_20240104212517_09_7.pkl',
#                       'ZN-coin_450-1_20240116203402_01_2_Batch-3.pkl', 'ZN-coin_450-2_20240116203410_01_4_Batch-3.pkl',
#                       'ZN-coin_450-3_20240116203417_03_3_Batch-3.pkl', 'ZN-coin_451-1_20240116203425_03_4_Batch-3.pkl']
# ZNcoin_val_files = ['ZN-coin_442-2_20240104212424_09_2.pkl', 'ZN-coin_420-1_20231205230010_01_1.pkl',
#                     'ZN-coin_209-2_20231205230252_07_8.pkl', 'ZN-coin_441-1_20231227204855_08_4.pkl',
#                     'ZN-coin_437-2_20231227204712_04_1.pkl', 'ZN-coin_406-1_20231209231531_08_5.pkl',
#                     'ZN-coin_411-1_20231209232756_09_4.pkl', 'ZN-coin_436-1_20231227204646_03_5.pkl',
#                     'ZN-coin_418-2_20231209234209_11_2.pkl', 'ZN-coin_440-3_20231227204837_08_3.pkl',
#                     'ZN-coin_429-1_20231212185129_01_5.pkl', 'ZN-coin_432-3_20231227204518_01_3.pkl',
#                     'ZN-coin_433-3_20231227204544_01_6.pkl', 'ZN-coin_445-2_20240104212521_09_8.pkl',
#                     'ZN-coin_414-3_20231209233430_06_7.pkl', 'ZN-coin_416-2_20231209233822_10_4.pkl',
#                     'ZN-coin_445-3_20240104212530_07_1.pkl', 'ZN-coin_440-1_20231227204827_08_1.pkl',
#                     'ZN-coin_415-3_20231209233637_10_2.pkl', 'ZN-coin_402-1_20231209225636_01_1.pkl']
# ZNcoin_test_files = ['ZN-coin_422-1_20231205230039_01_7.pkl', 'ZN-coin_438-3_20231227204754_04_5.pkl',
#                      'ZN-coin_435-1_20231227204625_03_2.pkl', 'ZN-coin_412-2_20231209233028_09_8.pkl',
#                      'ZN-coin_410-2_20231209232626_09_2.pkl', 'ZN-coin_439-1_20231227204804_04_6.pkl',
#                      'ZN-coin_204-2_20231205230217_07_2.pkl', 'ZN-coin_428-2_20231212185058_01_4.pkl',
#                      'ZN-coin_430-2_20231212185305_02_7.pkl', 'ZN-coin_436-2_20231227204653_03_6.pkl',
#                      'ZN-coin_205-3_20231205230239_07_6.pkl', 'ZN-coin_415-1_20231209233508_06_8.pkl',
#                      'ZN-coin_412-1_20231209232958_09_7.pkl', 'ZN-coin_413-1_20231209233202_06_2.pkl',
#                      'ZN-coin_209-1_20231205230248_07_7.pkl', 'ZN-coin_406-2_20231209231604_08_6.pkl',
#                      'ZN-coin_406-3_20231209231637_08_7.pkl', 'ZN-coin_205-1_20231205230230_07_4.pkl',
#                      'ZN-coin_446-1_20240104212538_07_2.pkl', 'ZN-coin_402-3_20231209225844_01_3.pkl']

ZNcoin_train_files = ['ZN-coin_202_20231213213655_03_3.pkl', 'ZN-coin_202_20231213213655_03_4.pkl', 'ZN-coin_202_20231213213655_03_5.pkl', 'ZN-coin_204-1_20231205230212_07_1.pkl', 'ZN-coin_204-3_20231205230221_07_3.pkl', 'ZN-coin_205-2_20231205230234_07_5.pkl', 'ZN-coin_402-2_20231209225727_01_2.pkl', 'ZN-coin_403-1_20231209225922_01_4.pkl', 'ZN-coin_404-3_20231209231250_08_1.pkl', 'ZN-coin_405-1_20231209231331_08_2.pkl', 'ZN-coin_405-2_20231209231413_08_3.pkl', 'ZN-coin_405-3_20231209231450_08_4.pkl', 'ZN-coin_407-1_20231209231725_08_8.pkl', 'ZN-coin_407-3_20231209231841_02_2.pkl', 'ZN-coin_408-1_20231209231918_02_3.pkl', 'ZN-coin_408-2_20231209231947_02_4.pkl', 'ZN-coin_408-3_20231209232028_05_1.pkl', 'ZN-coin_409-1_20231209232338_05_2.pkl', 'ZN-coin_409-2_20231209232422_05_3.pkl', 'ZN-coin_409-3_20231209232500_05_4.pkl', 'ZN-coin_410-1_20231209232559_09_1.pkl', 'ZN-coin_410-3_20231209232707_09_3.pkl', 'ZN-coin_412-3_20231209233120_06_1.pkl', 'ZN-coin_414-2_20231209233354_06_6.pkl', 'ZN-coin_415-2_20231209233606_10_1.pkl', 'ZN-coin_416-3_20231209233856_10_5.pkl', 'ZN-coin_417-3_20231209234058_10_8.pkl', 'ZN-coin_418-1_20231209234141_11_1.pkl', 'ZN-coin_418-3_20231209234252_11_3.pkl', 'ZN-coin_420-3_20231205230017_01_3.pkl', 'ZN-coin_422-3_20231205230049_02_1.pkl', 'ZN-coin_423-1_20231205230055_02_2.pkl', 'ZN-coin_425-2_20231205230124_03_1.pkl', 'ZN-coin_428-1_20231212185048_01_2.pkl', 'ZN-coin_429-2_20231212185157_01_8.pkl', 'ZN-coin_430-1_20231212185250_02_6.pkl', 'ZN-coin_432-2_20231227204437_01_2.pkl', 'ZN-coin_433-1_20231227204534_01_4.pkl', 'ZN-coin_433-2_20231227204539_01_5.pkl', 'ZN-coin_434-1_20231227204606_01_7.pkl', 'ZN-coin_434-2_20231227204612_01_8.pkl', 'ZN-coin_434-3_20231227204618_03_1.pkl', 'ZN-coin_435-2_20231227204630_03_3.pkl', 'ZN-coin_435-3_20231227204635_03_4.pkl', 'ZN-coin_436-3_20231227204657_03_7.pkl', 'ZN-coin_437-1_20231227204706_03_8.pkl', 'ZN-coin_437-3_20231227204717_04_2.pkl', 'ZN-coin_438-1_20231227204743_04_3.pkl', 'ZN-coin_438-2_20231227204748_04_4.pkl', 'ZN-coin_439-2_20231227204810_04_7.pkl', 'ZN-coin_439-3_20231227204817_04_8.pkl', 'ZN-coin_440-2_20231227204832_08_2.pkl', 'ZN-coin_442-1_20240104212418_09_1.pkl', 'ZN-coin_442-3_20240104212433_09_3.pkl', 'ZN-coin_443-2_20240104212500_09_5.pkl', 'ZN-coin_445-1_20240104212517_09_7.pkl', 'ZN-coin_450-1_20240116203402_01_2_Batch-3.pkl', 'ZN-coin_450-2_20240116203410_01_4_Batch-3.pkl', 'ZN-coin_450-3_20240116203417_03_3_Batch-3.pkl', 'ZN-coin_451-1_20240116203425_03_4_Batch-3.pkl']
ZNcoin_val_files = ['ZN-coin_442-2_20240104212424_09_2.pkl', 'ZN-coin_420-1_20231205230010_01_1.pkl', 'ZN-coin_209-2_20231205230252_07_8.pkl', 'ZN-coin_441-1_20231227204855_08_4.pkl', 'ZN-coin_437-2_20231227204712_04_1.pkl', 'ZN-coin_406-1_20231209231531_08_5.pkl', 'ZN-coin_411-1_20231209232756_09_4.pkl', 'ZN-coin_436-1_20231227204646_03_5.pkl', 'ZN-coin_418-2_20231209234209_11_2.pkl', 'ZN-coin_440-3_20231227204837_08_3.pkl', 'ZN-coin_429-1_20231212185129_01_5.pkl', 'ZN-coin_432-3_20231227204518_01_3.pkl', 'ZN-coin_433-3_20231227204544_01_6.pkl', 'ZN-coin_445-2_20240104212521_09_8.pkl', 'ZN-coin_414-3_20231209233430_06_7.pkl', 'ZN-coin_416-2_20231209233822_10_4.pkl', 'ZN-coin_445-3_20240104212530_07_1.pkl', 'ZN-coin_440-1_20231227204827_08_1.pkl', 'ZN-coin_415-3_20231209233637_10_2.pkl', 'ZN-coin_402-1_20231209225636_01_1.pkl']
ZNcoin_test_files = ['ZN-coin_422-1_20231205230039_01_7.pkl', 'ZN-coin_438-3_20231227204754_04_5.pkl', 'ZN-coin_435-1_20231227204625_03_2.pkl', 'ZN-coin_412-2_20231209233028_09_8.pkl', 'ZN-coin_410-2_20231209232626_09_2.pkl', 'ZN-coin_439-1_20231227204804_04_6.pkl', 'ZN-coin_204-2_20231205230217_07_2.pkl', 'ZN-coin_428-2_20231212185058_01_4.pkl', 'ZN-coin_430-2_20231212185305_02_7.pkl', 'ZN-coin_436-2_20231227204653_03_6.pkl', 'ZN-coin_205-3_20231205230239_07_6.pkl', 'ZN-coin_415-1_20231209233508_06_8.pkl', 'ZN-coin_412-1_20231209232958_09_7.pkl', 'ZN-coin_413-1_20231209233202_06_2.pkl', 'ZN-coin_209-1_20231205230248_07_7.pkl', 'ZN-coin_406-2_20231209231604_08_6.pkl', 'ZN-coin_406-3_20231209231637_08_7.pkl', 'ZN-coin_205-1_20231205230230_07_4.pkl', 'ZN-coin_446-1_20240104212538_07_2.pkl', 'ZN-coin_402-3_20231209225844_01_3.pkl']


CALB_train_files = ['CALB_35_B229.pkl', 'CALB_35_B173.pkl', 'CALB_35_B228.pkl', 'CALB_0_B184.pkl', 'CALB_35_B248.pkl',
                    'CALB_35_B227.pkl', 'CALB_0_B185.pkl', 'CALB_35_B249.pkl', 'CALB_35_B223.pkl', 'CALB_35_B224.pkl',
                    'CALB_0_B189.pkl', 'CALB_35_B250.pkl', 'CALB_0_B188.pkl', 'CALB_45_B256.pkl', 'CALB_0_B183.pkl',
                    'CALB_35_B175.pkl', 'CALB_0_B190.pkl']
CALB_val_files = ['CALB_0_B187.pkl', 'CALB_35_B222.pkl', 'CALB_25_T25-2.pkl', 'CALB_35_B247.pkl', 'CALB_45_B253.pkl']
CALB_test_files = ['CALB_0_B182.pkl', 'CALB_25_T25-1.pkl', 'CALB_35_B174.pkl', 'CALB_35_B230.pkl', 'CALB_45_B255.pkl']

NAion_2021_train_files = ['NA-ion_270040-1-2-63.pkl', 'NA-ion_270040-1-5-60.pkl', 'NA-ion_270040-1-7-58.pkl',
                          'NA-ion_270040-1-8-57.pkl', 'NA-ion_270040-2-2-12.pkl', 'NA-ion_270040-2-5-12.pkl',
                          'NA-ion_270040-3-1-56.pkl', 'NA-ion_270040-3-5-52.pkl', 'NA-ion_270040-5-2-38.pkl',
                          'NA-ion_270040-3-8-49.pkl', 'NA-ion_270040-5-1-39.pkl', 'NA-ion_270040-5-3-37.pkl',
                          'NA-ion_270040-5-6-34.pkl', 'NA-ion_270040-5-7-33.pkl', 'NA-ion_270040-6-2-30.pkl',
                          'NA-ion_270040-6-6-26.pkl', 'NA-ion_270040-7-1-23.pkl', 'NA-ion_270040-8-5-16.pkl',
                          'NA-ion_270040-3-3-54.pkl', 'NA-ion_270040-6-8-24.pkl']
NAion_2021_val_files = ['NA-ion_270040-4-2-47.pkl', 'NA-ion_270040-4-6-43.pkl', 'NA-ion_270040-5-5-35.pkl',
                        'NA-ion_270040-1-6-59.pkl', 'NA-ion_270040-3-4-53.pkl', 'NA-ion_270040-3-2-55.pkl']
NAion_2021_test_files = ['NA-ion_270040-5-8-32.pkl', 'NA-ion_270040-4-3-46.pkl', 'NA-ion_270040-4-1-48.pkl',
                         'NA-ion_270040-1-3-62.pkl', 'NA-ion_270040-3-7-50.pkl']

MIX_large_train_files = HUST_train_files + MATR_train_files + SNL_train_files + RWTH_train_files + MICH_train_files + MICH_EXP_train_files + UL_PUR_train_files + CALCE_train_files + HNEI_train_files + Tongji_train_files + Stanford_train_files + ISU_ILCC_train_files + XJTU_train_files
MIX_large_val_files = HUST_val_files + MATR_val_files + SNL_val_files + RWTH_val_files + MICH_val_files + MICH_EXP_val_files + UL_PUR_val_files + CALCE_val_files + HNEI_val_files + Tongji_val_files + Stanford_val_files + ISU_ILCC_val_files + XJTU_val_files
MIX_large_test_files = HUST_test_files + MATR_test_files + SNL_test_files + RWTH_test_files + MICH_test_files + MICH_EXP_test_files + UL_PUR_test_files + CALCE_test_files + HNEI_test_files + Tongji_test_files + Stanford_test_files + ISU_ILCC_test_files + XJTU_test_files

train_files = MIX_large_train_files + ZNcoin_train_files + CALB_train_files + NAion_2021_train_files
val_files = MIX_large_val_files + ZNcoin_val_files + CALB_val_files + NAion_2021_val_files
test_files = MIX_large_test_files + ZNcoin_test_files + CALB_test_files + NAion_2021_test_files
all_files = train_files + val_files + test_files

def relabel_dict_values(d):
    unique_values = sorted(set(d.values()))
    value_mapping = {old_value: new_index + 1 for new_index, old_value in enumerate(unique_values)}
    new_d = {k: value_mapping[v] for k, v in d.items()}

    return new_d

# label_path = '/data/trf/python_works/BatteryLife/dataset/Life labels'
label_path = '/data/trf/python_works/PBT_BatteryLife/dataset/Life labels'
label_files_path = os.listdir(label_path)
label_json_files = [i for i in label_files_path if i.endswith('.json')]
label_names = []
for file in label_json_files:
    if file.startswith('Stanford_labels'):
        continue

    with open(os.path.join(label_path, file), 'r') as f:
        label_data = json.load(f)
    print(file, len(label_data))
    for key, value in label_data.items():
        filename = key.split('.pkl')[0]
        label_names.append(filename)

processed_files = []
for file in all_files:
    filename = file.split('.pkl')[0]
    if filename.startswith('Tongji'):
        filename = filename.replace('--', '-#')
    if filename in label_names:
        processed_files.append(file)

protocols = {}
for file in tqdm(processed_files):
    if len(protocols) == 0:
        max_value = 0
    else:
        max_value = max(protocols.values())

    if 'CALCE' in file:
        if 'CS' in file:
            if '33' in file or '34' in file:
                protocols[file] = 1
            else:
                protocols[file] = 2
        elif 'CX' in file:
            if '16' in file or '33' in file or '35' in file:
                protocols[file] = 3
            else:
                protocols[file] = 4
    elif 'HNEI' in file:
        if 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_a' in file:
            protocols[file] = 5
        else:
            protocols[file] = 6
    elif 'MATR' in file:
        if 'b4c0.' in file or 'b4c1.' in file or 'b4c7.' in file or 'b4c17.' in file or 'b4c39.' in file:
            protocols[file] = 7
        elif 'b4c2.' in file or 'b4c22.' in file or 'b4c23.' in file or 'b4c25.' in file or 'b4c35.' in file or 'b4c41.' in file:
            protocols[file] = 8
        elif 'b4c3.' in file or 'b4c4.' in file or 'b4c34.' in file or 'b4c38.' in file or 'b4c42.' in file:
            protocols[file] = 9
        elif 'b4c5.' in file or 'b4c12.' in file or 'b4c15.' in file or 'b4c29.' in file or 'b4c44.' in file:
            protocols[file] = 10
        elif 'b4c6.' in file or 'b4c18.' in file or 'b4c19.' in file or 'b4c28.' in file or 'b4c33.' in file:
            protocols[file] = 11
        elif 'b4c8.' in file or 'b4c9.' in file or 'b4c24.' in file or 'b4c26.' in file:
            protocols[file] =12
        elif 'b4c10.' in file or 'b4c13.' in file or 'b4c20.' in file or 'b4c21.' in file or 'b4c43.' in file:
            protocols[file] = 13
        elif 'b4c11.' in file or 'b4c14.' in file or 'b4c27.' in file or 'b4c32.' in file or 'b4c36.' in file:
            protocols[file] = 14
        elif 'b4c16.' in file or 'b4c30.' in file or 'b4c31.' in file or 'b4c37.' in file or 'b4c40.' in file:
            protocols[file] = 15
        elif 'b1c0.' in file or 'b1c1.' in file or 'b1c2.' in file:
            protocols[file] = 16
        elif 'b1c3.' in file or 'b1c4.' in file:
            protocols[file] = 17
        elif 'b1c5.' in file:
            protocols[file] = 18
        elif 'b1c6.' in file or 'b1c7.' in file:
            protocols[file] = 19
        elif 'b1c9.' in file:
            protocols[file] = 20
        elif 'b1c11.' in file:
            protocols[file] = 21
        elif 'b1c14.' in file:
            protocols[file] = 22
        elif 'b1c15.' in file:
            protocols[file] = 23
        elif 'b1c16.' in file:
            protocols[file] = 24
        elif 'b1c17.' in file:
            protocols[file] = 25
        elif 'b1c18.' in file or 'b1c19.' in file:
            protocols[file] =26
        elif 'b1c20.' in file or 'b1c21.' in file:
            protocols[file] =27
        elif 'b1c23.' in file:
            protocols[file] = 28
        elif 'b1c24.' in file or 'b1c25.' in file:
            protocols[file] = 29
        elif 'b1c26.' in file or 'b1c27.' in file:
            protocols[file] = 30
        elif 'b1c28.' in file or 'b1c29.' in file:
            protocols[file] = 31
        elif 'b1c30.' in file or 'b1c31.' in file:
            protocols[file] = 32
        elif 'b1c32.' in file or 'b1c33.' in file or 'b2c47.' in file:
            protocols[file] = 33
        elif 'b1c34.' in file or 'b1c35.' in file:
            protocols[file] = 34
        elif 'b1c36.' in file or 'b1c37.' in file:
            protocols[file] = 35
        elif 'b1c38.' in file or 'b1c39.' in file:
            protocols[file] = 36
        elif 'b1c40.' in file or 'b1c41.' in file:
            protocols[file] =37
        elif 'b1c42.' in file:
            protocols[file] = 38
        elif 'b1c43.' in file or 'b1c44.' in file or 'b1c45.' in file:
            protocols[file] = 39
        elif 'b2c0.' in file:
            protocols[file] =40
        elif 'b2c1.' in file:
            protocols[file] = 41
        elif 'b2c2.' in file:
            protocols[file] = 42
        elif 'b2c3.' in file:
            protocols[file] = 43
        elif 'b2c4.' in file:
            protocols[file] = 44
        elif 'b2c5.' in file:
            protocols[file] = 45
        elif 'b2c6.' in file:
            protocols[file] = 46
        elif 'b2c10.' in file:
            protocols[file] = 47
        elif 'b2c11.' in file:
            protocols[file] = 48
        elif 'b2c12.' in file:
            protocols[file] = 49
        elif 'b2c13.' in file:
            protocols[file] = 50
        elif 'b2c14.' in file:
            protocols[file] = 51
        elif 'b2c17.' in file:
            protocols[file] = 52
        elif 'b2c18.' in file:
            protocols[file] = 53
        elif 'b2c19.' in file:
            protocols[file] = 54
        elif 'b2c20.' in file:
            protocols[file] = 55
        elif 'b2c21.' in file:
            protocols[file] = 56
        elif 'b2c22.' in file:
            protocols[file] = 57
        elif 'b2c23.' in file:
            protocols[file] = 58
        elif 'b2c24.' in file or 'b2c25.' in file or 'b2c26.' in file or 'b3c7.' in file or 'b3c16.' in file or 'b3c45.' in file or 'b3c46.' in file:
            protocols[file] = 59
        elif 'b2c27.' in file:
            protocols[file] =60
        elif 'b2c28.' in file:
            protocols[file] = 61
        elif 'b2c29.' in file:
            protocols[file] = 62
        elif 'b2c30.' in file:
            protocols[file] = 63
        elif 'b2c31.' in file:
            protocols[file] = 64
        elif 'b2c32.' in file:
            protocols[file] = 65
        elif 'b2c33.' in file:
            protocols[file] = 66
        elif 'b2c34.' in file:
            protocols[file] = 67
        elif 'b2c35.' in file:
            protocols[file] =68
        elif 'b2c36.' in file:
            protocols[file] = 69
        elif 'b2c37.' in file:
            protocols[file] = 70
        elif 'b2c38.' in file:
            protocols[file] = 71
        elif 'b2c39.' in file:
            protocols[file] = 72
        elif 'b2c40.' in file:
            protocols[file] = 73
        elif 'b2c41.' in file:
            protocols[file] = 74
        elif 'b2c42.' in file:
            protocols[file] = 75
        elif 'b2c43.' in file:
            protocols[file] = 76
        elif 'b2c44.' in file:
            protocols[file] = 77
        elif 'b2c45.' in file:
            protocols[file] = 78
        elif 'b2c46.' in file:
            protocols[file] = 79
        elif 'b3c0.' in file or 'b3c8.' in file or 'b3c20.' in file or 'b3c24.' in file or 'b3c33.' in file or 'b3c38.' in file:
            protocols[file] = 80
        elif 'b3c1.' in file or 'b3c9.' in file or 'b3c17.' in file or 'b3c25.' in file or 'b3c30.' in file or 'b3c34.' in file or 'b3c39.' in file or 'b3c44.' in file:
            protocols[file] = 81
        elif 'b3c3.' in file or 'b3c5.' in file or 'b3c12.' in file or 'b3c14.' in file or 'b3c19.' in file or 'b3c27.' in file or 'b3c36.' in file or 'b3c41.' in file:
            protocols[file] = 82
        elif 'b3c4.' in file or 'b3c10.' in file or 'b3c11.' in file or 'b3c13.' in file or 'b3c26.' in file or 'b3c35.' in file or 'b3c40.' in file:
            protocols[file] = 83
        elif 'b3c6.' in file or 'b3c21.' in file or 'b3c28.' in file:
            protocols[file] = 84
        elif 'b3c15.' in file or 'b3c29.' in file:
            protocols[file] = 85
        elif 'b3c18.' in file:
            protocols[file] = 86
        elif 'b3c22.' in file or 'b3c31.' in file:
            protocols[file] = 87
    elif 'UL-PUR' in file:
        protocols[file] = 88
    elif 'SNL' in file:
        if 'SNL_18650_LFP_25C_0-100_0.5-3C_a' in file or 'SNL_18650_LFP_25C_0-100_0.5-3C_b' in file or 'SNL_18650_LFP_25C_0-100_0.5-3C_c' in file or 'SNL_18650_LFP_25C_0-100_0.5-3C_d' in file:
            protocols[file] = 89
        elif 'SNL_18650_LFP_35C_0-100_0.5-1C_b' in file or 'SNL_18650_LFP_35C_0-100_0.5-1C_c' in file or 'SNL_18650_LFP_35C_0-100_0.5-1C_d' in file:
            protocols[file] = 90
        elif 'SNL_18650_LFP_35C_0-100_0.5-2C_a' in file or 'SNL_18650_LFP_35C_0-100_0.5-2C_b' in file:
            protocols[file] = 91
        elif 'SNL_18650_NCA_15C_0-100_0.5-1C_a' in file or 'SNL_18650_NCA_15C_0-100_0.5-1C_b' in file:
            protocols[file] = 92
        elif 'SNL_18650_NCA_15C_0-100_0.5-2C_a' in file or 'SNL_18650_NCA_15C_0-100_0.5-2C_b' in file:
            protocols[file] = 93
        elif 'SNL_18650_NCA_25C_0-100_0.5-0.5C_a' in file or 'SNL_18650_NCA_25C_0-100_0.5-0.5C_b' in file:
            protocols[file] = 94
        elif 'SNL_18650_NCA_25C_0-100_0.5-1C_a' in file or 'SNL_18650_NCA_25C_0-100_0.5-1C_b' in file or 'SNL_18650_NCA_25C_0-100_0.5-1C_c' in file or 'SNL_18650_NCA_25C_0-100_0.5-1C_d' in file:
            protocols[file] = 95
        elif 'SNL_18650_NCA_25C_0-100_0.5-2C_a' in file or 'SNL_18650_NCA_25C_0-100_0.5-2C_b' in file:
            protocols[file] = 96
        elif 'SNL_18650_NCA_25C_20-80_0.5-0.5C_b' in file or 'SNL_18650_NCA_25C_20-80_0.5-0.5C_a' in file or 'SNL_18650_NCA_25C_20-80_0.5-0.5C_c' in file or 'SNL_18650_NCA_25C_20-80_0.5-0.5C_d' in file:
            protocols[file] = 97
        elif 'SNL_18650_NCA_35C_0-100_0.5-1C_a' in file or 'SNL_18650_NCA_35C_0-100_0.5-1C_b' in file or 'SNL_18650_NCA_35C_0-100_0.5-1C_c' in file or 'SNL_18650_NCA_35C_0-100_0.5-1C_d' in file:
            protocols[file] = 98
        elif 'SNL_18650_NCA_35C_0-100_0.5-2C_a' in file or 'SNL_18650_NCA_35C_0-100_0.5-2C_b' in file:
            protocols[file] = 99
        elif 'SNL_18650_NMC_15C_0-100_0.5-1C_a' in file or 'SNL_18650_NMC_15C_0-100_0.5-1C_b' in file:
            protocols[file] = 100
        elif 'SNL_18650_NMC_15C_0-100_0.5-2C_a' in file or 'SNL_18650_NMC_15C_0-100_0.5-2C_b' in file:
            protocols[file] = 101
        elif 'SNL_18650_NMC_25C_0-100_0.5-0.5C_b' in file:
            protocols[file] = 102
        elif 'SNL_18650_NMC_25C_0-100_0.5-1C_a' in file or 'SNL_18650_NMC_25C_0-100_0.5-1C_b' in file or 'SNL_18650_NMC_25C_0-100_0.5-1C_c' in file or 'SNL_18650_NMC_25C_0-100_0.5-1C_d' in file:
            protocols[file] = 103
        elif 'SNL_18650_NMC_25C_0-100_0.5-2C_a' in file or 'SNL_18650_NMC_25C_0-100_0.5-2C_b' in file:
            protocols[file] = 104
        elif 'SNL_18650_NMC_25C_0-100_0.5-3C_a' in file or 'SNL_18650_NMC_25C_0-100_0.5-3C_b' in file or 'SNL_18650_NMC_25C_0-100_0.5-3C_c' in file or 'SNL_18650_NMC_25C_0-100_0.5-3C_d' in file:
            protocols[file] = 105
        elif 'SNL_18650_NMC_35C_0-100_0.5-1C_a' in file or 'SNL_18650_NMC_35C_0-100_0.5-1C_b' in file or 'SNL_18650_NMC_35C_0-100_0.5-1C_c' in file or 'SNL_18650_NMC_35C_0-100_0.5-1C_d' in file:
            protocols[file] =106
        elif 'SNL_18650_NMC_35C_0-100_0.5-2C_a' in file or 'SNL_18650_NMC_35C_0-100_0.5-2C_b' in file:
            protocols[file] = 107
        elif 'SNL_18650_LFP_15C_0-100_0.5-2C_b' in file:
            protocols[file] = 108
        elif 'SNL_18650_LFP_25C_0-100_0.5-0.5C_a' in file:
            protocols[file] = 109
        elif 'SNL_18650_LFP_25C_0-100_0.5-1C_a' in file or 'SNL_18650_LFP_25C_0-100_0.5-1C_b' in file or 'SNL_18650_LFP_25C_0-100_0.5-1C_c' in file or 'SNL_18650_LFP_25C_0-100_0.5-1C_d' in file:
            protocols[file] = 110
        elif 'SNL_18650_LFP_25C_0-100_0.5-2C_a' in file or 'SNL_18650_LFP_25C_0-100_0.5-2C_b' in file:
            protocols[file] = 111
        elif 'SNL_18650_LFP_35C_0-100_0.5-1C_a' in file:
            protocols[file] = 112
    elif 'MICH' in file:
        if 'MICH_01R_pouch_NMC_25C_0-100_0.2-0.2C' in file:
            protocols[file] = 113
        elif 'MICH_02C_pouch_NMC_-5C_0-100_0.2-0.2C' in file:
            protocols[file] = 114
        elif 'MICH_03H_pouch_NMC_45C_0-100_0.2-0.2C' in file:
            protocols[file] = 115
        elif 'MICH_04R_pouch_NMC_25C_0-100_1.5-1.5C' in file:
            protocols[file] = 116
        elif 'MICH_05C_pouch_NMC_-5C_0-100_1.5-1.5C' in file:
            protocols[file] = 117
        elif 'MICH_06H_pouch_NMC_45C_0-100_1.5-1.5C' in file:
            protocols[file] = 118
        elif 'MICH_07R_pouch_NMC_25C_0-100_2-2C' in file:
            protocols[file] = 119
        elif 'MICH_08C_pouch_NMC_-5C_0-100_2-2C' in file:
            protocols[file] = 120
        elif 'MICH_09H_pouch_NMC_45C_0-100_2-2C' in file:
            protocols[file] = 121
        elif 'MICH_10R_pouch_NMC_25C_0-100_0.2-1.5C' in file:
            protocols[file] = 122
        elif 'MICH_11C_pouch_NMC_-5C_0-100_0.2-1.5C' in file:
            protocols[file] = 123
        elif 'MICH_12H_pouch_NMC_45C_0-100_0.2-1.5C' in file:
            protocols[file] = 124
        elif 'MICH_13R_pouch_NMC_25C_50-100_0.2-0.2C' in file:
            protocols[file] = 125
        elif 'MICH_14C_pouch_NMC_-5C_50-100_0.2-0.2C' in file:
            protocols[file] = 126
        elif 'MICH_15H_pouch_NMC_45C_50-100_0.2-0.2C' in file:
            protocols[file] = 127
        elif 'MICH_16R_pouch_NMC_25C_50-100_0.2-1.5C' in file:
            protocols[file] = 128
        elif 'MICH_17C_pouch_NMC_-5C_50-100_0.2-1.5C' in file:
            protocols[file] = 129
        elif 'MICH_18H_pouch_NMC_45C_50-100_0.2-1.5C' in file:
            protocols[file] = 130
        elif 'MICH_BLForm' in file:
            if '45C' in file:
                protocols[file] = 131
            elif '25C' in file:
                protocols[file] = 132
        elif 'MICH_MCForm' in file:
            if '45C' in file:
                protocols[file] = 133
            elif '25C' in file:
                protocols[file] = 134
    elif 'RWTH' in file:
        protocols[file] = 135
    elif 'Tongji' in file:
        if file.startswith('Tongji1_CY25-025_1'):
            protocols[file] = 136
        elif file.startswith('Tongji1_CY25-05_1'):
            protocols[file] = 137
        elif file.startswith('Tongji1_CY25-1_1'):
            protocols[file] = 138
        elif file.startswith('Tongji1_CY35-05_1'):
            protocols[file] = 139
        elif file.startswith('Tongji1_CY45-05_1'):
            protocols[file] = 140
        elif file.startswith('Tongji2_CY25-05_1'):
            protocols[file] = 141
        elif file.startswith('Tongji2_CY35-05_1'):
            protocols[file] = 142
        elif file.startswith('Tongji2_CY45-05_1'):
            protocols[file] = 143
        elif file.startswith('Tongji3_CY25-05_1'):
            protocols[file] = 144
        elif file.startswith('Tongji3_CY25-05_2'):
            protocols[file] = 145
        elif file.startswith('Tongji3_CY25-05_4'):
            protocols[file] = 146
    elif 'Stanford' in file:
        if '100' in file or '101' in file or '102' in file:
            protocols[file] = 147
        elif '190' in file or '191' in file or '192' in file:
            protocols[file] = 148
        elif '193' in file or '194' in file or '195' in file:
            protocols[file] =149
        elif '196' in file or '198' in file:
            protocols[file] = 150
        elif '199' in file or '200' in file or '201' in file:
            protocols[file] = 151
        elif '202' in file or '203' in file or '204' in file:
            protocols[file] = 152
        elif '205' in file or '206' in file or '207' in file:
            protocols[file] = 153
        elif '208' in file or '209' in file or '210' in file:
            protocols[file] = 154
        elif '211' in file or '212' in file or '213' in file:
            protocols[file] = 155
        elif '214' in file or '215' in file or '216' in file:
            protocols[file] = 156
        elif '217' in file or '219' in file:
            protocols[file] = 157
        elif '220' in file or '221' in file or '222' in file:
            protocols[file] = 158
        elif '223' in file or '224' in file or '225' in file:
            protocols[file] = 159
        elif '226' in file or '227' in file or '228' in file:
            protocols[file] =160
        elif '229' in file or '230' in file or '269' in file:
            protocols[file] = 161
        elif '270' in file or '271' in file or '272' in file:
            protocols[file] = 162
        elif '273' in file or '274' in file or '275' in file:
            protocols[file] = 163
        elif '276' in file or '277' in file or '278' in file:
            protocols[file] = 164
        elif '279' in file or '280' in file or '281' in file:
            protocols[file] = 165
        elif '282' in file or '283' in file or '284' in file:
            protocols[file] = 166
        elif '285' in file or '286' in file or '287' in file:
            protocols[file] = 167
        elif '288' in file or '289' in file or '290' in file:
            protocols[file] = 168
        elif '291' in file or '292' in file or '293' in file:
            protocols[file] = 169
        elif '294' in file or '295' in file or '296' in file:
            protocols[file] = 170
        elif '297' in file or '299' in file:
            protocols[file] = 171
        elif '300' in file or '301' in file or '302' in file:
            protocols[file] = 172
        elif '303' in file or '304' in file or '305' in file:
            protocols[file] = 173
        elif '306' in file or '307' in file or '308' in file:
            protocols[file] = 174
        elif '309' in file or '310' in file or '311' in file:
            protocols[file] = 175
        elif '312' in file or '313' in file or '314' in file:
            protocols[file] = 176
        elif '315' in file or '316' in file or '317' in file:
            protocols[file] = 177
        elif '318' in file or '319' in file or '320' in file:
            protocols[file] = 178
        elif '321' in file or '322' in file or '323' in file:
            protocols[file] = 179
        elif '324' in file or '325' in file or '326' in file:
            protocols[file] = 180
        elif '103' in file or '104' in file or '105' in file:
            protocols[file] = 181
        elif '106' in file or '107' in file or '108' in file:
            protocols[file] = 182
        elif '109' in file or '110' in file:
            protocols[file] = 183
        elif '112' in file or '113' in file or '114' in file:
            protocols[file] = 184
        elif '115' in file or '116' in file or '117' in file or '118' in file or '119' in file or '120' in file:
            protocols[file] = 185
        elif '121' in file or '122' in file or '123' in file:
            protocols[file] = 186
        elif '124' in file or '125' in file or '126' in file:
            protocols[file] = 187
        elif '127' in file or '128' in file or '129' in file:
            protocols[file] = 188
        elif '130' in file or '131' in file:
            protocols[file] = 189
        elif '134' in file or '135' in file:
            protocols[file] = 190
        elif '136' in file or '137' in file or '138' in file:
            protocols[file] = 191
        elif '139' in file or '140' in file or '141' in file:
            protocols[file] = 192
        elif '142' in file or '143' in file or '144' in file:
            protocols[file] = 193
        elif '145' in file or '146' in file or '147' in file:
            protocols[file] = 194
        elif '148' in file or '149' in file or '150' in file:
            protocols[file] = 195
        elif '151' in file or '152' in file:
            protocols[file] = 196
        elif '154' in file or '155' in file or '156' in file:
            protocols[file] = 197
        elif '157' in file or '158' in file or '159' in file:
            protocols[file] = 198
        elif '160' in file or '161' in file or '162' in file:
            protocols[file] = 199
        elif '163' in file or '165' in file:
            protocols[file] = 200
        elif '166' in file or '167' in file or '168' in file:
            protocols[file] = 201
        elif '169' in file or '170' in file or '171' in file:
            protocols[file] = 202
        elif '172' in file or '173' in file or '174' in file:
            protocols[file] = 203
        elif '175' in file or '176' in file or '177' in file:
            protocols[file] = 204
        elif '178' in file or '179' in file or '180' in file:
            protocols[file] = 205
        elif '181' in file or '182' in file or '183' in file:
            protocols[file] = 206
        elif '184' in file or '185' in file or '186' in file:
            protocols[file] = 207
        elif '187' in file or '188' in file or '189' in file:
            protocols[file] = 208
    elif 'XJTU' in file:
        if file.startswith('XJTU_2C'):
            protocols[file] = 210
        elif file.startswith('XJTU_3C'):
            protocols[file] = 211
    elif 'ISU-ILCC' in file:
        if 'G1C1' in file:
            protocols[file] = 212
        elif 'G1C2' in file:
            protocols[file] = 213
        elif 'G1C3' in file:
            protocols[file] = 214
        elif 'G1C4' in file:
            protocols[file] = 215
        elif 'G2C1' in file:
            protocols[file] = 216
        elif 'G2C2' in file:
            protocols[file] = 217
        elif 'G2C3' in file:
            protocols[file] = 218
        elif 'G2C4' in file:
            protocols[file] = 219
        elif 'G3C1' in file:
            protocols[file] = 220
        elif 'G3C2' in file:
            protocols[file] = 221
        elif 'G3C3' in file:
            protocols[file] = 222
        elif 'G3C4' in file:
            protocols[file] = 223
        elif 'G4C1' in file:
            protocols[file] = 224
        elif 'G4C2' in file:
            protocols[file] = 225
        elif 'G4C3' in file:
            protocols[file] = 226
        elif 'G4C4' in file:
            protocols[file] = 227
        elif 'G5C' in file:
            protocols[file] = 228
        elif 'G6C1' in file:
            protocols[file] = 229
        elif 'G6C2' in file:
            protocols[file] = 230
        elif 'G6C3' in file:
            protocols[file] = 231
        elif 'G6C4' in file:
            protocols[file] = 232
        elif 'G7C1' in file:
            protocols[file] = 233
        elif 'G7C2' in file:
            protocols[file] = 234
        elif 'G7C3' in file:
            protocols[file] = 235
        elif 'G7C4' in file:
            protocols[file] = 236
        elif 'G8C1' in file:
            protocols[file] = 237
        elif 'G8C2' in file:
            protocols[file] = 238
        elif 'G8C3' in file:
            protocols[file] = 239
        elif 'G8C4' in file:
            protocols[file] = 240
        elif 'G9C' in file:
            protocols[file] = 241
        elif 'G10C1' in file:
            protocols[file] = 242
        elif 'G10C2' in file:
            protocols[file] = 243
        elif 'G10C3' in file:
            protocols[file] = 244
        elif 'G10C4' in file:
            protocols[file] = 245
        elif 'G12C1' in file:
            protocols[file] = 246
        elif 'G12C2' in file:
            protocols[file] = 247
        elif 'G12C3' in file:
            protocols[file] = 248
        elif 'G12C4' in file:
            protocols[file] = 249
        elif 'G13C1' in file:
            protocols[file] = 250
        elif 'G13C2' in file:
            protocols[file] = 251
        elif 'G13C3' in file:
            protocols[file] = 252
        elif 'G13C4' in file:
            protocols[file] = 253
        elif 'G14C1' in file:
            protocols[file] = 254
        elif 'G14C2' in file:
            protocols[file] = 255
        elif 'G14C3' in file:
            protocols[file] = 256
        elif 'G14C4' in file:
            protocols[file] = 257
        elif 'G16C' in file:
            protocols[file] = 258
        elif 'G17C1' in file:
            protocols[file] = 259
        elif 'G17C2' in file:
            protocols[file] = 260
        elif 'G17C3' in file:
            protocols[file] = 261
        elif 'G17C4' in file:
            protocols[file] = 262
        elif 'G18C1' in file:
            protocols[file] = 263
        elif 'G18C2' in file:
            protocols[file] = 264
        elif 'G18C3' in file:
            protocols[file] = 265
        elif 'G18C4' in file:
            protocols[file] = 266
        elif 'G19C1' in file:
            protocols[file] = 267
        elif 'G19C2' in file:
            protocols[file] = 268
        elif 'G19C3' in file:
            protocols[file] = 269
        elif 'G19C4' in file:
            protocols[file] = 270
        elif 'G20C' in file:
            protocols[file] = 271
        elif 'G21C' in file:
            protocols[file] = 272
        elif 'G22C' in file:
            protocols[file] = 273
        elif 'G23C' in file:
            protocols[file] = 274
        elif 'G24C' in file:
            protocols[file] = 275
        elif 'G25C1' in file:
            protocols[file] = 276
        elif 'G25C2' in file:
            protocols[file] = 277
        elif 'G25C3' in file:
            protocols[file] = 278
        elif 'G27C1' in file:
            protocols[file] = 279
        elif 'G27C2' in file:
            protocols[file] = 280
        elif 'G27C3' in file:
            protocols[file] = 281
        elif 'G27C4' in file:
            protocols[file] = 282
        elif 'G28C1' in file:
            protocols[file] = 283
        elif 'G28C2' in file:
            protocols[file] = 284
        elif 'G28C3' in file:
            protocols[file] = 285
        elif 'G28C4' in file:
            protocols[file] = 286
        elif 'G29C1' in file:
            protocols[file] = 287
        elif 'G29C2' in file:
            protocols[file] = 288
        elif 'G29C3' in file:
            protocols[file] = 289
        elif 'G29C4' in file:
            protocols[file] = 290
        elif 'G30C1' in file:
            protocols[file] = 291
        elif 'G30C2' in file:
            protocols[file] = 292
        elif 'G30C3' in file:
            protocols[file] = 293
        elif 'G30C4' in file:
            protocols[file] = 294
        elif 'G31C1' in file:
            protocols[file] = 295
        elif 'G31C2' in file:
            protocols[file] = 296
        elif 'G31C3' in file:
            protocols[file] = 297
        elif 'G31C4' in file:
            protocols[file] = 298
        elif 'G32C1' in file:
            protocols[file] = 299
        elif 'G32C2' in file:
            protocols[file] = 300
        elif 'G32C3' in file:
            protocols[file] = 301
        elif 'G32C4' in file:
            protocols[file] = 302
        elif 'G33C1' in file:
            protocols[file] = 303
        elif 'G33C2' in file:
            protocols[file] = 304
        elif 'G33C3' in file:
            protocols[file] = 305
        elif 'G33C4' in file:
            protocols[file] = 306
        elif 'G34C1' in file:
            protocols[file] = 307
        elif 'G34C2' in file:
            protocols[file] = 308
        elif 'G34C3' in file:
            protocols[file] = 309
        elif 'G34C4' in file:
            protocols[file] = 310
        elif 'G35C' in file:
            protocols[file] = 311
        elif 'G36C' in file:
            protocols[file] = 312
        elif 'G37C' in file:
            protocols[file] = 313
        elif 'G38C' in file:
            protocols[file] = 314
        elif 'G39C' in file:
            protocols[file] = 315
        elif 'G40C1' in file:
            protocols[file] = 316
        elif 'G40C2' in file:
            protocols[file] = 317
        elif 'G40C3' in file:
            protocols[file] = 318
        elif 'G40C4' in file:
            protocols[file] = 319
        elif 'G41C1' in file:
            protocols[file] = 320
        elif 'G41C2' in file:
            protocols[file] = 321
        elif 'G41C3' in file:
            protocols[file] = 322
        elif 'G41C4' in file:
            protocols[file] = 323
        elif 'G42C1' in file:
            protocols[file] = 324
        elif 'G42C2' in file:
            protocols[file] = 325
        elif 'G42C3' in file:
            protocols[file] = 326
        elif 'G42C4' in file:
            protocols[file] = 327
        elif 'G43C1' in file:
            protocols[file] = 328
        elif 'G43C2' in file:
            protocols[file] = 329
        elif 'G43C3' in file:
            protocols[file] = 330
        elif 'G43C4' in file:
            protocols[file] = 331
        elif 'G44C1' in file:
            protocols[file] = 332
        elif 'G44C2' in file:
            protocols[file] = 333
        elif 'G44C3' in file:
            protocols[file] = 334
        elif 'G44C4' in file:
            protocols[file] = 335
        elif 'G45C1' in file:
            protocols[file] = 336
        elif 'G45C2' in file:
            protocols[file] = 337
        elif 'G45C3' in file:
            protocols[file] = 338
        elif 'G45C4' in file:
            protocols[file] = 339
        elif 'G46C1' in file:
            protocols[file] = 340
        elif 'G46C2' in file:
            protocols[file] = 341
        elif 'G46C3' in file:
            protocols[file] = 342
        elif 'G46C4' in file:
            protocols[file] = 343
        elif 'G47C1' in file:
            protocols[file] = 344
        elif 'G47C2' in file:
            protocols[file] = 345
        elif 'G47C3' in file:
            protocols[file] = 346
        elif 'G47C4' in file:
            protocols[file] = 347
        elif 'G48C1' in file:
            protocols[file] = 348
        elif 'G48C2' in file:
            protocols[file] = 349
        elif 'G48C3' in file:
            protocols[file] = 350
        elif 'G48C4' in file:
            protocols[file] = 351
        elif 'G49C1' in file:
            protocols[file] = 352
        elif 'G49C2' in file:
            protocols[file] = 353
        elif 'G49C3' in file:
            protocols[file] = 354
        elif 'G49C4' in file:
            protocols[file] = 355
        elif 'G50C1' in file:
            protocols[file] = 356
        elif 'G50C2' in file:
            protocols[file] = 357
        elif 'G50C3' in file:
            protocols[file] = 358
        elif 'G50C4' in file:
            protocols[file] = 359
        elif 'G51C1' in file:
            protocols[file] = 360
        elif 'G51C2' in file:
            protocols[file] = 361
        elif 'G51C3' in file:
            protocols[file] = 362
        elif 'G51C4' in file:
            protocols[file] = 363
        elif 'G52C1' in file:
            protocols[file] = 364
        elif 'G52C2' in file:
            protocols[file] = 365
        elif 'G52C3' in file:
            protocols[file] = 366
        elif 'G52C4' in file:
            protocols[file] = 367
        elif 'G53C' in file:
            protocols[file] = 368
        elif 'G54C1' in file:
            protocols[file] = 369
        elif 'G54C2' in file:
            protocols[file] = 370
        elif 'G54C3' in file:
            protocols[file] = 371
        elif 'G54C4' in file:
            protocols[file] = 372
        elif 'G55C1' in file:
            protocols[file] = 373
        elif 'G55C2' in file:
            protocols[file] = 374
        elif 'G55C3' in file:
            protocols[file] = 375
        elif 'G55C4' in file:
            protocols[file] = 376
        elif 'G56C1' in file:
            protocols[file] = 377
        elif 'G56C2' in file:
            protocols[file] = 378
        elif 'G56C3' in file:
            protocols[file] = 379
        elif 'G56C4' in file:
            protocols[file] = 380
        elif 'G57C1' in file:
            protocols[file] = 381
        elif 'G57C2' in file:
            protocols[file] = 382
        elif 'G57C3' in file:
            protocols[file] = 383
        elif 'G57C4' in file:
            protocols[file] = 384
        elif 'G58C1' in file:
            protocols[file] = 385
        elif 'G58C2' in file:
            protocols[file] = 386
        elif 'G58C3' in file:
            protocols[file] = 387
        elif 'G58C4' in file:
            protocols[file] = 388
        elif 'G59C1' in file:
            protocols[file] = 389
        elif 'G59C2' in file:
            protocols[file] = 390
        elif 'G59C3' in file:
            protocols[file] = 391
        elif 'G59C4' in file:
            protocols[file] = 392
        elif 'G60C1' in file:
            protocols[file] = 393
        elif 'G60C2' in file:
            protocols[file] = 394
        elif 'G60C3' in file:
            protocols[file] = 395
        elif 'G60C4' in file:
            protocols[file] = 396
        elif 'G61C1' in file:
            protocols[file] = 397
        elif 'G61C2' in file:
            protocols[file] = 398
        elif 'G61C3' in file:
            protocols[file] = 399
        elif 'G61C4' in file:
            protocols[file] = 400
        elif 'G62C1' in file:
            protocols[file] = 401
        elif 'G62C2' in file:
            protocols[file] = 402
        elif 'G62C3' in file:
            protocols[file] = 403
        elif 'G62C4' in file:
            protocols[file] = 404
        elif 'G63C1' in file:
            protocols[file] = 405
        elif 'G63C2' in file:
            protocols[file] = 406
        elif 'G63C3' in file:
            protocols[file] = 407
        elif 'G63C4' in file:
            protocols[file] = 408
        elif 'G64C' in file:
            protocols[file] = 409
    elif 'CALB' in file:
        if 'CALB_0' in file:
            protocols[file] = 535
        elif 'CALB_25' in file:
            protocols[file] = 536
        elif 'CALB_35' in file:
            protocols[file] = 537
        elif 'CALB_45' in file:
            protocols[file] = 538
    elif 'NA-ion' in file:
        if 'NA-ion_270040-1-1-64' in file or 'NA-ion_270040-4-8-41' in file or 'NA-ion_270040-6-5-27' in file or 'NA-ion_270040-8-3-18' in file:
            protocols[file] = 539
        elif 'NA-ion_270040-1-2-63' in file or 'NA-ion_270040-1-5-60' in file or 'NA-ion_270040-5-7-33' in file:
            protocols[file] = 540
        elif 'NA-ion_270040-1-3-62' in file or 'NA-ion_270040-3-7-50' in file:
            protocols[file] = 541
        elif 'NA-ion_270040-1-4-61' in file or 'NA-ion_270040-1-8-57' in file or 'NA-ion_270040-4-3-46' in file or 'NA-ion_270040-4-3-46' in file:
            protocols[file] = 542
        elif 'NA-ion_270040-1-6-59' in file or 'NA-ion_270040-4-4-45' in file or 'NA-ion_270040-5-5-35' in file or 'NA-ion_270040-5-5-35' in file or 'NA-ion_270040-6-8-24' in file:
            protocols[file] = 543
        elif 'NA-ion_270040-1-7-58' in file or 'NA-ion_270040-3-3-54' in file or 'NA-ion_270040-3-4-54' in file:
            protocols[file] = 544
        elif file.startswith('NA-ion_270040-2'):
            protocols[file] = 545
        elif 'NA-ion_270040-3-1-56' in file:
            protocols[file] = 546
        elif 'NA-ion_270040-3-2-55' in file or 'NA-ion_270040-5-1-39' in file or 'NA-ion_270040-5-2-38' in file or 'NA-ion_270040-5-3-37' in file or 'NA-ion_270040-5-6-34':
            protocols[file] = 547
        elif 'NA-ion_270040-3-5-52' in file or 'NA-ion_270040-3-5-52' in file or 'NA-ion_270040-3-8-49' in file or 'NA-ion_270040-4-1-48' in file or 'NA-ion_270040-4-5-44' in file or 'NA-ion_270040-4-7-42' in file or 'NA-ion_270040-6-1-31' in file or 'NA-ion_270040-6-3-29' in file or ' NA-ion_270040-6-4-28' in file or 'NA-ion_270040-6-7-25' in file or 'NA-ion_270040-7-3-21' in file or 'NA-ion_270040-8-1-20' in file or 'NA-ion_270040-8-2-19' in file or 'NA-ion_270040-8-4-17' in file or 'NA-ion_270040-8-6-15' in file or 'NA-ion_270040-8-7-14' in file or 'NA-ion_270040-8-7-14' in file:
            protocols[file] = 548
        elif 'NA-ion_270040-4-2-47' in file:
            protocols[file] = 549
        elif 'NA-ion_270040-4-6-43' in file or 'NA-ion_270040-5-8-32' in file:
            protocols[file] = 550
        elif 'NA-ion_270040-5-4-36' in file:
            protocols[file] = 551
        elif 'NA-ion_270040-6-2-30' in file or 'NA-ion_270040-7-2-22' in file:
            protocols[file] = 552
    elif 'HUST' in file:
        if '1-1' in file:
            protocols[file] = 553
        elif '1-2' in file:
            protocols[file] = 554
        elif '1-3' in file:
            protocols[file] = 555
        elif '1-4' in file:
            protocols[file] = 556
        elif '1-5' in file:
            protocols[file] = 557
        elif '1-6' in file:
            protocols[file] = 558
        elif '1-7' in file:
            protocols[file] = 559
        elif '1-8' in file:
            protocols[file] = 560
        elif '2-2' in file:
            protocols[file] = 561
        elif '2-3' in file:
            protocols[file] = 562
        elif '2-4' in file:
            protocols[file] = 563
        elif '2-5' in file:
            protocols[file] = 564
        elif '2-6' in file:
            protocols[file] = 565
        elif '2-7' in file:
            protocols[file] = 566
        elif '2-8' in file:
            protocols[file] = 567
        elif '3-1' in file:
            protocols[file] = 568
        elif '3-2' in file:
            protocols[file] = 569
        elif '3-3' in file:
            protocols[file] = 570
        elif '3-4' in file:
            protocols[file] = 571
        elif '3-5' in file:
            protocols[file] = 572
        elif '3-6' in file:
            protocols[file] = 573
        elif '3-7' in file:
            protocols[file] = 574
        elif '3-8' in file:
            protocols[file] = 575
        elif '4-1' in file:
            protocols[file] = 576
        elif '4-2' in file:
            protocols[file] = 577
        elif '4-3' in file:
            protocols[file] = 578
        elif '4-4' in file:
            protocols[file] = 579
        elif '4-5' in file:
            protocols[file] = 580
        elif '4-6' in file:
            protocols[file] = 581
        elif '4-7' in file:
            protocols[file] = 582
        elif '4-8' in file:
            protocols[file] = 583
        elif '5-1' in file:
            protocols[file] = 584
        elif '5-2' in file:
            protocols[file] = 585
        elif '5-3' in file:
            protocols[file] = 586
        elif '5-4' in file:
            protocols[file] = 587
        elif '5-5' in file:
            protocols[file] = 588
        elif '5-6' in file:
            protocols[file] = 589
        elif '5-7' in file:
            protocols[file] = 590
        elif '6-1' in file:
            protocols[file] = 591
        elif '6-2' in file:
            protocols[file] = 592
        elif '6-3' in file:
            protocols[file] = 593
        elif '6-4' in file:
            protocols[file] = 594
        elif '6-5' in file:
            protocols[file] = 595
        elif '6-6' in file:
            protocols[file] = 596
        elif '6-8' in file:
            protocols[file] = 597
        elif '7-1' in file:
            protocols[file] = 598
        elif '7-2' in file:
            protocols[file] = 599
        elif '7-3' in file:
            protocols[file] = 600
        elif '7-4' in file:
            protocols[file] = 601
        elif '7-5' in file:
            protocols[file] = 602
        elif '7-6' in file:
            protocols[file] = 603
        elif '7-7' in file:
            protocols[file] = 604
        elif '7-8' in file:
            protocols[file] = 605
        elif '8-1' in file:
            protocols[file] = 606
        elif '8-2' in file:
            protocols[file] = 607
        elif '8-3' in file:
            protocols[file] = 608
        elif '8-4' in file:
            protocols[file] = 609
        elif '8-5' in file:
            protocols[file] = 610
        elif '8-6' in file:
            protocols[file] = 611
        elif '8-7' in file:
            protocols[file] = 612
        elif '8-8' in file:
            protocols[file] = 613
        elif '9-1' in file:
            protocols[file] = 614
        elif '9-2' in file:
            protocols[file] = 615
        elif '9-3' in file:
            protocols[file] = 616
        elif '9-4' in file:
            protocols[file] = 617
        elif '9-5' in file:
            protocols[file] = 618
        elif '9-6' in file:
            protocols[file] = 619
        elif '9-7' in file:
            protocols[file] = 620
        elif '9-8' in file:
            protocols[file] = 621
        elif '10-1' in file:
            protocols[file] = 622
        elif '10-2' in file:
            protocols[file] = 623
        elif '10-3' in file:
            protocols[file] = 624
        elif '10-4' in file:
            protocols[file] = 625
        elif '10-5' in file:
            protocols[file] = 626
        elif '10-6' in file:
            protocols[file] = 627
        elif '10-7' in file:
            protocols[file] = 628
        elif '10-8' in file:
            protocols[file] = 629
    elif 'ZN-coin' in file:
        if file.startswith('ZN-coin_202_20231213213655_03_3'):
            max_value = 629

        protocols[file] = max_value + 1

new_d = relabel_dict_values(protocols)

with open("./name2agingConditionID.json", "w") as json_file:
    json.dump(new_d, json_file, indent=4)
