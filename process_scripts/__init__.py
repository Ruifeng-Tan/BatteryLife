import logging
from .download import DOWNLOAD_LINKS, download_file
from .preprocess_CALCE import CALCEPreprocessor
from .preprocess_HNEI import HNEIPreprocessor
from .preprocess_HUST import HUSTPreprocessor
from .preprocess_MATR import MATRPreprocessor
from .preprocess_RWTH import RWTHPreprocessor
from .preprocess_SNL import SNLPreprocessor
from .preprocess_UL_PUR import UL_PURPreprocessor
from .preprocess_arbin import ARBINPreprocessor
from .preprocess_neware import NEWAREPreprocessor
from .preprocess_MICH import MICHPreprocessor
from .preprocess_MICH_EXP import MICH_EXPPreprocessor
from .preprocess_ISU_ILCC import ISU_ILCCPreprocessor
from .preprocess_Stanford import StanfordPreprocessor
from .preprocess_XJTU import XJTUPreprocessor
from .preprocess_Tongji import TongjiPreprocessor
from .preprocess_ZNion import ZNionPreprocessor
from .preprocess_CALB import CALBPreprocessor
from .preprocess_NA import NAPreprocessor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

SUPPORTED_SOURCES = {
    'DATASETS': ['CALCE', 'HNEI', 'HUST', 'MATR', 'RWTH', 'SNL', 'UL_PUR', 'MICH', 'MICH_EXP', 'ISU_ILCC', 'Stanford', 'ZNion', 'XJTU', 'Tongji', 'NA', 'CALB'],
    'CYCLERS': ['ARBIN', 'BATTERYARCHIVE', "BIOLOGIC",  'INDIGO', "MACCOR", 'NEWARE', 'NOVONIX', 'BATTERYARCHIVE', 'BATTERYARCHIVE', 'NEWARE', 'Unknown', 'ME', 'LISHEN', 'SAMSUNG', 'NEWARE', 'CALB']
}
