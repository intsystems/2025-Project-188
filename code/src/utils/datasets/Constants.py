from pathlib import Path

# ROOT_FOLDER = Path('/home/khuzin/Projects/2025-Project-188/')
ROOT_FOLDER = Path('/home/khuzin/Projects/2025-m1p-private/')
DATA_FOLDER = ROOT_FOLDER / 'data'
ARTIFACTS_FOLDER = DATA_FOLDER / 'artifacts'

DIR_HYDRO_FILES = DATA_FOLDER / 'HydrologySeries_target/'
DIR_METEO_SERIES_FEATURES = DATA_FOLDER / 'Era5Land_MeteoSeries_feature/'
DIR_HYDRO_STATIC_FEATURES = DATA_FOLDER / 'HydroATLAS_static_feature.csv'

STATIC_FEATURES = [
    "for_pc_sse",
    "crp_pc_sse",
    "inu_pc_ult",
    "ire_pc_sse",
    "lka_pc_use",
    "prm_pc_sse",
    "pst_pc_sse",
    "cly_pc_sav",
    "slt_pc_sav",
    "snd_pc_sav",
    "kar_pc_sse",
    "urb_pc_sse",
    "gwt_cm_sav",
    "lkv_mc_usu",
    "rev_mc_usu",
    "sgr_dk_sav",
    "slp_dg_sav",
    "ws_area",
    "ele_mt_sav",
    "height_bs",
    "lat",
    "lon",
]
