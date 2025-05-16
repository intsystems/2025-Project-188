from pathlib import Path

ARTIFACTS = Path("artifacts")

STATIC_FEATURES = [
    "for_pc_sse",  # Forest Cover Extent
    "crp_pc_sse",  # Cropland Extent
    "pst_pc_sse",  # Pasture Extent
    "ire_pc_sse",  # Irrigated Area Extent
    "urb_pc_sse",  # Urban Extent
    "kar_pc_sse",  # Karst Area Extent
    "ele_mt_sav",  # Elevation
    "slp_dg_sav",  # Terrain Slope
    "ari_ix_sav",  # Aridity Index
    "gwt_cm_sav",  # Groundwater Table Depth
    "swc_pc_syr",  # Soil Water Content (annual)
    "cly_pc_sav",  # Clay Fraction in Soil
    "hft_ix_s09",  # Human Footprint Index
    "lat",  # Latitude
    "lon",  # Longitude
]
