import pandas as pd
import os
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# Path to parquet files (relative to frontend directory)
DATA_DIR = Path(__file__).parent.parent / "data" / "parquet"

PARQUET_FILES = {
    "block_municipality": "dbo_ml_block_municipality.parquet",
    "bsk_master": "dbo_ml_bsk_master.parquet",
    "citizen_master": "dbo_ml_citizen_master_v2.parquet",
    "deo_master": "dbo_ml_deo_master.parquet",
    "department_master": "dbo_ml_department_master.parquet",
    "district": "dbo_ml_district.parquet",
    "gp_ward_master": "dbo_ml_gp_ward_master.parquet",
    "post_office_master": "dbo_ml_post_office_master.parquet",
    "provision": "dbo_ml_provision.parquet",
    "service_master": "dbo_ml_service_master.parquet",
    "sub_division": "dbo_ml_sub_division.parquet",
}

# Global cache
_DATA_CACHE: Dict[str, pd.DataFrame] = {}
_INITIALIZED = False

@st.cache_data
def load_parquet_file(table_name: str) -> pd.DataFrame:
    """Load a single parquet file with caching."""
    filename = PARQUET_FILES[table_name]
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return pd.DataFrame()
    
    logger.info(f"Loading {filename}...")
    df = pd.read_parquet(filepath)
    logger.info(f"✅ Loaded {len(df):,} rows from {filename}")
    return df

def init_data():
    """Initialize all data files."""
    global _DATA_CACHE, _INITIALIZED
    
    if _INITIALIZED:
        return
    
    logger.info("Loading all parquet files...")
    
    for table_name in PARQUET_FILES.keys():
        _DATA_CACHE[table_name] = load_parquet_file(table_name)
    
    _INITIALIZED = True
    logger.info("✅ All data loaded!")

def get_bsk_master() -> pd.DataFrame:
    return _DATA_CACHE.get("bsk_master", pd.DataFrame()).copy()

def get_service_master() -> pd.DataFrame:
    return _DATA_CACHE.get("service_master", pd.DataFrame()).copy()

def get_deo_master() -> pd.DataFrame:
    return _DATA_CACHE.get("deo_master", pd.DataFrame()).copy()

def get_provisions() -> pd.DataFrame:
    return _DATA_CACHE.get("provision", pd.DataFrame()).copy()

def is_initialized() -> bool:
    return _INITIALIZED
