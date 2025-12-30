"""
Local data loader for frontend-only architecture.
Loads parquet files directly without backend API.
"""

import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# Path to parquet files (adjust this based on your structure)
# Option 1: If parquet files are in project root /data folder
DATA_DIR = Path(__file__).parent.parent / "data" / "parquet"

# Option 2: If parquet files are in frontend/data folder
# DATA_DIR = Path(__file__).parent / "data" / "parquet"

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


@st.cache_data(show_spinner=False)
def load_parquet_file(table_name: str) -> pd.DataFrame:
    """Load a single parquet file with Streamlit caching."""
    if table_name not in PARQUET_FILES:
        logger.error(f"Unknown table: {table_name}")
        return pd.DataFrame()
    
    filename = PARQUET_FILES[table_name]
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        st.error(f"❌ Data file not found: {filename}")
        st.info(f"Expected location: {filepath}")
        return pd.DataFrame()
    
    try:
        logger.info(f"Loading {filename}...")
        df = pd.read_parquet(filepath, engine='pyarrow')
        
        # Basic data cleaning
        df = normalize_dataframe(df, table_name)
        
        logger.info(f"✅ Loaded {len(df):,} rows from {filename}")
        return df
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        st.error(f"❌ Error loading {filename}: {e}")
        return pd.DataFrame()


def normalize_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalize DataFrame columns (types, dates, etc.)."""
    df = df.copy()
    
    # Parse date columns
    date_columns = ["prov_date", "date_of_engagement", "dob"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Ensure numeric ID columns
    id_columns = ["bsk_id", "service_id", "district_id", "agent_id", 
                  "department_id", "block_mun_id", "gp_id", "sub_div_id"]
    for col in id_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Handle boolean columns
    bool_columns = ["is_active", "is_paid_service", "is_aadhar_center"]
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    
    # Fill NaN in string columns with empty string
    string_columns = df.select_dtypes(include=["object"]).columns
    for col in string_columns:
        if col not in date_columns:
            df[col] = df[col].fillna("")
    
    return df


def init_data():
    """Initialize all data files (called once at startup)."""
    global _DATA_CACHE, _INITIALIZED
    
    if _INITIALIZED:
        logger.info("Data already initialized")
        return
    
    logger.info("=" * 60)
    logger.info("Loading BSK Data from Parquet Files")
    logger.info("=" * 60)
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        st.error(f"❌ Data directory not found: {DATA_DIR}")
        st.info("Please ensure parquet files are in the correct location")
        return
    
    success_count = 0
    total_rows = 0
    
    for table_name in PARQUET_FILES.keys():
        df = load_parquet_file(table_name)
        _DATA_CACHE[table_name] = df
        
        if not df.empty:
            success_count += 1
            total_rows += len(df)
    
    _INITIALIZED = True
    
    logger.info("=" * 60)
    logger.info(f"✅ Loaded {success_count}/{len(PARQUET_FILES)} tables")
    logger.info(f"📊 Total rows: {total_rows:,}")
    logger.info("=" * 60)


def get_df(name: str) -> pd.DataFrame:
    """Get a DataFrame by name."""
    if not _INITIALIZED:
        raise RuntimeError("Data not initialized. Call init_data() first.")
    
    if name not in _DATA_CACHE:
        logger.warning(f"DataFrame '{name}' not found in cache")
        return pd.DataFrame()
    
    return _DATA_CACHE[name].copy()


# Convenience getters
def get_provisions() -> pd.DataFrame:
    return get_df("provision")

def get_bsk_master() -> pd.DataFrame:
    return get_df("bsk_master")

def get_service_master() -> pd.DataFrame:
    return get_df("service_master")

def get_deo_master() -> pd.DataFrame:
    return get_df("deo_master")

def get_block_municipality() -> pd.DataFrame:
    return get_df("block_municipality")

def get_citizen_master() -> pd.DataFrame:
    return get_df("citizen_master")

def get_department_master() -> pd.DataFrame:
    return get_df("department_master")

def get_district() -> pd.DataFrame:
    return get_df("district")

def get_sub_division() -> pd.DataFrame:
    return get_df("sub_division")

def is_initialized() -> bool:
    """Check if data has been initialized."""
    return _INITIALIZED


def get_cache_stats() -> Dict:
    """Get statistics about loaded data."""
    if not _INITIALIZED:
        return {"initialized": False}
    
    stats = {
        "initialized": True,
        "tables": {},
        "total_rows": 0,
        "data_dir": str(DATA_DIR),
    }
    
    for name, df in _DATA_CACHE.items():
        stats["tables"][name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        }
        stats["total_rows"] += len(df)
    
    return stats
