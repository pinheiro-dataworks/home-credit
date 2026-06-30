"""Central configuration — all paths and params loaded from params.yaml."""
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent

DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_FEATURES  = ROOT / "data" / "features"
MODELS_DIR     = ROOT / "models" / "artifacts"
ASSET_DIR      = ROOT / "asset"

with open(ROOT / "params.yaml") as _f:
    PARAMS = yaml.safe_load(_f)

TARGET_COL    = PARAMS["data"]["target_col"]
ID_COL        = PARAMS["data"]["id_col"]
RANDOM_STATE  = PARAMS["data"]["random_state"]
TEST_SIZE     = PARAMS["data"]["test_size"]
NAN_THRESHOLD = PARAMS["feature_engineering"]["nan_threshold"]

MLFLOW_URI        = PARAMS["mlflow"]["tracking_uri"]
EXPERIMENT_NAME   = PARAMS["mlflow"]["experiment_name"]
REGISTERED_MODEL  = PARAMS["mlflow"]["registered_model_name"]

BOOTSTRAP_ITERS = PARAMS["statistics"]["bootstrap_n_iterations"]
BOOTSTRAP_CI    = PARAMS["statistics"]["bootstrap_ci"]
ALPHA           = PARAMS["statistics"]["alpha"]
