# -*- coding: utf-8 -*-
import os
import sys
import tempfile
from pathlib import Path

from videotrans.configure.contants import no_proxy

IS_FROZEN = True if getattr(sys, 'frozen', False) else False
SYS_TMP = Path(tempfile.gettempdir()).as_posix()
ROOT_DIR = Path(sys.executable).parent.as_posix() if IS_FROZEN else Path(__file__).parent.parent.parent.as_posix()
TEMP_ROOT = f'{ROOT_DIR}/tmp'
LOGS_DIR = f'{ROOT_DIR}/logs'
TEMP_DIR = f'{TEMP_ROOT}/None'
TRANSLATE_CACHE = f'{TEMP_ROOT}/translate_cache'
DUBBING_CACHE = f'{TEMP_ROOT}/dubbing_cache'

# Queue data file and stop signal file for re-dubbing in single video mode
REDUBB_QUEUE_FILE=f'{TEMP_ROOT}/redubbing.json'
REDUBB_STATUS_FILE=f'{TEMP_ROOT}/stopredubbing.pid'

Path(f"{ROOT_DIR}/models").mkdir(parents=True, exist_ok=True)
Path(f"{ROOT_DIR}/logs").mkdir(parents=True, exist_ok=True)
Path(f"{TRANSLATE_CACHE}").mkdir(parents=True, exist_ok=True)
Path(f"{DUBBING_CACHE}").mkdir(parents=True, exist_ok=True)

def fix_ssl_cert_env():
    """
    Fix incorrect global SSL certificate environment variables on some user machines,
    forcing them to point to the certifi certificate path bundled with the application.
    """
    try:
        import certifi
        ca_bundle = certifi.where()
        
        # Ensure the path exists (compatible with PyInstaller unpacked temporary directory)
        if os.path.exists(ca_bundle):
            # Force overwrite erroneous user environment variables to point to the correct bundle
            os.environ['CURL_CA_BUNDLE'] = ca_bundle
            os.environ['REQUESTS_CA_BUNDLE'] = ca_bundle
            os.environ['SSL_CERT_FILE'] = ca_bundle
        else:
            raise FileNotFoundError
            
    except Exception:
        # If certifi fails to load, remove interfering environment variables so system falls back to default logic
        for key in ['CURL_CA_BUNDLE', 'REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE']:
            os.environ.pop(key, None)



def _set_env():
    try:
        from videotrans.configure._env_loader import load_env
        load_env()
    except Exception:
        pass
    if IS_FROZEN:
        os.environ['TQDM_DISABLE'] = '1'
    os.environ['no_proxy'] = no_proxy
    os.environ['NO_PROXY'] = no_proxy
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CT2_VERBOSE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PYTHONWARNINGS"]="ignore"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ['PYTHONUTF8'] = '1'
    os.environ['SOFT_NAME'] = 'pyvideotrans'
    os.environ['MODELSCOPE_CACHE'] = ROOT_DIR + "/models"
    os.environ['HF_HOME'] = ROOT_DIR + "/models"
    os.environ['PYANNOTE_CACHE'] = ROOT_DIR + "/models"
    os.environ['HF_HUB_CACHE'] = ROOT_DIR + "/models"
    os.environ['HF_TOKEN_PATH'] = ROOT_DIR + "/models/hf_token.txt"
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = "3600"
    os.environ['HF_HUB_ETAG_TIMEOUT'] = "30"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'
    # Must be executed before importing requests, modelscope, or other network libraries!
    fix_ssl_cert_env()
    if Path(f'{ROOT_DIR}/netoffline.txt').is_file():
        os.environ['HF_HUB_OFFLINE'] = '1'

    if sys.platform == 'win32':
        if IS_FROZEN:
            os.environ['PATH'] = f'{ROOT_DIR}/_internal/torch/lib;' + os.environ.get("PATH", "")
        os.environ['PATH'] = ROOT_DIR + os.pathsep + f'{ROOT_DIR}/ffmpeg' + os.pathsep + f'{ROOT_DIR}/ffmpeg/sox' + os.pathsep + os.environ.get(
            "PATH", "")
