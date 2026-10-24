import os

os.environ.setdefault("FWA_NO_APP", "1")  # don't connect at import time; tests create apps explicitly
