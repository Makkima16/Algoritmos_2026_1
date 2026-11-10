import os

INT_ZERO: int = int(0)
INT_ONE: int = int(1)

FLOAT_ZERO: float = float(INT_ZERO)

BASE_TWO: int = INT_ONE + INT_ONE

COLS_IDX = INT_ONE

STR_ZERO: str = str(INT_ZERO)

WHITESPACE: str = " "
COLON_DELIM: str = ","
VOID_STR: str = "∅"
ABC_START: str = "A"

_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

NET_LABEL: str = "NET"
PATH_LOGS: str = os.path.join(_base_dir, ".logs")
PATH_SAMPLES: str = os.path.join(_base_dir, "src", ".samples")
PATH_PROFILING: str = os.path.join(_base_dir, "review", "profiling")
PATH_RESOLVER: str = os.path.join(_base_dir, "review", "resolver")

CSV_EXTENSION: str = "csv"
HTML_EXTENSION: str = "html"

TYPE_TAG = "type"
