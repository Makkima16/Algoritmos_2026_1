INFTY_POS: float = float("inf")
INFTY_NEG: float = float("-inf")

INT_ZERO: int = int(0)
INT_ONE: int = int(1)

FLOAT_ONE: float = float(INT_ONE)
FLOAT_ZERO: float = float(INT_ZERO)

BASE_TWO: int = INT_ONE + INT_ONE

ABC_LEN: int = 26
LAST_IDX = -INT_ONE
ROWS_IDX = ACTUAL = INT_ZERO
COLS_IDX = EFFECT = INT_ONE

STR_ZERO: str = str(INT_ZERO)
STR_ONE: str = str(INT_ONE)

EMPTY_STR: str = ""
WHITESPACE: str = " "
COLON_DELIM: str = ","
VOID_STR: str = "∅"
SMALL_PHI_STR: str = "φ"
ABC_START: str = "A"

EQUIV_SYM: str = "≡"
EQUAL_SYM: str = "="
DASH_SYM: str = "—"
MINUS_SYM: str = "–"
LINE_SYM: str = "-"

EQUITIES = "≌", "≆", "≇", "≄", "≒"
NEQ_SYM: str = "≠"

BITS: tuple[int, int] = (INT_ZERO, INT_ONE)
ACTIVE, INACTIVE = True, False

import os

_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

NET_LABEL: str = "NET"
PATH_LOGS: str = os.path.join(_base_dir, ".logs")
PATH_SAMPLES: str = os.path.join(_base_dir, "src", ".samples")
PATH_PROFILING: str = os.path.join(_base_dir, "review", "profiling")
PATH_RESOLVER: str = os.path.join(_base_dir, "review", "resolver")

CSV_EXTENSION: str = "csv"
HTML_EXTENSION: str = "html"
EXCEL_EXTENSION: str = "xlsx"

TYPE_TAG = "type"
