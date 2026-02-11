
import re
from typing import Final


# EEG bands (Hz)
ALPHA_BAND: Final[tuple[float, float]] = (8.0, 12.0)
BETA_BAND: Final[tuple[float, float]] = (13.0, 30.0)

# Filename parsing
# Supports both:
#   sub-004_conversation_run-3_epo.fif
#   sub-005_task-conversation_run-1_epochs-epo.fif
EPOCH_FILE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"sub-(\d{3}).*run-(\d+).*epo",
    flags=re.IGNORECASE,
)

# Output filtering
DROP_COLUMN_SUBSTRINGS: Final[tuple[str, ...]] = (
    "_n_",    # e.g., self_n_words
    "_rate",  # e.g., self_rate
)

DROP_COLUMN_EXACT: Final[set[str]] = {
    "index",
    "index.1",
}

DROP_COLUMN_PREFIXES: Final[tuple[str, ...]] = (
    "Unnamed:",  # pandas CSV index artifacts
)
