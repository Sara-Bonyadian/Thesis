"""Duration-specific lag grids and endpoint contracts for confirmatory analysis.

These contracts resolve the incompatibility between nested analysis durations and
fixed ±60-s lag / ZLPI flank support. Endpoint *calculations* (Fisher-z, flank
means, etc.) are intentionally not implemented here — only the frozen names and
windows that M5/M6 must honor.

===============================================================================
Duration analysis contracts
===============================================================================

D240 (primary)
  lag grid           : −60 … +60 s, step 1 s (121 lags)
  endpoint_name      : ``zlpi``
  endpoint_alias     : ``ZLPI``  (display acronym; not used for joins)
  distant flanks     : |τ| ∈ [20, 60] s
  local shoulders    : |τ| ∈ [5, 15] s
  standard ZLPI?     : yes
  pool with ZLPI?    : yes  (D240/D180 only)

D180 (mandatory nested sensitivity)
  lag grid           : −60 … +60 s, step 1 s (121 lags)
  endpoint_name      : ``zlpi``
  endpoint_alias     : ``ZLPI``
  distant flanks     : |τ| ∈ [20, 60] s
  local shoulders    : |τ| ∈ [5, 15] s
  standard ZLPI?     : yes
  pool with ZLPI?    : yes  (D240/D180 only)

D120 (nested sensitivity; not standard ZLPI)
  lag grid           : −30 … +30 s, step 1 s (61 lags)
  endpoint_name      : ``mid_window_proximal_index``
  endpoint_alias     : ``MWPI``
  distant flanks     : |τ| ∈ [20, 30] s
  local shoulders    : |τ| ∈ [5, 15] s
  standard ZLPI?     : no
  pool with ZLPI?    : no  (never pool with D240/D180 ZLPI)

D60 (short-window sensitivity; not standard ZLPI)
  lag grid           : −20 … +20 s, step 1 s (41 lags)
  endpoint_name      : ``short_window_proximal_index``
  endpoint_alias     : ``SWPI``
  distant flanks     : |τ| ∈ [10, 20] s
  local shoulders    : |τ| ∈ [5, 15] s
  standard ZLPI?     : no
  pool with ZLPI?    : no  (never pool with D240/D180 ZLPI)

Common-support lag correlations use anchors in [L, N−L) so ``n_overlap`` is
constant across lags. Fully finite segments therefore yield:

  D240 → n_overlap = 120
  D180 → n_overlap = 60
  D120 → n_overlap = 60
  D60  → n_overlap = 20
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

EXPECTED_DURATIONS_S = (240, 180, 120, 60)
EXPECTED_PRIMARY_DURATION_S = 240
EXPECTED_LAG_STEP_S = 1
EXPECTED_SHOULDERS_S = (5, 15)
EXPECTED_PEAK_CENTER_EQUIVALENCE_S = 2

# Standard ZLPI (D240/D180 only).
ENDPOINT_ZLPI = "zlpi"
ZLPI_FLANKS_S = (20, 60)
STANDARD_ZLPI_DURATIONS_S = (240, 180)
STANDARD_ZLPI_LAG_MAX_S = 60

# D120 sensitivity-only alternative (never labeled ZLPI; never pooled with ZLPI).
ENDPOINT_MID_WINDOW_PROXIMAL_INDEX = "mid_window_proximal_index"
MWPI_ALIAS = "MWPI"
MWPI_FLANKS_S = (20, 30)
MWPI_LAG_MAX_S = 30
MID_WINDOW_DURATION_S = 120

# D60 short-window sensitivity-only alternative (never labeled ZLPI; never pooled).
ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX = "short_window_proximal_index"
SWPI_ALIAS = "SWPI"
SWPI_FLANKS_S = (10, 20)
SWPI_LAG_MAX_S = 20
SHORT_WINDOW_DURATION_S = 60

ANALYSIS_ROLE_STANDARD_ZLPI = "standard_zlpi"
ANALYSIS_ROLE_MID_WINDOW_SENSITIVITY = "mid_window_sensitivity"
ANALYSIS_ROLE_SHORT_WINDOW_SENSITIVITY = "short_window_sensitivity"


@dataclass(frozen=True)
class DurationAnalysisContract:
    """Frozen lag-grid + named-endpoint contract for one confirmatory duration.

    ``endpoint_name`` is the machine-stable identifier used for filtering and
    joins (``zlpi``, ``mid_window_proximal_index``, ``short_window_proximal_index``).

    ``endpoint_alias`` is the short display acronym only (``ZLPI``, ``MWPI``,
    ``SWPI``). It is not interchangeable with ``endpoint_name``.
    """

    duration_s: int
    lag_min_s: int
    lag_max_s: int
    lag_step_s: int
    n_lags: int
    endpoint_name: str
    endpoint_alias: str
    flank_inner_s: int
    flank_outer_s: int
    shoulders_inner_s: int
    shoulders_outer_s: int
    is_standard_zlpi: bool
    pool_with_standard_zlpi: bool
    lag_analysis_role: str

    @property
    def flanks_s(self) -> tuple[int, int]:
        return (self.flank_inner_s, self.flank_outer_s)

    @property
    def shoulders_s(self) -> tuple[int, int]:
        return (self.shoulders_inner_s, self.shoulders_outer_s)

    @property
    def expected_constant_overlap_if_fully_finite(self) -> int:
        """Common-support overlap for a fully finite segment of length duration_s."""
        return max(0, int(self.duration_s) - 2 * int(self.lag_max_s))


def _contract(
    *,
    duration_s: int,
    lag_max_s: int,
    endpoint_name: str,
    endpoint_alias: str,
    flanks_s: tuple[int, int],
    is_standard_zlpi: bool,
    pool_with_standard_zlpi: bool,
    lag_analysis_role: str,
) -> DurationAnalysisContract:
    if lag_max_s <= 0:
        raise ValueError("lag_max_s must be > 0.")
    if flanks_s[0] > flanks_s[1]:
        raise ValueError("flanks_s must be ordered (inner, outer).")
    if flanks_s[1] > lag_max_s:
        raise ValueError(
            f"Outer flank {flanks_s[1]} exceeds lag_max_s={lag_max_s} "
            f"for duration_s={duration_s}."
        )
    if EXPECTED_SHOULDERS_S[1] > lag_max_s:
        raise ValueError(
            f"Shoulder outer bound exceeds lag_max_s for duration_s={duration_s}."
        )
    if is_standard_zlpi != pool_with_standard_zlpi:
        raise ValueError(
            "is_standard_zlpi and pool_with_standard_zlpi must match: only "
            "standard ZLPI endpoints may enter the primary ZLPI pool."
        )
    if is_standard_zlpi and endpoint_name != ENDPOINT_ZLPI:
        raise ValueError("Standard ZLPI contracts must use endpoint_name='zlpi'.")
    if (not is_standard_zlpi) and endpoint_name == ENDPOINT_ZLPI:
        raise ValueError("Non-standard contracts must not use endpoint_name='zlpi'.")

    return DurationAnalysisContract(
        duration_s=duration_s,
        lag_min_s=-lag_max_s,
        lag_max_s=lag_max_s,
        lag_step_s=EXPECTED_LAG_STEP_S,
        n_lags=lag_max_s * 2 + 1,
        endpoint_name=endpoint_name,
        endpoint_alias=endpoint_alias,
        flank_inner_s=flanks_s[0],
        flank_outer_s=flanks_s[1],
        shoulders_inner_s=EXPECTED_SHOULDERS_S[0],
        shoulders_outer_s=EXPECTED_SHOULDERS_S[1],
        is_standard_zlpi=is_standard_zlpi,
        pool_with_standard_zlpi=pool_with_standard_zlpi,
        lag_analysis_role=lag_analysis_role,
    )


DURATION_ANALYSIS_CONTRACTS: dict[int, DurationAnalysisContract] = {
    240: _contract(
        duration_s=240,
        lag_max_s=STANDARD_ZLPI_LAG_MAX_S,
        endpoint_name=ENDPOINT_ZLPI,
        endpoint_alias="ZLPI",
        flanks_s=ZLPI_FLANKS_S,
        is_standard_zlpi=True,
        pool_with_standard_zlpi=True,
        lag_analysis_role=ANALYSIS_ROLE_STANDARD_ZLPI,
    ),
    180: _contract(
        duration_s=180,
        lag_max_s=STANDARD_ZLPI_LAG_MAX_S,
        endpoint_name=ENDPOINT_ZLPI,
        endpoint_alias="ZLPI",
        flanks_s=ZLPI_FLANKS_S,
        is_standard_zlpi=True,
        pool_with_standard_zlpi=True,
        lag_analysis_role=ANALYSIS_ROLE_STANDARD_ZLPI,
    ),
    120: _contract(
        duration_s=MID_WINDOW_DURATION_S,
        lag_max_s=MWPI_LAG_MAX_S,
        endpoint_name=ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
        endpoint_alias=MWPI_ALIAS,
        flanks_s=MWPI_FLANKS_S,
        is_standard_zlpi=False,
        pool_with_standard_zlpi=False,
        lag_analysis_role=ANALYSIS_ROLE_MID_WINDOW_SENSITIVITY,
    ),
    60: _contract(
        duration_s=SHORT_WINDOW_DURATION_S,
        lag_max_s=SWPI_LAG_MAX_S,
        endpoint_name=ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
        endpoint_alias=SWPI_ALIAS,
        flanks_s=SWPI_FLANKS_S,
        is_standard_zlpi=False,
        pool_with_standard_zlpi=False,
        lag_analysis_role=ANALYSIS_ROLE_SHORT_WINDOW_SENSITIVITY,
    ),
}


def contract_for_duration(duration_s: int) -> DurationAnalysisContract:
    try:
        return DURATION_ANALYSIS_CONTRACTS[int(duration_s)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported duration_s={duration_s}; expected one of "
            f"{EXPECTED_DURATIONS_S}."
        ) from exc


def standard_zlpi_pool_durations() -> tuple[int, ...]:
    """Durations whose endpoint may enter the primary ZLPI analysis pool."""
    return tuple(
        duration
        for duration, contract in DURATION_ANALYSIS_CONTRACTS.items()
        if contract.pool_with_standard_zlpi
    )


def assert_contracts_internally_consistent() -> None:
    """Fail fast if frozen contracts violate support or naming rules."""
    if tuple(DURATION_ANALYSIS_CONTRACTS) != EXPECTED_DURATIONS_S:
        raise RuntimeError("DURATION_ANALYSIS_CONTRACTS keys must match durations.")
    if standard_zlpi_pool_durations() != STANDARD_ZLPI_DURATIONS_S:
        raise RuntimeError("Standard ZLPI pool must be exactly D240 and D180.")
    for duration, contract in DURATION_ANALYSIS_CONTRACTS.items():
        if contract.duration_s != duration:
            raise RuntimeError(f"Mismatched contract duration key {duration}.")
        if contract.expected_constant_overlap_if_fully_finite <= 0:
            raise RuntimeError(
                f"Duration D{duration} lag_max={contract.lag_max_s} leaves no "
                "common-support overlap."
            )


assert_contracts_internally_consistent()


__all__ = [
    "ANALYSIS_ROLE_MID_WINDOW_SENSITIVITY",
    "ANALYSIS_ROLE_SHORT_WINDOW_SENSITIVITY",
    "ANALYSIS_ROLE_STANDARD_ZLPI",
    "DURATION_ANALYSIS_CONTRACTS",
    "ENDPOINT_MID_WINDOW_PROXIMAL_INDEX",
    "ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX",
    "ENDPOINT_ZLPI",
    "EXPECTED_DURATIONS_S",
    "EXPECTED_LAG_STEP_S",
    "EXPECTED_PEAK_CENTER_EQUIVALENCE_S",
    "EXPECTED_PRIMARY_DURATION_S",
    "EXPECTED_SHOULDERS_S",
    "MID_WINDOW_DURATION_S",
    "MWPI_ALIAS",
    "MWPI_FLANKS_S",
    "MWPI_LAG_MAX_S",
    "SHORT_WINDOW_DURATION_S",
    "STANDARD_ZLPI_DURATIONS_S",
    "STANDARD_ZLPI_LAG_MAX_S",
    "SWPI_ALIAS",
    "SWPI_FLANKS_S",
    "SWPI_LAG_MAX_S",
    "ZLPI_FLANKS_S",
    "DurationAnalysisContract",
    "assert_contracts_internally_consistent",
    "contract_for_duration",
    "standard_zlpi_pool_durations",
]
