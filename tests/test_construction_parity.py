"""Stage-4 acceptance entry point; numerical cases also ship in installed wheels."""
from pathlib import Path
import subprocess
import sys

import pytest

from tradablefactors.tests.construction_test import (
    test_composite_keeps_its_own_daily_anchor_under_outer_quarterly_override,
    test_construction_works_with_rosaa_imports_blocked,
    test_future_prices_cannot_change_past_weights_or_nav,
    test_production_parameters_are_explicit_and_reach_every_basket,
    test_quarterly_nav_matches_independent_lagged_units_and_cash,
    test_shared_gate_opens_all_gated_factors_and_implies_other_signs,
    test_splice_preserves_newer_segment_exactly_and_scales_only_old_history,
    test_thirteen_return_sample_seed_and_single_next_update,
)

__all__ = [
    "test_composite_keeps_its_own_daily_anchor_under_outer_quarterly_override",
    "test_construction_works_with_rosaa_imports_blocked",
    "test_future_prices_cannot_change_past_weights_or_nav",
    "test_production_parameters_are_explicit_and_reach_every_basket",
    "test_quarterly_nav_matches_independent_lagged_units_and_cash",
    "test_shared_gate_opens_all_gated_factors_and_implies_other_signs",
    "test_splice_preserves_newer_segment_exactly_and_scales_only_old_history",
    "test_thirteen_return_sample_seed_and_single_next_update",
]


@pytest.mark.parametrize("case", ["all", "papers"])
def test_consumer_replay_requires_an_explicit_active_research_checkout(case):
    """Installed TF or frozen paper source cannot silently stand in for active consumers."""
    runner = Path(__file__).resolve().parents[1] / "tools/replay_reference.py"
    completed = subprocess.run(
        [sys.executable, str(runner), "--mode", "verify", "--implementation", "consumer",
         "--case", case, "--manifest", "unused.json", "--directory", "unused"],
        check=False, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "requires --consumer-op-root" in completed.stderr
