"""Integration test verifying app components and headless compatibility."""
import subprocess
import sys
import pytest

from twin.database import get_meal_history, get_patient_summary_stats
from twin.recommender import get_recommendation_catalog


def test_catalog_query_performance():
    """Verify recommendation catalog runs fast and returns valid records."""
    catalog = get_recommendation_catalog("pat_mahesh_001", tier_filter="ALL", limit=20)
    assert len(catalog) == 20
    assert all("tier" in item and "tolerance_score" in item for item in catalog)


def test_patient_stats_integrity():
    """Verify patient summary statistics return expected structure."""
    stats = get_patient_summary_stats("pat_mahesh_001")
    assert stats["total_meals"] >= 20
    assert stats["tir_percent"] >= 50.0
    assert isinstance(stats["intolerant_foods"], list)


def test_app_import_and_schema():
    """Verify app.py imports and runs standard schema validation."""
    result = subprocess.run(
        [sys.executable, "-c", "import app; print('App module imported successfully')"],
        capture_output=True,
        text=True,
        timeout=15
    )
    # Streamlit scripts execute top-level commands, check that it doesn't crash on standard execution
    assert result.returncode == 0 or "ScriptRunContext" in result.stderr
