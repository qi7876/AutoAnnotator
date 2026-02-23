from scripts.summary_stats import SUMMARY_TASKS


def test_summary_tasks_include_spatial_imagination_and_exclude_object_tracking() -> None:
    assert "Spatial_Imagination" in SUMMARY_TASKS
    assert "Object_Tracking" not in SUMMARY_TASKS
