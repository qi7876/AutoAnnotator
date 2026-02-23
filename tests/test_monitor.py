from scripts.monitor import TASKS, build_row_values, build_write_range


def test_tasks_include_spatial_imagination_and_exclude_object_tracking() -> None:
    assert "Spatial_Imagination" in TASKS
    assert "Object_Tracking" not in TASKS


def test_build_row_values_matches_task_count() -> None:
    counts = {task: idx for idx, task in enumerate(TASKS, start=1)}
    values = build_row_values(counts)
    assert len(values) == len(TASKS)
    assert values[0] == 1
    assert values[-1] == len(TASKS)


def test_build_write_range_uses_dynamic_columns() -> None:
    # source=1 -> row 2; len(TASKS)==12 means B..M
    assert build_write_range("sheet1", 1) == "sheet1!B2:M2"
