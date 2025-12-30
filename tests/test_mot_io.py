from pathlib import Path

from mot_editor.mot_io import MotBox, MotStore


def test_mot_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "test.txt"
    store = MotStore()
    store.update_box(1, 1, MotBox(1, 1, 10, 20, 30, 40))
    store.update_box(2, 2, MotBox(2, 2, 15.5, 25.5, 35.5, 45.5))
    store.save(path)

    loaded = MotStore.load(path)
    assert len(loaded.get_frame(1)) == 1
    assert len(loaded.get_frame(2)) == 1
    box = loaded.get_frame(2)[0]
    assert box.track_id == 2
    assert box.left == 15.5
