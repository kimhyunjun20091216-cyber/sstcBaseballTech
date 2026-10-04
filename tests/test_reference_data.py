from src.reference_data import resolve_reference_csv_path


def test_resolve_reference_csv_path_prefers_workspace_rec_data():
    path = resolve_reference_csv_path()

    assert path is not None
    assert path.endswith("reanalyze_keypoints.csv")
