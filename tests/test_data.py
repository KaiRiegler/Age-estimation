from age_estimation.data import load_file_names, split_data


def test_labels_come_from_file_names(tmp_path):
    for name in ["68_0_0_2017.jpg.chip.jpg", "5_1_2_2016.jpg.chip.jpg", "broken.jpg"]:
        (tmp_path / name).write_bytes(b"")
    files, labels = load_file_names(tmp_path)
    assert sorted(labels) == [5, 68]
    assert len(files) == 2


def test_split_is_complete_and_reproducible():
    files = [f"{i}.jpg" for i in range(100)]
    labels = list(range(100))
    a = split_data(files, labels, seed=1)
    b = split_data(files, labels, seed=1)
    assert a == b
    assert [len(a[k]) for k in ("train", "val", "test")] == [80, 10, 10]
    assert len({f for part in a.values() for f, _ in part}) == 100
