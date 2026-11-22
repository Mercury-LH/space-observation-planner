import pytest

from obsplan.data import norm_satnum, tle_checksum_ok


@pytest.mark.parametrize("raw, num", [("08827", 8827), ("50574", 50574), ("A0045", 100045),
                                      ("T0288", 270288), ("Z9999", 339999), (" 1234", 1234)])
def test_norm_satnum(raw, num):
    assert norm_satnum(raw) == num


def test_tle_checksum():
    line = "1 50574U 21135A   26215.54610063 -.00000290  00000-0  00000-0 0  9996"
    assert tle_checksum_ok(line)
    assert not tle_checksum_ok(line[:-1] + "5")


def test_dataset_consistency(ds):
    assert ds.n_targets == 10000 and ds.n_stations == 10
    assert ds.issues == []
    counts = {k: ds.priority_label.count(k) for k in "高中低"}
    assert counts == {"高": 2024, "中": 3990, "低": 3986}
    assert (ds.satnum >= 100000).sum() == 108          # Alpha-5 编号
    assert len(set(ds.satnum.tolist())) == 10000
    assert all(-180 <= x < 180 for x in ds.st_lon)
