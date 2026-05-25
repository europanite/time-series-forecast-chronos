from __future__ import annotations

import local_ts_forecast


def test_package_has_version() -> None:
    assert isinstance(local_ts_forecast.__version__, str)
    assert local_ts_forecast.__version__
