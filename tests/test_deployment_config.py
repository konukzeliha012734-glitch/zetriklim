from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_cloud_disables_runtime_file_watcher() -> None:
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))

    assert config["server"]["fileWatcherType"] == "none"
    assert config["server"]["runOnSave"] is False
