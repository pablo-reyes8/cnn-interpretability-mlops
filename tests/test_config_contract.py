from pathlib import Path

import yaml

from src.utils.config import get_config


ROOT = Path(__file__).resolve().parents[1]


def test_project_yaml_configs_are_centralized_and_parseable():
    expected = [
        ROOT / "config" / "api.yaml",
        ROOT / "config" / "params.yaml",
        ROOT / "config" / "mlops.yaml",
        ROOT / "config" / "data_governance.yaml",
        ROOT / "config" / "model" / "oxford_pets_binary_resnet101.yaml",
        ROOT / "dvc.yaml",
        ROOT / "docker-compose.yml",
        ROOT / "docker-compose.airflow.yml",
    ]
    for path in expected:
        assert path.exists(), f"Missing YAML config: {path}"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict), f"Config must parse as a mapping: {path}"


def test_legacy_loose_yaml_configs_were_moved_to_config_folder():
    assert not (ROOT / "params.yaml").exists()
    assert not (ROOT / "resnet101" / "oxford_pets_binary_resnet101.yaml").exists()
    assert not (ROOT / "data" / "data_governance.yaml").exists()


def test_api_config_points_to_existing_model_yaml_and_stats_contract():
    cfg = get_config()
    assert cfg.MODEL_YAML_PATH == ROOT / "config" / "model" / "oxford_pets_binary_resnet101.yaml"
    assert cfg.MODEL_YAML_PATH.exists()
    assert cfg.PET_STATS_PATH == ROOT / "data" / "pet_stats.json"
    assert cfg.PET_STATS_PATH.exists()
    assert ".png" in cfg.ALLOWED_EXTS
    assert "image/png" in cfg.ALLOWED_MIMES


def test_dvc_references_config_params_file():
    dvc = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    assert "config/params.yaml" in dvc.get("vars", [])

    params = yaml.safe_load((ROOT / "config" / "params.yaml").read_text(encoding="utf-8"))
    assert params["train"]["config_path"] == "config/model/oxford_pets_binary_resnet101.yaml"
    assert params["dataops"]["governance_path"] == "config/data_governance.yaml"
