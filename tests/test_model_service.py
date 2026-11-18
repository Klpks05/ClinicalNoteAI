import pytest
from pathlib import Path

from backend.app.services.model_service import ModelService, ModelServiceError

@pytest.fixture
def model_service():
    return ModelService()

def test_singleton_pattern():
    s1 = ModelService()
    s2 = ModelService()
    assert s1 is s2

def test_validate_nonexistent_adapter(model_service, tmp_path):
    bad_path = tmp_path / "nonexistent_adapter"
    with pytest.raises(ModelServiceError, match="not found"):
        model_service.validate_adapter(bad_path)

def test_validate_missing_config(model_service, tmp_path):
    adapter_dir = tmp_path / "broken_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"dummy")
    with pytest.raises(ModelServiceError, match="Missing adapter_config.json"):
        model_service.validate_adapter(adapter_dir)

def test_validate_missing_weights(model_service, tmp_path):
    adapter_dir = tmp_path / "broken_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")
    with pytest.raises(ModelServiceError, match="Missing adapter_model.safetensors"):
        model_service.validate_adapter(adapter_dir)

def test_validate_real_mts_adapter(model_service):
    repo_root = Path(__file__).resolve().parent.parent
    mts_adapter = repo_root / "outputs" / "qwen_clinical_lora_mts"
    if not mts_adapter.exists():
        pytest.skip("outputs/qwen_clinical_lora_mts not found")
    assert model_service.validate_adapter(mts_adapter) is True

def test_no_silent_fallback(model_service, tmp_path):
    broken_adapter = tmp_path / "broken"
    broken_adapter.mkdir()
    with pytest.raises(ModelServiceError):
        model_service.load_model(adapter_path=broken_adapter)
