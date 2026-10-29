"""Offline research tools must inspect metadata without loading model weights."""
import json
import struct
from unittest.mock import patch

import pytest

from scripts.probe_upgrade_readiness import hub_record, local_tensors


def test_local_tensor_count_reads_headers_without_tensor_payload(tmp_path):
    header = json.dumps({"__metadata__": {"format": "pt"},
                         "a": {"shape": [2, 3], "dtype": "F16", "data_offsets": [0, 12]},
                         "b": {"shape": [4], "dtype": "F16", "data_offsets": [12, 20]}}).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(20))
    report = local_tensors(tmp_path)
    assert report["stored_parameters"] == 10
    assert report["files"] == [{"name": path.name, "bytes": path.stat().st_size}]


def test_tensor_header_limit(tmp_path):
    (tmp_path / "bad.safetensors").write_bytes(struct.pack("<Q", 33_000_000))
    with pytest.raises(ValueError, match="Excessive tensor header"):
        local_tensors(tmp_path)


def test_metadata_revision_pinned_without_weights_or_remote_execution():
    responses = [{"sha": "abc123", "cardData": {"license": "other", "license_name": "example"},
                  "safetensors": {"total": 100}, "gated": False},
                 {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}]
    with patch("scripts.probe_upgrade_readiness.fetch_json", side_effect=responses) as fetch:
        result = hub_record({"model_id": "Publisher/Example"})
    assert result["revision"] == "abc123"
    assert result["stored_parameters"] == 100
    assert fetch.call_args_list[1].args[0].endswith("/resolve/abc123/config.json")
    assert len(fetch.call_args_list) == 2


def test_gated_config_does_not_masquerade_as_loader_success():
    with patch("scripts.probe_upgrade_readiness.fetch_json",
               side_effect=[{"sha": "rev", "gated": "manual"}, PermissionError("401 gated")]):
        result = hub_record({"model_id": "Publisher/Gated"})
    assert result["gated"] == "manual"
    assert "401" in result["config_error"]
    assert "installed_loader_mapping" not in result
