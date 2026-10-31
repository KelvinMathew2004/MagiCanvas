"""A model package must be selected consistently, without changing the runtime."""
import json
import os
from unittest.mock import patch

import pytest

from scripts.check_model_profiles import inspect_profile


@pytest.fixture
def package(tmp_path):
    model = tmp_path / 'base'
    adapter = tmp_path / 'adapter'
    model.mkdir(); adapter.mkdir()
    (model / 'config.json').write_text(json.dumps(dict(model_type='qwen3', hidden_size=1024)))
    (adapter / 'adapter_config.json').write_text(json.dumps(dict(base_model_name_or_path='original-base')))
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        (model / name).write_text('{}')
    return tmp_path, dict(model_dir='base', adapter_dir='adapter', model_type='qwen3',
                          hidden_size=1024, training_base_alias='original-base', prompt_version=2)


def test_package_reports_environment_without_modifying_it(package):
    root, profile = package
    before = dict(os.environ)
    with patch('scripts.verify_qwen_adapter.verify', return_value=dict(prompt_version=2)):
        result = inspect_profile(profile, root)
    assert result['environment']['MAGICANVAS_REQUIRE_QWEN_ADAPTER'] == '1'
    assert result['environment']['MAGICANVAS_QWEN_ADAPTER_DIR'] == str(root / 'adapter')
    assert not result['release_provenance_pinned']
    assert dict(os.environ) == before


def test_wrong_declared_base_is_rejected_before_adapter_loading(package):
    root, profile = package
    profile['training_base_alias'] = 'different-base'
    with patch('scripts.verify_qwen_adapter.verify') as verify:
        with pytest.raises(ValueError, match='training-base metadata differ'):
            inspect_profile(profile, root)
        verify.assert_not_called()


def test_prompt_contract_mismatch_rejected(package):
    root, profile = package
    with patch('scripts.verify_qwen_adapter.verify', return_value=dict(prompt_version=3)):
        with pytest.raises(ValueError, match='prompt contracts differ'):
            inspect_profile(profile, root)


def test_missing_local_tokenizer_rejected(package):
    root, profile = package
    (root / 'base/tokenizer.json').unlink()
    with patch('scripts.verify_qwen_adapter.verify', return_value=dict(prompt_version=2)):
        with pytest.raises(ValueError, match='Missing offline tokenizer'):
            inspect_profile(profile, root)
