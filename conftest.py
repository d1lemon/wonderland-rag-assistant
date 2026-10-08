import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import os
from types import SimpleNamespace
from unittest.mock import Mock, patch


_test_supabase_patch = None
_test_environment_before = {}


def pytest_configure(config):
    global _test_supabase_patch, _test_environment_before

    dummy_environment = {
        "SUPABASE_URL": "https://supabase.invalid",
        "SUPABASE_ANON_KEY": "unit-test-anon-key",
        "SUPABASE_SERVICE_ROLE_KEY": "unit-test-service-role-key",
    }

    _test_environment_before = {
        name: os.environ.get(name)
        for name in dummy_environment
    }
    os.environ.update(dummy_environment)

    def make_test_client(*args, **kwargs):
        def unexpected_call(*args, **kwargs):
            raise AssertionError(
                "Unexpected Supabase call in isolated unit tests."
            )

        return SimpleNamespace(
            auth=SimpleNamespace(
                get_user=Mock(side_effect=unexpected_call),
            ),
            table=Mock(side_effect=unexpected_call),
            rpc=Mock(side_effect=unexpected_call),
        )

    _test_supabase_patch = patch(
        "supabase.create_client",
        side_effect=make_test_client,
    )
    _test_supabase_patch.start()


def pytest_unconfigure(config):
    global _test_supabase_patch

    if _test_supabase_patch is not None:
        _test_supabase_patch.stop()
        _test_supabase_patch = None

    for name, previous_value in _test_environment_before.items():
        if previous_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous_value
