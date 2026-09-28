import rollout
import rollout.core


def test_package_imports() -> None:
    assert rollout.__version__ == "0.0.0"
    assert rollout.core.__doc__ is not None
