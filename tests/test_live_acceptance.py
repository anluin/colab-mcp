import pytest

from scripts.live_acceptance import PHASES, InjectedFailure, checkpoint, worker_source
from scripts.live_workspace_probe import digest_tree


def test_worker_source_is_exactly_63_kib_and_workload_parameters_are_general():
    source = worker_source(300, 20, 1_900_000)
    assert len(source.encode("utf-8")) == 63 * 1024
    compile(source, "worker.py", "exec")


def test_workspace_probe_digest_tree_uses_posix_relative_paths(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "data.bin").write_bytes(b"verified")

    assert digest_tree(tmp_path) == {
        "nested/data.bin": "1c34f88707b55e6104c4eb20e71ffa3d33e414b71ef689a15fad0640d0ac58cb"
    }


@pytest.mark.parametrize("phase", PHASES)
def test_every_live_phase_has_a_failure_injection_boundary(phase):
    evidence = {}
    with pytest.raises(InjectedFailure, match=phase):
        checkpoint(phase, phase, evidence)
    assert evidence["phases"][-1]["phase"] == phase
