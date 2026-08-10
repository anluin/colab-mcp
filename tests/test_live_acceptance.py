import pytest

from scripts.live_acceptance import PHASES, InjectedFailure, checkpoint, worker_source


def test_worker_source_is_exactly_63_kib_and_workload_parameters_are_general():
    source = worker_source(300, 20, 1_900_000)
    assert len(source.encode("utf-8")) == 63 * 1024
    compile(source, "worker.py", "exec")


@pytest.mark.parametrize("phase", PHASES)
def test_every_live_phase_has_a_failure_injection_boundary(phase):
    evidence = {}
    with pytest.raises(InjectedFailure, match=phase):
        checkpoint(phase, phase, evidence)
    assert evidence["phases"][-1]["phase"] == phase
