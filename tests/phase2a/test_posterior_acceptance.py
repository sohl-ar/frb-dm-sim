"""Full trained-model gates. No numerical criteria are changed here."""
import pytest
from frbsbi.acceptance import Acceptance,GATE_ORDER

pytestmark = pytest.mark.phase2a


@pytest.mark.parametrize("name",GATE_ORDER)
def test_conditioned_posterior_gate(request,name):
    if not hasattr(request.config,"phase2a_runtime"):
        request.config.phase2a_runtime = Acceptance()
    request.config.phase2a_runtime.run(name)
