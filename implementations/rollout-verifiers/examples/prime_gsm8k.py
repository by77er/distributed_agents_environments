"""GSM8K from the Environments Hub (`primeintellect/gsm8k` 0.1.4), played by verifiers' tool-less `null` harness.

The project's `spike` group installs the environment (`uv sync --group spike`). Train on `prime_gsm8k:train`; make
suites of `prime_gsm8k:test`. (This module is not called `gsm8k`: that is the environment's own package.)
"""

from rollout_verifiers import VerifiersEnvironment

train = VerifiersEnvironment("primeintellect/gsm8k", train={"split": "train"}, eval={"split": "test"})
test = train.evaluation()
