"""GSM8K from the Environments Hub (`primeintellect/gsm8k` 0.1.4), played by verifiers' tool-less `null` harness.

The project's `spike` group installs the environment (`uv sync --group spike`). Train on
`prime_gsm8k:environment`; its eval data, the test split's first 100 tasks, is the suite `gsm8k-test-100`. (This module
is not called `gsm8k`: that is the environment's own package.)
"""

from rollout_verifiers import VerifiersEnvironment

environment = VerifiersEnvironment(
    "primeintellect/gsm8k", train={"split": "train"}, eval={"split": "test"}, eval_size=100
)
