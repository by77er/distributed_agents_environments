"""GSM8K from the Environments Hub (`primeintellect/gsm8k`), played by verifiers' tool-less `null` harness.

Install the environment first: `uv pip install` its wheel, version 0.1.4 at
https://hub.primeintellect.ai/primeintellect/gsm8k/@eb4818f9/gsm8k-0.1.4-py3-none-any.whl (an exact `uv sync`
takes it out again). Train on `prime_gsm8k:train`; make suites of `prime_gsm8k:test`. (This module is not called
`gsm8k`: that is the environment's own package.)
"""

from rollout_verifiers import VerifiersEnvironment

train = VerifiersEnvironment("primeintellect/gsm8k", train={"split": "train"}, eval={"split": "test"})
test = train.evaluation()
