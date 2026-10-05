"""Environments this package gives, each named by `module:name` wherever an environment is named (a run, a suite's
entry, the offers).

- `gsm8k`: GSM8K, grade-school math word problems, from the Environments Hub (`primeintellect/gsm8k` 0.1.4, a
  dependency of this project), played by verifiers' tool-less `null` harness: the model answers in one turn, giving its
  final number after `#### `, and `math-verify` compares it with the problem's (reward 1 or 0). It trains on the 7,473
  problems of the train split. Its eval data is the test split: all 1,319 problems (`gsm8k-test`), and the first 100
  (`gsm8k-test-100`).
"""

from rollout_verifiers.environment import VerifiersEnvironment

__all__ = ["gsm8k"]

gsm8k = VerifiersEnvironment(
    "primeintellect/gsm8k", train={"split": "train"}, eval={"split": "test"}, eval_subsets=[100]
)
