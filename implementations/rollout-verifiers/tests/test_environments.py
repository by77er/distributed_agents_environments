"""The environments the package gives, by `module:name`: GSM8K from the Environments Hub, with all of its test split as
eval data and its first 100 problems as eval data of their own. No network: the tasks are not loaded, but stood in
for where the eval data is listed."""

import copy
import importlib.metadata
from typing import Any

import pytest

pytest.importorskip("verifiers")

from rollout.names import named
from rollout_verifiers import VerifiersEnvironment, VerifiersProgram

GSM8K = "rollout_verifiers.environments:gsm8k"


def test_gsm8k_is_an_environment_named_by_module_and_name() -> None:
    gsm8k = named(GSM8K)
    assert isinstance(gsm8k, VerifiersEnvironment) and gsm8k.program.program.endswith("VerifiersProgram")
    assert gsm8k.configuration(gsm8k.eval)["taskset"] == {"id": "primeintellect/gsm8k", "split": "test"}
    assert gsm8k.configuration(gsm8k.train)["taskset"] == {"id": "primeintellect/gsm8k", "split": "train"}
    assert gsm8k.configuration(gsm8k.eval)["agent"] == {"harness": {"id": "null"}, "runtime": {"type": "subprocess"}}
    assert gsm8k.eval_size is None and gsm8k.eval_subsets == (100,)  # (all of the test split, and its first 100)
    assert importlib.metadata.version("gsm8k") == "0.1.4"  # (the Hub's package, a dependency of the project)
    assert gsm8k.version.startswith("gsm8k 0.1.4, verifiers 0.3.2")


def test_its_eval_data_is_every_test_problem_and_the_first_100_the_same_starts() -> None:
    gsm8k: Any = copy.copy(named(GSM8K))
    problems = [{"idx": index, "prompt": f"Problem {index}.", "answer": str(index)} for index in range(1319)]
    gsm8k.__dict__["eval_tasks"] = problems  # (as the test split loads: 1,319 problems)
    evals = gsm8k.evals()
    assert {name: len(starts) for name, starts in evals.items()} == {"gsm8k-test": 1319, "gsm8k-test-100": 100}
    assert evals["gsm8k-test-100"] == evals["gsm8k-test"][:100]  # (a problem is the same start in each)
    first = evals["gsm8k-test"][0]
    assert (first.task, first.title, first.seed) == ("eval", "task 0", 0)
    parameters: Any = first.parameters
    assert parameters["task"] == problems[0] and parameters["environment"]["taskset"]["split"] == "test"
    assert isinstance(VerifiersProgram(parameters), VerifiersProgram)


def test_an_eval_subset_is_its_first_tasks_and_one_as_large_as_the_whole_is_left_out() -> None:
    environment: Any = VerifiersEnvironment("any/words", eval={"split": "test"}, eval_subsets=[2, 5, 2])
    environment.__dict__["eval_tasks"] = [{"idx": index, "prompt": "Say it."} for index in range(5)]
    assert {name: len(starts) for name, starts in environment.evals().items()} == {"words-test": 5, "words-test-2": 2}
    assert VerifiersEnvironment("any/words", eval={"split": "test"}).version != environment.version
    with pytest.raises(ValueError, match="one task at least"):
        VerifiersEnvironment("any/words", eval={"split": "test"}, eval_subsets=[0])
