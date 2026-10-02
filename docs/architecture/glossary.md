# Glossary

| Term | Meaning |
|---|---|
| **Program** | `async` code with a `main(run)` method. The agent loop is one program (`AgentProgram`). |
| **Task** | The environment an agent acts in, in the reinforcement-learning sense: tools, lifecycle hooks, responses to model turns, scoring. See [tasks](../guide/tasks.md). |
| **Agent** | The policy side of the loop: selects what the model sees and produces one reply per turn. See [agents](../guide/agents.md). |
| **Harness** | The loop that drives a task with an agent. See [harness](../libraries/rollout/README.md). |
| **Run** | One episode of a program under a `RunBinding`. Identified by `run_id`. |
| **RunSpecification / RunBinding** | What to run (a program reference and parameters) / how it is served here (model endpoints, imported tool sets, delivery policy). |
| **Deployment** | A named, addressable `RunSpecification`. |
| **Runner** | Executes runs: `LocalRunner` in process, or `DurableRunner`, whose runs survive it ([durable runner](../implementations/rollout-durable/README.md)). |
| **Observation** | The task's response to a model turn: messages shown to the model next, an optional reward, an optional ending, logged info. |
| **Ending** | How an episode ended: `TERMINATED` (a real end state) or `TRUNCATED` (stopped by a limit). |
| **WaitFor** | A return value of `start`, `respond` or `resume` that suspends the run until a message arrives or a timeout passes. |
| **Conversation** | A sequence of runs of one deployment under a conversation key; at most one run consumes its messages at a time. See [conversations](../guide/conversations.md). |
| **Envelope** | A message: kind, canonical content, optional data, sender, reply address, `message_id`. |
| **Priority / delivery mode** | `LOW` / `NORMAL` / `HIGH`, mapped by the deployment to `QUEUE` (wait for the next `WaitFor`), `STEER` (merge into the next observation) or `INTERRUPT` (cancel the sample in flight). |
| **Model slot** | A named model a program uses (`policy`, `user`, …); each is bound to an endpoint and recorded as its own session. |
| **Effect** | An operation that reaches outside code: a model sample, a tool call, an environment operation, an output. Identified by `effect_id = {run_id}:{generation}:{ordinal}` and a digest of its arguments. |
| **Replay** | Running a program again while its recorded effects return their results; how a durable runner resumes. |
| **Tool** | A `ToolSpecification` (what the model sees) and its implementation: a `@tool` method, or an imported tool. |
| **Tool set / ToolBinding** | Tools a program imports by name / where a run finds them: in the runner's process or at a URL. |
| **Canonical content** | Model-agnostic messages and content blocks; the only content form that code, run events and tool sets use. |
| **Model endpoint** | Anything that implements the [model endpoint contract](../libraries/rollout/contracts/model-endpoint.md): the recorder or an adapter. |
| **Channel** | A trainable policy being served, by name: its engines, its limits, and the weights version it samples from. |
| **Engine** | One replica serving a model: tokens in; tokens, logprobs and a finish reason out. `VllmEngine` is one ([vLLM engine](../implementations/rollout-vllm.md)). |
| **Recorder** | The model endpoint for channels: renders contexts to tokens, samples, and keeps what was sampled. |
| **Session / epoch** | The recorder's record of one model slot of one run / one token sequence of it: a context that only grew, with the spans the policy sampled. |
| **Renderer** | The chat template, tokenizer and parser of one model family ([Qwen renderers](../implementations/rollout-qwen.md)). |
| **Behavior logprob** | The log-probability of a sampled token under the distribution it was sampled from. |
| **Weights version** | A channel's count of published weights; every sampled span carries the one it was sampled at. |
| **Rollout job** | Runs rows submitted by a caller and delivers their episodes through one ordered log. It knows no algorithm. |
| **Episode** | A finished run as training sees it: labels, outcome, result, and for each model slot its epochs with behavior logprobs, weights versions and rewards. |
| **Catalog / row** | What an environment offers to train on, easiest first / one situation of it, of which a start is drawn for each group. |
| **Group** | Episodes of one row from one start, compared with each other. |
| **Trainer** | Turns weighted token sequences into new weights, within a budget it states. `LoraTrainer` is one ([LoRA trainer](../implementations/rollout-lora.md)). |
| **Profile** | A deployment, described: channels and engines, the trainer, the runner, where tool sets live. See [deploying](../guide/deploying.md). |
| **Library / implementation / product / environment** | The four kinds of package in the repository: what code is written against; one implementation of an interface a library defines; an application; something to train on. See [overview](overview.md#layers). |
| **Environment** | A computer a task creates through `run.environments` ([computers](../implementations/rollout-computers.md)). Also, in the reinforcement-learning sense, what a task is; the packages under `environments/` are environments in that sense. |
| **Harness inside an environment** | A program's own agent, given an OpenAI-compatible address for a model slot: recorded like any other sample. |
