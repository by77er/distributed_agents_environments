# Glossary

Status: **Proposed**

| Term | Meaning |
|---|---|
| **Core** | The Python library that defines and runs everything task authors, agent authors and trainers use. See [layers and profiles](layers-and-profiles.md). |
| **Layer** | Core, inference, environments, durability or platform. Only the core and some form of inference are required. |
| **Deployment profile** | Local, cluster or fleet: which layers are present and which implementation each protocol uses. |
| **Program** | Durable-able `async` code with a `main(run)` method. The agent loop is one program (`AgentProgram`). |
| **Task** | The environment an agent acts in, in the reinforcement-learning sense: tools, lifecycle hooks, responses to model turns, scoring. See [task](../core/harness/task.md). |
| **Agent** | The policy side of the loop: selects what the model sees and produces one action per turn. See [agent](../core/harness/agent.md). |
| **Harness** | The framework-owned loop that drives a task with an agent. See [harness](../core/harness/README.md). |
| **Run** | One episode of a program under a `RunBinding`. Identified by `run_id`. |
| **RunSpecification / RunBinding** | What to run (program reference and parameters) / how it is bound in this deployment (model endpoints, imported tools, environment binding, delivery policy). |
| **Deployment** | A named, addressable `RunSpecification`, e.g. `acme/support-bot`. |
| **Runner** | Executes runs: `LocalRunner` (core, in process) or `DurableRunner` (durability layer). |
| **Observation** | The environment's response to a model turn: messages shown to the model next, optional reward, optional ending, logged info. |
| **Ending** | How an episode ended: `TERMINATED` (a real end state) or `TRUNCATED` (stopped by a limit). |
| **WaitFor** | A return value of `start`, `respond` or `resume` that suspends the run until a message arrives or a timeout passes. |
| **Conversation** | A sequence of runs of one deployment keyed by a conversation key; at most one run consumes its messages at a time. See [conversations](../core/harness/conversations.md). |
| **Envelope** | A message: kind, canonical content, optional data, sender, reply address, `message_id`. |
| **Priority / delivery mode** | `LOW` / `NORMAL` / `HIGH`, mapped by the deployment to `QUEUE` (wait for the next `WaitFor`), `STEER` (merge into the next observation) or `INTERRUPT` (cancel the in-flight sample). |
| **Model slot** | A named model a task declares (`policy`, `user`, …); each is bound to an endpoint and recorded as its own session. |
| **Effect** | An operation that reaches outside code (model sample, tool call, environment operation, message, child run, timer). Identified by `effect_id = {run_id}:{generation}:{ordinal}` plus an argument digest. |
| **Generation** | A segment of a long run; a run hands over to a new generation from exported state to bound replay. |
| **Replay** | Re-running code while resolving its effects from recorded results; how a durable runner resumes. |
| **Pump** | The trusted DBOS workflow in the durable runner that performs each effect a task host requests as a durable step. |
| **Task host** | The sandboxed process that runs program, task and agent code for the durable runner on a deterministic event loop. |
| **Trust tier** | T0–T3: how task hosts are sandboxed, pooled and placed depending on who writes the code. |
| **Tool** | A `ToolSpecification` (what the model sees) plus its implementation: a `@tool` method, or an imported tool. Memory and other cross-run state are tools. |
| **Imported tool / ToolBinding** | An external tool set (MCP server, HTTP service, another agent, a human) declared by a task and bound per run. |
| **Canonical content** | Model-agnostic messages and content blocks; the only content form code, run events and tool bindings use. |
| **Model endpoint** | Anything implementing the [model endpoint contract](../contracts/model-endpoint.md): the recorder or a direct adapter. |
| **Recorder** | The model endpoint that owns tokenization for recorded channels and records what was sampled. |
| **Session / epoch** | The recorder's record of one model slot of one run / one token sequence of it: a context that only grew, with the spans the policy sampled. |
| **Renderer** | Chat template + tokenizer + parser for one model family. |
| **Behavior logprob** | Log-probability of a sampled token under the distribution it was actually sampled from. |
| **Routed experts** | For mixture-of-experts policies, the experts selected at each position; recorded for routing replay in training. |
| **Channel** | A policy being served, by name: its engines, its limits, and the weights version it samples from. |
| **Policy / weights version** | Whatever produces tokens (a weights version + renderer + sampling configuration, or an API model) / a monotonic version within a policy lineage. |
| **Rollout job** | Runs task rows submitted by a caller and delivers their `Episode`s through one ordered log. Knows no algorithm concepts. |
| **Episode** | A finished run as training sees it: labels, outcome, result, and for each trainable slot its epochs with behavior logprobs, per-span weights versions and rewards. |
| **Catalog** | What an environment offers to train on: rows, easiest first, and how to draw a start of one. |
| **Profile** | A deployment, described: channels and engines, the trainer, the runner, where tool sets live. |
| **Environment** | A computer a task can create through the `Environments` protocol; provided by the separately designed [environment system](../environments/README.md). |
| **Cell** | Fleet profile only: a self-contained slice (cluster, Postgres, executors); the unit of scale and failure. |
| **Unmanaged harness** | A third-party agent that uses the recorder's compatible endpoints (and optionally the tool router's MCP facade) directly: trainable, not durable. |
