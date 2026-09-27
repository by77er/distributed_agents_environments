# Glossary

Status: **Proposed**

| Term | Meaning |
|---|---|
| **Task** | The environment an agent acts in, in the reinforcement-learning sense: compute environments, tools, lifecycle hooks, responses to model turns, scoring. A Python class. See [task](../components/harness/task.md). |
| **Agent** | The policy side of the loop: selects what the model sees and produces one action per turn. A Python class. See [agent](../components/harness/agent.md). |
| **Harness** | The framework-owned rollout loop that drives a task with an agent. See [harness](../components/harness/README.md). |
| **Task host** | Python process that runs the loop, task and agent code under deterministic replay; implements `HarnessHost` for the runtime. |
| **Run** | One episode: a task and an agent under a `RunBinding`. Identified by `run_id`. Its durable state is its log. |
| **RunSpecification** | What a client submits: task reference + parameters, agent reference + configuration, `RunBinding`. |
| **RunBinding** | Deployment-specific part of a run: which endpoint serves each model slot, which binding serves each imported tool set, concrete security profiles per class, placement preferences, durability. |
| **Observation** | The environment's response to a model turn: messages shown to the model next, optional reward, optional ending, logged info. |
| **Ending** | How an episode ended: `TERMINATED` (a real end state) or `TRUNCATED` (stopped by a limit). |
| **Model slot** | A named model a task declares (`policy`, `user`, …). Each slot is bound to an endpoint and recorded as its own session. |
| **Replay** | Re-running task and agent code while resolving its effects from the log. The mechanism of durability. |
| **Checkpoint** | A memoized unit of work (`@checkpoint`): on replay its recorded return value and task state are restored instead of re-running it. |
| **Snapshot (task)** | Pickled task and agent state at a resumable point; bounds replay. Never moves between runs. Disposable. |
| **Partition** | `hash(run_id) mod P` bucket within a cell. The unit of leasing and failover. |
| **Lease / epoch** | Time-bounded ownership of a partition by one worker. The epoch is a monotonically increasing fencing token checked on every append. |
| **Event** | An immutable record in a run's log at position `seq`. |
| **Effect** | Side-effecting work requested by task code (model sample, environment operation, imported tool call, spawn, send, timer). Identified by `effect_id`. |
| **Inbox** | Per-run queue of externally delivered inputs (signals, messages, long-running completions). Consumed into the log. |
| **Signal** | External input to a run: user message, cancel, custom. |
| **Canonical content** | Model-agnostic messages and content blocks. The only content form task and agent code, the run log and the tool router use. |
| **Tool** | A `ToolSpecification` (what the model sees) plus its implementation: a `@tool` method in the task, an environment-provided tool, or an imported tool. |
| **Imported tool** | An external tool (MCP server, HTTP service, another agent, a human) declared by name in a task and bound per run; executed by the tool router. |
| **Binding** | An implementation of an imported tool set: `mcp`, `http`, `agent`, `human`. |
| **Environment** | A stateful compute resource with a lifecycle, reached through the Environment API. In task code, a handle object. |
| **EnvironmentSpecification** | What a task asks for when creating an environment: image or template, resources, security class, persistence, lifetime. |
| **Driver** | An implementation of environment lifecycle for one backend (firecracker, pod, vps, local). |
| **envlet** | Per-host daemon that runs microVMs, enforces security profiles, and proxies the Environment API over vsock. |
| **envd** | In-guest daemon that serves the Environment API. Untrusted. |
| **SecurityProfile / security class** | Declared isolation, network, credential, limit requirements of an environment; a class is a named minimum that a `RunBinding` maps to a concrete profile. |
| **Template** | A content-addressed build recipe (base image + files + commands). Built once per distinct recipe into a golden snapshot; every environment from it restores that snapshot. |
| **Pool** | A warm set of restored environments of one template. |
| **Model endpoint** | Anything implementing the [model endpoint contract](../contracts/model-endpoint.md): the recorder, or a direct provider adapter. |
| **Channel** | A named, movable pointer to a policy version, with a capability contract (like a container tag). |
| **Policy** | Whatever produces tokens: a weights version + renderer + sampling configuration, an API model, or a human. |
| **Weights version** | Monotonic integer identifying one set of weights within a policy lineage. |
| **Renderer** | Chat template + tokenizer + parser for one model family. Owned by the recorder. |
| **Renderer epoch** | A span of a session within which the token sequence is extended incrementally with one renderer. |
| **Recorder** | Optional proxy between the runtime and inference that owns tokenization and records session trees. |
| **Session** | Recorder-side record of one model slot of one run: `session_id = {run_id}/{model_slot}`. |
| **Session tree** | Prefix tree of token spans for a session. A trajectory is a path in it. |
| **Behavior logprob** | Log-probability of a sampled token under the distribution it was actually sampled from. |
| **Weight Update Controller (WUC)** | Coordinates weight loads on engines: pause admission, abort in-flight, load, resume. |
| **Rollout job** | Runs task rows submitted by a caller and delivers their `Sample`s through one ordered, durable log. Knows no algorithm concepts. |
| **Sample** | Training record for one run and trainable slot: token sequences, loss mask, behavior logprobs, per-token weights versions, rewards at token positions, ending, caller labels. |
| **Cell** | Self-contained slice of the system; unit of scale and failure. |
| **Managed / unmanaged run** | Managed: a task and agent hosted by the runtime (durable). Unmanaged: a foreign harness using the recorder and tool facade directly (trainable, not durable). |
