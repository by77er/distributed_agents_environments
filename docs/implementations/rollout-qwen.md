# Qwen renderers

Code: `rollout_qwen`

A [`Renderer`](../guide/reference.md#renderer) is a model family's token format: it renders canonical messages and
tool specifications to prompt tokens, parses sampled tokens back into a canonical message, and says which tokens end
a turn and how thinking is delimited. The [recorder](../libraries/rollout-train/recorder.md) uses it for every sample.
This package has the renderers of two Qwen families. It is installed with `uv sync --all-extras`.

| Function | Family | Tool calls in the model's output | Thinking |
|---|---|---|---|
| `qwen35` | Qwen3.5 | `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` ([`XmlFunctionCalls`](../guide/reference.md#xmlfunctioncalls)) | The generation prompt opens the block; the model closes it |
| `qwen3` | Qwen3 | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` ([`JsonToolCalls`](../guide/reference.md#jsontoolcalls)) | The model opens and closes the block |

## What a renderer function is called with

A [profile](../guide/deploying.md) names one for a channel (`renderer = "rollout_qwen:qwen35"`) and calls it with the
channel's `model`. Given a checkpoint's name or path, the function loads that checkpoint's tokenizer (`tokenizer_of`).
Given a tokenizer, it uses it as it is. Either way it returns a
[`ChatTemplateRenderer`](../guide/reference.md#chattemplaterenderer): prompts are rendered with the tokenizer's own
chat template, and a turn ends at `<|im_end|>`.

## How thinking is delimited

Both families write thinking between `<think>` and `</think>`
([`ThinkingFormat`](../guide/reference.md#thinkingformat)).

| | Qwen3.5 | Qwen3 |
|---|---|---|
| Who opens the block | The generation prompt ends inside it (`prompt_opens`) | The model |
| A sample that never closes it | All of it is reasoning | It is text |
| Thinking that runs out of its budget | The recorder appends `forced_close` (a newline, `</think>`, two newlines) and samples the answer | Not held apart: one phase samples with the thinking and answer room together |

What precedes the last close becomes a `Reasoning` block of the canonical message; what follows is text and tool
calls. How the recorder holds thinking to its budget is described under
[thinking](../libraries/rollout-train/recorder.md#thinking).
