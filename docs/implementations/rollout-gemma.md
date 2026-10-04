# Gemma renderers

Code: `rollout_gemma`

A [`Renderer`](../guide/reference.md#renderer) is a model family's token format (see [Qwen renderers](rollout-qwen.md)
for what one does). This package has the renderer of Gemma 4. It is installed with `uv sync --extra gemma`.

```toml
[channels.policy]
model = "google/gemma-4-12B-it"
renderer = "rollout_gemma:gemma4"
```

`gemma4` is called with the channel's `model` (a checkpoint's name or path, whose tokenizer it loads with
`tokenizer_of`, or a tokenizer) and returns a [`ChatTemplateRenderer`](../guide/reference.md#chattemplaterenderer)
over the tokenizer's own chat template, which Gemma 4 ships beside its weights.

## The format

| | Gemma 4 |
|---|---|
| A turn | `<|turn>user` … `<turn|>`; the roles are `system`, `user` and `model` |
| Tools offered | `<|tool>declaration:name{description:…,parameters:{…}}<tool|>` in the system turn |
| A tool call in the model's output | `<|tool_call>call:name{key:<|"|>text<|"|>,count:3}<tool_call|>` ([`GemmaFunctionCalls`](../guide/reference.md#gemmafunctioncalls)) |
| A tool's response | `<|tool_response>response:name{value:<|"|>text<|"|>}<tool_response|>`, inside the model's own turn |
| Thinking | `<|channel>thought` … `<channel|>`, turned on by the template's `enable_thinking`, which writes `<|think|>` at the top of the system turn |
| What ends a sampled turn | `<turn|>`, `<|tool_response>` (the model waits there for its tools' results), or `<eos>`: the model's own end tokens |

A call's arguments are read as values: strings quoted with `<|"|>`, bare numbers, `true`, `false` and `null`, objects
in braces and lists in brackets. A quoted value whose parameter the tool's schema types as a number or a flag is read
as one.

## How thinking is delimited

Gemma's template asks for thinking with `enable_thinking`, and opens the thought channel in the generation prompt
itself only after a tool's response. The renderer opens it in every generation prompt (`<|channel>thought` and a
newline: [`ThinkingFormat`](../guide/reference.md#thinkingformat) with `prompt_opens`), so that the gateway can hold
thinking to its budget: thinking that runs out of it is closed with a newline and `<channel|>`, unsampled, and the
answer is sampled after it. An earlier turn's thinking is not shown again, as the template has it.
