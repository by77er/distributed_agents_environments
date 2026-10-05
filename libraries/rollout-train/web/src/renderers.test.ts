import { describe, expect, it } from "vitest";
import type { OfferedProvider } from "./api/types";
import { rendererFor } from "./lib/form";

const provider = {
  name: "local-vllm", kind: "vllm", gpus: 1, replicas: 1, capabilities: { token_exact: true }, allocation: "scheduled", concurrency: null, weights: ["lora"],
  models: [
    { model: "Qwen/Qwen3-0.6B", renderers: ["rollout_qwen:qwen3"] },
    { model: "Qwen/Qwen3.5-4B", renderers: ["rollout_qwen:qwen35"] },
    { model: "org/unknown", renderers: [] },
  ],
} as unknown as OfferedProvider;

describe("a channel's renderer", () => {
  it("follows its model: the one said where it still renders the model, else the one that does", () => {
    expect(rendererFor(provider, "Qwen/Qwen3.5-4B", undefined)).toBe("rollout_qwen:qwen35");
    expect(rendererFor(provider, "Qwen/Qwen3-0.6B", "rollout_qwen:qwen35")).toBe("rollout_qwen:qwen3");  // (the model changed)
    expect(rendererFor(provider, "Qwen/Qwen3.5-4B", "rollout_qwen:qwen35")).toBe("rollout_qwen:qwen35");
  });

  it("is none where no renderer renders the model", () => {
    expect(rendererFor(provider, "org/unknown", "rollout_qwen:qwen3")).toBeUndefined();
  });
});
