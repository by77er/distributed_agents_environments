import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { StepProgress } from "./api/types";
import { progressParts } from "./lib/model";
import { Stepped } from "./pages/Run";

const progress = (said: Partial<StepProgress> = {}): StepProgress => ({
  step: 12, phase: "minibatch", minibatch: 23, minibatches: 58, packs: 812, packs_total: 1990, fraction: 0.413,
  seconds: 1440, tokens_per_second: 5912, eta_seconds: 2040, loss: 0.01, kl: 0.0123, max_kl: 0.05, clip_fraction: 0.02,
  gpu_gib: 61.2, peak_gpu_gib: 70.4, gpu_utilization: [97], ...said,
});

describe("a step being taken", () => {
  afterEach(cleanup);

  it("says how far it has got in one line, its percentage second", () => {
    expect(progressParts(progress()).join(" · ")).toBe("minibatch 23/58 · 41% · 5.9k tok/s · KL 0.012/0.05 · ETA 34 min");
    const starting = progress({ phase: "start", minibatch: 0, fraction: 0.031, tokens_per_second: 812, kl: null, max_kl: null, eta_seconds: null });
    expect(progressParts(starting).join(" · ")).toBe("start · 3% · 812 tok/s");
    expect(progressParts(progress({ fraction: 1.2, max_kl: null }))[1]).toBe("100%");
    expect(progressParts(progress({ max_kl: null }))[3]).toBe("KL 0.012");
  });

  it("draws a bar as long as the share done, beside the line, its percentage bold", () => {
    const { container } = render(<Stepped progress={progress()} />);
    expect(container.querySelector<HTMLElement>(".track i")!.style.width).toBe("41.3%");
    expect(container.querySelector("b")!.textContent).toBe("41%");
    expect(container.textContent).toBe("minibatch 23/58 · 41% · 5.9k tok/s · KL 0.012/0.05 · ETA 34 min");
  });
});
