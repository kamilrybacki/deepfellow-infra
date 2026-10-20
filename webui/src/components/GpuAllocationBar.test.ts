/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { describe, expect, it } from "vitest";
import { computeGpuAllocationLayout } from "./GpuAllocationBar";

describe("computeGpuAllocationLayout", () => {
  it("computes percentages for existing segments against total VRAM", () => {
    const layout = computeGpuAllocationLayout(
      [
        { id: "a", label: "model-a", gb: 4 },
        { id: "b", label: "model-b", gb: 2 },
      ],
      10,
    );

    expect(layout.segments).toEqual([
      { id: "a", label: "model-a", gb: 4, pct: 40 },
      { id: "b", label: "model-b", gb: 2, pct: 20 },
    ]);
    expect(layout.freeGb).toBe(4);
    expect(layout.freePct).toBe(40);
    expect(layout.otherGb).toBe(0);
    expect(layout.newModelGb).toBeUndefined();
  });

  it("carves the new model's share out of remaining free space, not existing segments", () => {
    const layout = computeGpuAllocationLayout(
      [{ id: "a", label: "model-a", gb: 4 }],
      10,
      { newModelPriority: 0.5 },
    );

    expect(layout.segments).toEqual([
      { id: "a", label: "model-a", gb: 4, pct: 40 },
    ]);
    // Same sizing as the backend: 0.5 * min(0.9 * 10, 6 free * 0.9) = 2.7.
    expect(layout.newModelGb).toBeCloseTo(2.7);
    expect(layout.newModelPct).toBeCloseTo(27);
    expect(layout.freeGb).toBeCloseTo(3.3);
    expect(layout.freePct).toBeCloseTo(33);
  });

  it("clamps an out-of-range priority to [0, 1]", () => {
    const over = computeGpuAllocationLayout([], 10, { newModelPriority: 1.5 });
    expect(over.newModelGb).toBeCloseTo(9);

    const under = computeGpuAllocationLayout([], 10, {
      newModelPriority: -0.5,
    });
    expect(under.newModelGb).toBe(0);
  });

  it("floors negative segment sizes and free space at zero instead of going negative", () => {
    const layout = computeGpuAllocationLayout(
      [{ id: "a", label: "model-a", gb: -2 }],
      10,
    );
    expect(layout.segments[0]?.gb).toBe(0);
    expect(layout.freeGb).toBe(10);

    const overAllocated = computeGpuAllocationLayout(
      [{ id: "a", label: "model-a", gb: 12 }],
      10,
    );
    expect(overAllocated.freeGb).toBe(0);
    expect(overAllocated.freePct).toBe(0);
  });

  it("returns all-zero percentages when total VRAM is zero or unknown", () => {
    const layout = computeGpuAllocationLayout(
      [{ id: "a", label: "model-a", gb: 4 }],
      0,
      { newModelPriority: 0.5 },
    );
    expect(layout.segments[0]?.pct).toBe(0);
    expect(layout.freePct).toBe(0);
    expect(layout.newModelPct).toBe(0);
  });

  it("shows live usage beyond the listed segments as 'other' and sizes the new model off what's really free", () => {
    const layout = computeGpuAllocationLayout(
      [{ id: "a", label: "model-a", gb: 4 }],
      10,
      { usedGb: 7, newModelPriority: 1 },
    );

    expect(layout.otherGb).toBe(3);
    expect(layout.otherPct).toBe(30);
    // 1 * min(9, 3 free * 0.9) = 2.7, leaving the 0.3 GB safety margin free.
    expect(layout.newModelGb).toBeCloseTo(2.7);
    expect(layout.freeGb).toBeCloseTo(0.3);
  });

  it("never lets live usage below the listed segments shrink them, and caps usage at total VRAM", () => {
    const under = computeGpuAllocationLayout(
      [{ id: "a", label: "model-a", gb: 4 }],
      10,
      { usedGb: 2 },
    );
    expect(under.otherGb).toBe(0);
    expect(under.freeGb).toBe(6);

    const over = computeGpuAllocationLayout([], 10, {
      usedGb: 15,
      newModelPriority: 1,
    });
    expect(over.otherGb).toBe(10);
    expect(over.freeGb).toBe(0);
    expect(over.newModelGb).toBe(0);
  });

  it("rounds like the backend: the grantable fraction and the utilization each to 2 decimals", () => {
    // free fraction 0.333 * 0.9 = 0.2997 -> 0.3; 0.33 * 0.3 = 0.099 -> 0.1 of 30 GB = 3 GB.
    const layout = computeGpuAllocationLayout([], 30, {
      usedGb: 20,
      newModelPriority: 0.33,
    });
    expect(layout.newModelGb).toBeCloseTo(3);
    expect(layout.newModelBelowMinimum).toBe(false);
  });

  it("flags a share that rounds below the backend's 1% minimum", () => {
    // 0.01 * round(0.5 * 0.9, 2) = 0.0045 -> 0.00, which the backend rejects with a 422.
    const layout = computeGpuAllocationLayout([], 10, {
      usedGb: 5,
      newModelPriority: 0.01,
    });
    expect(layout.newModelGb).toBe(0);
    expect(layout.newModelBelowMinimum).toBe(true);
  });

  it("does not flag the minimum when total VRAM is unknown", () => {
    const layout = computeGpuAllocationLayout([], 0, {
      newModelPriority: 0.01,
    });
    expect(layout.newModelBelowMinimum).toBe(false);
  });

  it("uses a fixed size instead of recomputing when one is given, capped at free space", () => {
    const layout = computeGpuAllocationLayout(
      [{ id: "a", label: "model-a", gb: 4 }],
      10,
      { newModelPriority: 0.5, newModelFixedGb: 3.5 },
    );
    expect(layout.newModelGb).toBe(3.5);
    expect(layout.freeGb).toBeCloseTo(2.5);
    expect(layout.newModelBelowMinimum).toBe(false);

    const capped = computeGpuAllocationLayout([], 10, {
      usedGb: 8,
      newModelPriority: 0.5,
      newModelFixedGb: 5,
    });
    expect(capped.newModelGb).toBe(2);
    expect(capped.freeGb).toBe(0);
  });
});
