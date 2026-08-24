/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { describe, expect, it } from "vitest";
import { applyDuplicateMode } from "./DynamicFormModal";

describe("applyDuplicateMode", () => {
  it("clears the id field when duplicate mode is enabled", () => {
    const result = applyDuplicateMode(
      { id: "my-custom", image: "test/image:latest" },
      { id: "my-custom" },
      true,
    );

    expect(result.id).toBe("");
    expect(result.image).toBe("test/image:latest");
  });

  it("restores the original id when duplicate mode is disabled", () => {
    const result = applyDuplicateMode(
      { id: "", image: "test/image:latest" },
      { id: "my-custom" },
      false,
    );

    expect(result.id).toBe("my-custom");
  });

  it("does not mutate other fields in either direction", () => {
    const formData = { id: "my-custom", size: "1GB", envs: { A: "1" } };

    const duplicated = applyDuplicateMode(formData, { id: "my-custom" }, true);
    const restored = applyDuplicateMode(duplicated, { id: "my-custom" }, false);

    expect(restored).toEqual(formData);
  });

  it("clears default_prefix when duplicate mode is enabled", () => {
    const result = applyDuplicateMode(
      { id: "my-custom", default_prefix: "custom-pfx" },
      { id: "my-custom", default_prefix: "custom-pfx" },
      true,
    );

    expect(result.default_prefix).toBe("");
  });

  it("restores default_prefix from the original snapshot, not from formData, when toggled off", () => {
    // Regression: toggling on then off used to restore default_prefix from `formData` - which by
    // the "off" call is already the blanked duplicate-mode value - instead of the true original.
    const original = { id: "my-custom", default_prefix: "custom-pfx" };

    const duplicated = applyDuplicateMode(original, original, true);
    const restored = applyDuplicateMode(duplicated, original, false);

    expect(restored.id).toBe("my-custom");
    expect(restored.default_prefix).toBe("custom-pfx");
  });

  it("never touches default_prefix for a model type that has no such field", () => {
    const result = applyDuplicateMode(
      { id: "my-custom", image: "test/image:latest" },
      { id: "my-custom" },
      true,
    );

    expect("default_prefix" in result).toBe(false);
  });
});
