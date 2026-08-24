/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import type { SpecField } from "@/deepfellow/types";
import { describe, expect, it } from "vitest";
import {
  filterExistingModelsForCollisionCheck,
  hasFormDataChanged,
  initFormData,
  mergeInitialData,
  validateFields,
} from "./DynamicFormFields";

const ollamaExternalFields: SpecField[] = [
  { name: "id", description: "Model ID", type: "text", required: true },
  {
    name: "type",
    description: "Model type",
    type: "oneof",
    required: true,
    values: ["llm", "embedding"],
  },
  { name: "size", description: "Model size", type: "text", required: false },
];

describe("mergeInitialData", () => {
  it("pre-fills a oneof field from initial data instead of the first value default", () => {
    const merged = mergeInitialData(ollamaExternalFields, {
      id: "my-emb",
      type: "embedding",
      size: "500MB",
    });

    expect(merged.type).toBe("embedding");
    expect(merged.id).toBe("my-emb");
    expect(merged.size).toBe("500MB");
  });

  it("falls back to the first oneof value when no initial data is given", () => {
    const merged = mergeInitialData(ollamaExternalFields, undefined);

    expect(merged.type).toBe("llm");
  });

  it("keeps the first-value default when initial data omits the oneof field", () => {
    const merged = mergeInitialData(ollamaExternalFields, { id: "x" });

    expect(merged.type).toBe("llm");
    expect(merged.id).toBe("x");
  });
});

describe("initFormData", () => {
  it("defaults a oneof field to its first value", () => {
    expect(initFormData(ollamaExternalFields).type).toBe("llm");
  });
});

const envsField: SpecField = {
  name: "envs",
  description: "Custom environmental variables.",
  type: "map",
  required: false,
  required_keys: ["BRAVE_API_KEY"],
};

describe("validateFields - required map keys", () => {
  it("flags a missing required key when the map is empty", () => {
    const errors = validateFields([envsField], { envs: {} });
    expect(errors.envs).toContain("BRAVE_API_KEY");
  });

  it("flags a required key present with an empty value", () => {
    const errors = validateFields([envsField], {
      envs: { BRAVE_API_KEY: "" },
    });
    expect(errors.envs).toContain("BRAVE_API_KEY");
  });

  it("flags a required key present with whitespace-only value", () => {
    const errors = validateFields([envsField], {
      envs: { BRAVE_API_KEY: "   " },
    });
    expect(errors.envs).toContain("BRAVE_API_KEY");
  });

  it("passes when the required key has a non-empty value", () => {
    const errors = validateFields([envsField], {
      envs: { BRAVE_API_KEY: "secret" },
    });
    expect(errors.envs).toBeUndefined();
  });

  it("lists every missing required key", () => {
    const field: SpecField = { ...envsField, required_keys: ["A", "B"] };
    const errors = validateFields([field], { envs: { A: "set" } });
    expect(errors.envs).toContain("B");
    expect(errors.envs).not.toContain("A");
  });

  it("ignores a map field with no required keys", () => {
    const field: SpecField = { ...envsField, required_keys: undefined };
    const errors = validateFields([field], { envs: {} });
    expect(errors.envs).toBeUndefined();
  });
});

const idAndPrefixFields: SpecField[] = [
  { name: "id", description: "Model ID", type: "text", required: true },
  {
    name: "default_prefix",
    description: "Endpoint prefix",
    type: "text",
    required: false,
  },
];

describe("validateFields - id/prefix collisions", () => {
  it("flags an id already used by another model", () => {
    const errors = validateFields(
      idAndPrefixFields,
      { id: "open-websearch", default_prefix: "new-prefix" },
      [{ id: "open-websearch", effective_prefix: "open-websearch" }],
    );
    expect(errors.id).toBe("This id is already in use.");
  });

  it("flags a default_prefix already used by another model, naming it", () => {
    const errors = validateFields(
      idAndPrefixFields,
      { id: "new-id", default_prefix: "open-websearch" },
      [{ id: "open-websearch", effective_prefix: "open-websearch" }],
    );
    expect(errors.default_prefix).toBe(
      'This prefix is already used by "open-websearch".',
    );
  });

  it("passes when the id/prefix don't collide with any existing model", () => {
    const errors = validateFields(
      idAndPrefixFields,
      { id: "new-id", default_prefix: "new-prefix" },
      [{ id: "open-websearch", effective_prefix: "open-websearch" }],
    );
    expect(errors.id).toBeUndefined();
    expect(errors.default_prefix).toBeUndefined();
  });

  it("ignores collisions when no existingModels are given", () => {
    const errors = validateFields(idAndPrefixFields, {
      id: "open-websearch",
      default_prefix: "open-websearch",
    });
    expect(errors.id).toBeUndefined();
    expect(errors.default_prefix).toBeUndefined();
  });

  it("doesn't check default_prefix collisions when the field isn't in the form", () => {
    const errors = validateFields([idAndPrefixFields[0]], { id: "new-id" }, [
      { id: "open-websearch", effective_prefix: "open-websearch" },
    ]);
    expect(errors.default_prefix).toBeUndefined();
  });
});

describe("filterExistingModelsForCollisionCheck", () => {
  const models = [
    { id: "open-websearch", effective_prefix: "open-websearch" },
    { id: "lemmatizer", effective_prefix: "lemmatizer" },
  ];

  it("excludes the model being edited when not duplicating", () => {
    const result = filterExistingModelsForCollisionCheck(
      models,
      "open-websearch",
      false,
    );
    expect(result).toEqual([
      { id: "lemmatizer", effective_prefix: "lemmatizer" },
    ]);
  });

  it("keeps every model, including the source, when duplicating", () => {
    const result = filterExistingModelsForCollisionCheck(
      models,
      "open-websearch",
      true,
    );
    expect(result).toEqual(models);
  });

  it("is a no-op when there's no original id (a plain add, not an edit)", () => {
    const result = filterExistingModelsForCollisionCheck(
      models,
      undefined,
      false,
    );
    expect(result).toEqual(models);
  });
});

describe("hasFormDataChanged", () => {
  it("returns false for identical data", () => {
    const data = { id: "my-custom", envs: { A: "1" }, volumes: ["/a:/b"] };
    expect(hasFormDataChanged({ ...data }, data)).toBe(false);
  });

  it("returns true when a primitive field differs", () => {
    expect(hasFormDataChanged({ id: "new-id" }, { id: "my-custom" })).toBe(
      true,
    );
  });

  it("returns true when a nested map value differs", () => {
    expect(hasFormDataChanged({ envs: { A: "2" } }, { envs: { A: "1" } })).toBe(
      true,
    );
  });

  it("returns false when nested maps/lists are equal but different object instances", () => {
    expect(
      hasFormDataChanged(
        { envs: { A: "1" }, volumes: ["/a:/b"] },
        { envs: { A: "1" }, volumes: ["/a:/b"] },
      ),
    ).toBe(false);
  });

  it("returns true when a key is only present on one side", () => {
    expect(hasFormDataChanged({ id: "x", extra: "y" }, { id: "x" })).toBe(true);
  });

  it("returns true when an array length differs", () => {
    expect(
      hasFormDataChanged(
        { volumes: ["/a:/b", "/c:/d"] },
        { volumes: ["/a:/b"] },
      ),
    ).toBe(true);
  });
});
