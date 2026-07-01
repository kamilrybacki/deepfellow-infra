import type { SpecField } from "@/deepfellow/types";
import { describe, expect, it } from "vitest";
import {
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
