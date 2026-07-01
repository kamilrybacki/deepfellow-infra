import type { SpecField } from "@/deepfellow/types";
import { describe, expect, it } from "vitest";
import { initFormData, mergeInitialData } from "./DynamicFormFields";

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
