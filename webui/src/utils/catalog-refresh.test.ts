/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { describe, expect, it } from "vitest";
import {
  getCatalogRefreshToast,
  getRefreshButtonLabel,
  serviceHasCatalogRefresh,
} from "./catalog-refresh";

describe("serviceHasCatalogRefresh", () => {
  it.each(["ollama", "vllm", "llamacpp", "sglang"])(
    "returns true for %s",
    (type) => {
      expect(serviceHasCatalogRefresh(type)).toBe(true);
    },
  );

  it.each(["ollama-external", "custom", undefined])(
    "returns false for %s",
    (type) => {
      expect(serviceHasCatalogRefresh(type)).toBe(false);
    },
  );
});

describe("getCatalogRefreshToast", () => {
  const base = {
    syncFailed: false,
    catalogFailed: false,
    isOllamaExternal: false,
    catalogAdded: null as number | null,
  };

  it("reports both failing", () => {
    const result = getCatalogRefreshToast({
      ...base,
      syncFailed: true,
      catalogFailed: true,
    });
    expect(result).toEqual({
      variant: "error",
      message: "Failed to refresh models and catalog.",
    });
  });

  it("reports sync failure alone, phrased for ollama-external", () => {
    const result = getCatalogRefreshToast({
      ...base,
      syncFailed: true,
      isOllamaExternal: true,
    });
    expect(result.variant).toBe("error");
    expect(result.message).toContain("Failed to sync models");
  });

  it("reports catalog failure alone with the underlying reason", () => {
    const result = getCatalogRefreshToast({
      ...base,
      catalogFailed: true,
      catalogFailureReason: "HTTP 429: throttled",
    });
    expect(result.variant).toBe("error");
    expect(result.message).toBe(
      "Models refreshed successfully, but catalog refresh failed: HTTP 429: throttled",
    );
  });

  it("reports catalog failure alone without a reason", () => {
    const result = getCatalogRefreshToast({ ...base, catalogFailed: true });
    expect(result.message).toBe(
      "Models refreshed successfully, but catalog refresh failed.",
    );
  });

  it("reports success with new models added", () => {
    const result = getCatalogRefreshToast({ ...base, catalogAdded: 3 });
    expect(result).toEqual({
      variant: "success",
      message: "Models refreshed successfully. 3 new models added to catalog.",
    });
  });

  it("uses singular wording for exactly one model added", () => {
    const result = getCatalogRefreshToast({ ...base, catalogAdded: 1 });
    expect(result.message).toBe(
      "Models refreshed successfully. 1 new model added to catalog.",
    );
  });

  it("reports success with no new models found when the refresh ran but found nothing", () => {
    const result = getCatalogRefreshToast({ ...base, catalogAdded: 0 });
    expect(result.message).toBe(
      "Models refreshed successfully. No new models found.",
    );
  });

  it("omits catalog messaging entirely when no refresh ran (e.g. ollama-external)", () => {
    const result = getCatalogRefreshToast({
      ...base,
      isOllamaExternal: true,
      catalogAdded: null,
    });
    expect(result.message).toBe("Models synced successfully.");
  });
});

describe("getRefreshButtonLabel", () => {
  it("shows Refreshing while pending for a catalog-backed service", () => {
    expect(getRefreshButtonLabel(true, false)).toBe("Refreshing…");
  });

  it("shows Refresh when idle", () => {
    expect(getRefreshButtonLabel(false, false)).toBe("↺ Refresh");
  });

  it("shows Syncing while pending for ollama-external", () => {
    expect(getRefreshButtonLabel(true, true)).toBe("Syncing…");
  });

  it("shows Sync when idle for ollama-external", () => {
    expect(getRefreshButtonLabel(false, true)).toBe("↺ Sync");
  });
});
