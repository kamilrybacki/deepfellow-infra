/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/

/** Service types whose "Refresh" button drives a real, admin-triggered catalog refresh (vs. a sync-only no-op). */
const CATALOG_REFRESH_SERVICE_TYPES = new Set([
  "ollama",
  "vllm",
  "llamacpp",
  "sglang",
  "openai",
  "claude",
  "google",
]);

/**
 * Service types whose backend `sync_models` delegates to the very same HuggingFace catalog
 * refresh. Calling sync and refresh together for these would fire two concurrent triggers at the
 * shared in-flight guard, so the caller must pick one.
 */
const SYNC_IS_CATALOG_REFRESH_SERVICE_TYPES = new Set([
  "vllm",
  "llamacpp",
  "sglang",
]);

export function serviceHasCatalogRefresh(
  serviceType: string | undefined,
): boolean {
  return !!serviceType && CATALOG_REFRESH_SERVICE_TYPES.has(serviceType);
}

export function syncModelsTriggersCatalogRefresh(
  serviceType: string | undefined,
): boolean {
  return (
    !!serviceType && SYNC_IS_CATALOG_REFRESH_SERVICE_TYPES.has(serviceType)
  );
}

export interface CatalogRefreshOutcome {
  syncFailed: boolean;
  catalogFailed: boolean;
  isOllamaExternal: boolean;
  catalogAdded: number | null;
  catalogFailureReason?: string;
}

export interface CatalogRefreshToast {
  variant: "success" | "error";
  message: string;
}

/**
 * Build the single toast message for a completed refresh/sync attempt.
 *
 * `catalogAdded` is `null` when no catalog refresh ran at all (e.g. ollama-external, which only
 * syncs); `0` means the refresh ran and found nothing new, distinct from not running at all.
 */
export function getCatalogRefreshToast({
  syncFailed,
  catalogFailed,
  isOllamaExternal,
  catalogAdded,
  catalogFailureReason,
}: CatalogRefreshOutcome): CatalogRefreshToast {
  const syncedLabel = isOllamaExternal
    ? "Models synced successfully"
    : "Models refreshed successfully";

  if (syncFailed && catalogFailed) {
    return {
      variant: "error",
      message: "Failed to refresh models and catalog.",
    };
  }
  if (syncFailed) {
    return {
      variant: "error",
      message: `Failed to ${isOllamaExternal ? "sync" : "refresh"} models, but the catalog was refreshed.`,
    };
  }
  if (catalogFailed) {
    return {
      variant: "error",
      message: catalogFailureReason
        ? `${syncedLabel}, but catalog refresh failed: ${catalogFailureReason}`
        : `${syncedLabel}, but catalog refresh failed.`,
    };
  }

  const catalogMsg =
    catalogAdded === null
      ? ""
      : catalogAdded > 0
        ? ` ${catalogAdded} new model${catalogAdded === 1 ? "" : "s"} added to catalog.`
        : " No new models found.";
  return { variant: "success", message: `${syncedLabel}.${catalogMsg}` };
}

export function getRefreshButtonLabel(
  isPending: boolean,
  isOllamaExternal: boolean,
): string {
  if (isPending) {
    return isOllamaExternal ? "Syncing…" : "Refreshing…";
  }
  return isOllamaExternal ? "↺ Sync" : "↺ Refresh";
}
