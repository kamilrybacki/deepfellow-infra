/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { apiClient } from "@/deepfellow/client";
import type { GpuCardStats, GpuStats } from "@/deepfellow/types";
import type { McpOAuthStatusValue, ServiceModel } from "@/deepfellow/types";
import { InstallationWarningsError } from "@/deepfellow/types";
import { MODEL_TYPES } from "@/deepfellow/types";
import { useModal } from "@/hooks/use-modal";
import {
  clearModelInstallProgress,
  getSnapshot,
  setModelInstallProgress,
  useModelInstallProgress,
} from "@/state/install-progress-store";
import {
  getCatalogRefreshToast,
  getRefreshButtonLabel,
  serviceHasCatalogRefresh,
  syncModelsTriggersCatalogRefresh,
} from "@/utils/catalog-refresh";
import {
  renderMarkdownLinks,
  stripMarkdownLinks,
} from "@/utils/markdown-links";
import {
  COMPLETION_SMOOTH_MIN_MS,
  COMPLETION_SMOOTH_MS,
  getStepPerTick,
  startProgressSimulation,
} from "@/utils/progress-simulation";
import type { SimulationHandle } from "@/utils/progress-simulation";
import type { ProgressEvent } from "@/utils/sse-stream";
import { getStageLabel } from "@/utils/sse-stream";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { AlertCircle, ExternalLink, Info, MoreVertical } from "lucide-react";
import {
  type RefObject,
  memo,
  startTransition,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import { AddMcpServerModal } from "./AddMcpServerModal";
import type {
  AddMcpServerPayload,
  AddMcpServerSpec,
  ProxyMcpServerSpec,
} from "./AddMcpServerModal";
import { ConfirmModal } from "./ConfirmModal";
import { ContentModal } from "./ContentModal";
import { DynamicFormModal } from "./DynamicFormModal";

import { ProgressBadge } from "./ProgressBadge";
import { TestResultModal } from "./TestResultModal";
import { UninstallWithPurgeModal } from "./UninstallWithPurgeModal";
import { WarningsModal } from "./WarningsModal";

// Backend surfaces these env var names in error messages when a model download needs
// credentials — point the user at the config page instead of leaving them to guess.
const CONFIG_ENV_VAR_MARKERS = ["DF_HUGGING_FACE_TOKEN", "DF_CIVITAI_TOKEN"];

function needsConfigPageLink(message: string): boolean {
  return CONFIG_ENV_VAR_MARKERS.some((marker) => message.includes(marker));
}

function configPageToastAction(navigate: ReturnType<typeof useNavigate>) {
  return {
    label: "Open Configuration",
    onClick: () => navigate({ to: "/dashboard/config" }),
  };
}

// Auto-install submits install-time options directly (no separate "click Install" dialog to fill
// them in), so without this they'd default to empty - leaving the row's Configuration column blank
// even though the container is actually running with the definition's own envs/headers/prefix.
function buildDuplicateInstallSpec(
  definitionSpec: Record<string, unknown>,
): Record<string, unknown> {
  const installSpec: Record<string, unknown> = {};
  if (
    typeof definitionSpec.default_prefix === "string" &&
    definitionSpec.default_prefix.trim()
  ) {
    installSpec.prefix = definitionSpec.default_prefix;
  }
  if (definitionSpec.envs && typeof definitionSpec.envs === "object") {
    installSpec.envs = definitionSpec.envs;
  }
  if (definitionSpec.headers && typeof definitionSpec.headers === "object") {
    installSpec.headers = definitionSpec.headers;
  }
  return installSpec;
}

interface ServiceModelsProps {
  serviceId: string;
}

export function ServiceModels({ serviceId }: ServiceModelsProps) {
  const modal = useModal();
  const navigate = useNavigate();
  const [filterText, setFilterText] = useState("");
  const deferredFilterText = useDeferredValue(filterText);
  const [filterType, setFilterType] = useState<string>("__all");
  const [filterInstalled, setFilterInstalled] = useState<string>("__all");
  const [filterDownloaded, setFilterDownloaded] = useState<string>("__all");
  const [filterCustom, setFilterCustom] = useState<string>("__all");
  const [addMcpServerOpen, setAddMcpServerOpen] = useState(false);
  const [editMcpServer, setEditMcpServer] = useState<{
    customModelId: string;
    spec: AddMcpServerSpec | ProxyMcpServerSpec;
  } | null>(null);
  const [mcpApiError, setMcpApiError] = useState<string | null>(null);
  const [showEntrySkeleton, setShowEntrySkeleton] = useState(true);
  const [installingModelId, setInstallingModelId] = useState<string | null>(
    null,
  );
  const [editingModelId, setEditingModelId] = useState<string | null>(null);
  const [oauthPromptModelId, setOauthPromptModelId] = useState<string | null>(
    null,
  );
  const pendingInstallationRef = useRef<{
    modelId: string;
    spec: Record<string, unknown>;
    size?: string;
  } | null>(null);
  const lastAddedCustomModelIdRef = useRef<string | null>(null);
  const hasWarningsRef = useRef(false);
  const simulationStopFnsRef = useRef<Record<string, SimulationHandle>>({});
  const hasRealProgressByModelRef = useRef<Record<string, boolean>>({});
  const toastIdsRef = useRef<Record<string, string | number>>({});
  const restartDockerToastIdRef = useRef<string | number | null>(null);
  const testAbortControllerRef = useRef<AbortController | null>(null);
  const installAbortControllersRef = useRef<Record<string, AbortController>>(
    {},
  );
  const cancelledInstallsRef = useRef<Set<string>>(new Set());
  const queryClient = useQueryClient();

  const serviceInfoQuery = useQuery({
    queryKey: ["admin", "services", serviceId],
    queryFn: () => apiClient.getAdminService(serviceId),
    refetchInterval: 10_000,
    refetchIntervalInBackground: false,
  });

  const serviceInfo = serviceInfoQuery.data;

  const isOllamaExternal = serviceInfo?.type === "ollama-external";

  const isCpuOnly = (() => {
    const installed = serviceInfo?.installed;
    if (!installed || typeof installed !== "object") return false;
    const spec = installed as Record<string, unknown>;
    if ("stage" in spec) return false;
    return spec.hardware === "CPU" || spec.hardware === false;
  })();

  const modelsQuery = useQuery({
    queryKey: ["admin", "services", serviceId, "models"],
    queryFn: () => apiClient.listAdminServiceModels(serviceId),
    refetchInterval: isOllamaExternal ? 60_000 : 10_000,
    refetchIntervalInBackground: false,
  });

  const modelsData = modelsQuery.data;
  const existingModelRefs = useMemo(
    () =>
      (modelsData?.list ?? []).map((m) => ({
        id: m.id,
        effective_prefix: m.effective_prefix,
      })),
    [modelsData],
  );

  const { refetch: refetchModels } = modelsQuery;
  const handleRetryLoadModels = useCallback(() => {
    void refetchModels();
  }, [refetchModels]);

  const gpuStatsQuery = useQuery<GpuStats | null>({
    queryKey: ["admin", "settings", "hardware", "gpu-stats"],
    queryFn: () => apiClient.getGpuStats(),
    retry: false,
    refetchInterval: 10_000,
  });

  // Always show skeleton on entry to this page, even if React Query has cached data.
  // This avoids a "blank/lag" feel on subsequent navigations.
  useEffect(() => {
    void serviceId;
    setShowEntrySkeleton(true);
  }, [serviceId]);

  const isEntryBusy =
    modelsQuery.isLoading ||
    modelsQuery.isFetching ||
    serviceInfoQuery.isLoading ||
    serviceInfoQuery.isFetching;

  useEffect(() => {
    if (!showEntrySkeleton) return;
    if (isEntryBusy) return;
    const t = window.setTimeout(() => setShowEntrySkeleton(false), 120);
    return () => window.clearTimeout(t);
  }, [showEntrySkeleton, isEntryBusy]);

  // Clean up any running simulations on unmount
  useEffect(() => {
    return () => {
      for (const sim of Object.values(simulationStopFnsRef.current)) sim.stop();
    };
  }, []);

  // Progress polling for existing installations
  useEffect(() => {
    if (!modelsData?.list) return;

    // Reconcile: clear progress for models that are no longer in-progress.
    const inProgressKeys = new Set<string>();
    for (const model of modelsData.list) {
      const inst = model.installed;
      if (!inst || typeof inst !== "object") continue;
      const s = (inst as { stage?: unknown }).stage;
      const v = (inst as { value?: unknown }).value;
      if (
        (s === "install" || s === "download") &&
        typeof v === "number" &&
        Number.isFinite(v)
      ) {
        inProgressKeys.add(`${serviceId}::${model.id}`);
      }
    }
    const prefix = `${serviceId}::`;
    for (const key of Object.keys(getSnapshot().models)) {
      if (key.startsWith(prefix) && !inProgressKeys.has(key)) {
        clearModelInstallProgress(serviceId, key.slice(prefix.length));
      }
    }

    const cleanups: Array<() => void> = [];

    for (const model of modelsData.list) {
      const installed = model.installed;
      if (!installed || typeof installed !== "object") continue;

      const installedStage = (installed as { stage?: unknown }).stage;
      const installedValue = (installed as { value?: unknown }).value;
      const installedIsProgress =
        (installedStage === "install" || installedStage === "download") &&
        typeof installedValue === "number" &&
        Number.isFinite(installedValue);

      if (!installedIsProgress) continue;

      const modelId = model.id;
      const simKey = `${serviceId}::${modelId}`;
      // Skip if already tracked by an active install mutation.
      if (simulationStopFnsRef.current[simKey]) continue;

      let currentStage: "install" | "download" = installedStage as
        | "install"
        | "download";
      const abortController = new AbortController();
      let isCancelled = false;

      // Use existing store value or backend value as starting point.
      const existingProgress = getSnapshot().models[simKey];
      const startValue = existingProgress?.value ?? (installedValue as number);
      if (!existingProgress) {
        setModelInstallProgress(serviceId, modelId, {
          stage: currentStage,
          value: installedValue as number,
        });
      }

      const installStartTime = Date.now();
      const sim = startProgressSimulation({
        stepPerTick: getStepPerTick(
          model.size,
          serviceId === "vllm" ? 0.2 : 1.6,
        ),
        initialValue: startValue,
        onTick: (value) => {
          if (isCancelled) return;
          setModelInstallProgress(serviceId, modelId, {
            stage: currentStage,
            value,
          });
        },
      });
      simulationStopFnsRef.current[simKey] = sim;

      apiClient
        .getModelProgress(
          serviceId,
          modelId,
          (event: ProgressEvent) => {
            if (isCancelled) return;

            const stage = event.stage;
            const value = event.value;

            if (event.type === "progress" && stage && value !== undefined) {
              currentStage = stage;
              hasRealProgressByModelRef.current[modelId] = true;
              setModelInstallProgress(serviceId, modelId, { stage, value });
              return;
            }

            if (event.type === "finish") {
              if (event.status === "ok") {
                sim.smoothComplete(
                  Math.max(
                    COMPLETION_SMOOTH_MIN_MS,
                    Math.min(
                      Date.now() - installStartTime,
                      COMPLETION_SMOOTH_MS,
                    ),
                  ),
                  () => {
                    delete simulationStopFnsRef.current[simKey];
                    queryClient.invalidateQueries({
                      queryKey: ["admin", "services", serviceId, "models"],
                    });
                  },
                  getSnapshot().models[simKey]?.value,
                );
              } else {
                sim.stop();
                delete simulationStopFnsRef.current[simKey];
                clearModelInstallProgress(serviceId, modelId);
                const message = `Installation failed for ${modelId}: ${event.details || "Unknown error"}`;
                toast.error(renderMarkdownLinks(message), {
                  action: needsConfigPageLink(message)
                    ? configPageToastAction(navigate)
                    : undefined,
                });
              }
            }
          },
          abortController.signal,
        )
        .catch((error) => {
          if (isCancelled) return;
          if (error instanceof Error && error.name === "AbortError") return;
          if (!isCancelled) {
            console.error(`Error polling progress for ${modelId}:`, error);
          }
        });

      cleanups.push(() => {
        isCancelled = true;
        abortController.abort();
        sim.stop();
        delete simulationStopFnsRef.current[simKey];
        delete hasRealProgressByModelRef.current[modelId];
        // Progress is NOT cleared here — reconciliation at the top handles cleanup.
      });
    }

    return () => {
      for (const cleanup of cleanups) cleanup();
    };
  }, [modelsData, serviceId, queryClient, navigate]);

  // Cancel an in-progress install: stop the backend Docker pull, abort the local SSE read, and reset the UI.
  const handleCancelInstall = useCallback(
    async (modelId: string) => {
      const simKey = `${serviceId}::${modelId}`;
      // Flag the cancel so the install mutation's error path swallows the AbortError it triggers.
      cancelledInstallsRef.current.add(modelId);

      // Abort the local SSE fetch (terminates the pending install promise immediately).
      installAbortControllersRef.current[modelId]?.abort();
      delete installAbortControllersRef.current[modelId];

      // Stop the progress simulation and clear UI progress.
      simulationStopFnsRef.current[simKey]?.stop();
      delete simulationStopFnsRef.current[simKey];
      delete hasRealProgressByModelRef.current[modelId];
      clearModelInstallProgress(serviceId, modelId);

      // Dismiss the long-lived loading progress toast and show a FRESH cancellation toast.
      const toastId = toastIdsRef.current[modelId];
      if (toastId) {
        toast.dismiss(toastId);
        delete toastIdsRef.current[modelId];
      }
      toast.success(`Installation cancelled for ${modelId}`, {
        duration: 8000,
      });

      setInstallingModelId(null);

      try {
        // Stop the actual Docker image pull on the backend.
        await apiClient.cancelAdminServiceModelInstall(serviceId, modelId);
      } catch (error) {
        // A 404 just means the install already finished/cleared on the backend — safe to ignore.
        const isAlreadyGone =
          error instanceof Error && error.message.includes("HTTP 404");
        if (!isAlreadyGone) {
          toast.error(
            `Failed to cancel installation for ${modelId} — please try again`,
          );
        }
        console.error(`Failed to cancel install for ${modelId}:`, error);
      }

      // Let reconciliation confirm the reset.
      queryClient.invalidateQueries({
        queryKey: ["admin", "services", serviceId, "models"],
      });
    },
    [serviceId, queryClient],
  );

  // Shared sonner options for an installing model's progress toast, including the Cancel action.
  const progressToastOptions = useCallback(
    (modelId: string, toastId: string | number) => ({
      id: toastId,
      action: { label: "Cancel", onClick: () => handleCancelInstall(modelId) },
    }),
    [handleCancelInstall],
  );

  // Render the progress toast from the store, so its % matches the table row.
  const syncProgressToast = useCallback(
    (modelId: string) => {
      const toastId = toastIdsRef.current[modelId];
      if (!toastId) return;
      const progress = getSnapshot().models[`${serviceId}::${modelId}`];
      if (!progress) return;
      toast.loading(
        `${getStageLabel(progress.stage)} ${modelId}: ${(progress.value * 100).toFixed(1)}%`,
        progressToastOptions(modelId, toastId),
      );
    },
    [serviceId, progressToastOptions],
  );

  const installMutation = useMutation({
    mutationFn: async ({
      modelId,
      spec,
      size,
      ignoreWarnings = false,
    }: {
      modelId: string;
      spec: Record<string, unknown>;
      size?: string;
      ignoreWarnings?: boolean;
    }) => {
      const abortController = new AbortController();
      installAbortControllersRef.current[modelId] = abortController;
      try {
        return await new Promise<void>((resolve, reject) => {
          let currentStage: "install" | "download" = "download";
          const simKey = `${serviceId}::${modelId}`;
          const installStartTime = Date.now();
          const sim = startProgressSimulation({
            stepPerTick: getStepPerTick(size ?? ""),
            onTick: (value) => {
              setModelInstallProgress(serviceId, modelId, {
                stage: currentStage,
                value,
              });
              syncProgressToast(modelId);
            },
          });
          simulationStopFnsRef.current[simKey] = sim;

          apiClient
            .installAdminServiceModelStreaming(
              serviceId,
              modelId,
              spec,
              (event: ProgressEvent) => {
                const stage = event.stage;
                const value = event.value;

                if (event.type === "progress" && stage && value !== undefined) {
                  currentStage = stage;
                  hasRealProgressByModelRef.current[modelId] = true;
                  setModelInstallProgress(serviceId, modelId, { stage, value });
                  syncProgressToast(modelId);
                } else if (event.type === "finish") {
                  if (event.status === "ok") {
                    // On success `event.details` is the backend's full install-result
                    // object (not a string) — see `ProgressEvent`.
                    const installResult =
                      typeof event.details === "object"
                        ? event.details
                        : undefined;
                    sim.smoothComplete(
                      Math.max(
                        COMPLETION_SMOOTH_MIN_MS,
                        Math.min(
                          Date.now() - installStartTime,
                          COMPLETION_SMOOTH_MS,
                        ),
                      ),
                      () => {
                        delete simulationStopFnsRef.current[simKey];
                        // Do NOT clear progress here — reconciliation clears it once the
                        // backend confirms the model is no longer in-progress. Clearing early
                        // would cause the bar to jump back to the stale backend value.
                        const toastId = toastIdsRef.current[modelId];
                        if (toastId) {
                          // action: undefined removes the Cancel button — the install is done.
                          toast.success(
                            `Model ${modelId} installed successfully`,
                            { id: toastId, action: undefined },
                          );
                          delete toastIdsRef.current[modelId];
                        }
                        if (
                          serviceId === "mcp" &&
                          installResult?.requires_oauth
                        ) {
                          setOauthPromptModelId(modelId);
                        }
                        resolve();
                      },
                      getSnapshot().models[simKey]?.value,
                    );
                  } else {
                    sim.stop();
                    delete simulationStopFnsRef.current[simKey];
                    // Failure events always carry a plain string (see `ProgressEvent`).
                    reject(
                      new Error(
                        typeof event.details === "string"
                          ? event.details
                          : "Installation failed",
                      ),
                    );
                  }
                }
              },
              ignoreWarnings,
              abortController.signal,
            )
            .catch(reject);
        });
      } catch (e) {
        const simKey = `${serviceId}::${modelId}`;
        simulationStopFnsRef.current[simKey]?.stop();
        delete simulationStopFnsRef.current[simKey];

        // Intentional cancel: the UI was already reset by handleCancelInstall — swallow the AbortError.
        if (cancelledInstallsRef.current.has(modelId)) {
          cancelledInstallsRef.current.delete(modelId);
          return;
        }

        if (e instanceof InstallationWarningsError) {
          throw e;
        }

        clearModelInstallProgress(serviceId, modelId);
        const toastId = toastIdsRef.current[modelId];
        if (toastId) {
          const message = `Failed to install model ${modelId}: ${(e instanceof Error ? e.message : "") || "Installation failed"}`;
          // action: undefined removes the Cancel button — there is nothing left to cancel,
          // unless the error itself points at a config value the user should go fix.
          toast.error(renderMarkdownLinks(message), {
            id: toastId,
            action: needsConfigPageLink(message)
              ? configPageToastAction(navigate)
              : undefined,
          });
          delete toastIdsRef.current[modelId];
        }
      } finally {
        delete installAbortControllersRef.current[modelId];
      }
    },
    onMutate: ({ modelId }) => {
      setInstallingModelId(modelId);
    },
    onSuccess: (_data, variables) => {
      const simKey = `${serviceId}::${variables.modelId}`;
      delete simulationStopFnsRef.current[simKey];
      delete hasRealProgressByModelRef.current[variables.modelId];
      queryClient.invalidateQueries({
        queryKey: ["admin", "services", serviceId, "models"],
      });
      pendingInstallationRef.current = null;
    },
    onError: (error, variables) => {
      const simKey = `${serviceId}::${variables.modelId}`;
      const simStop = simulationStopFnsRef.current[simKey];
      if (simStop) {
        simStop.stop();
        delete simulationStopFnsRef.current[simKey];
      }
      delete hasRealProgressByModelRef.current[variables.modelId];

      if (error instanceof InstallationWarningsError) {
        // Show warnings modal instead of error toast
        hasWarningsRef.current = true;

        const toastId = toastIdsRef.current[variables.modelId];
        if (toastId) {
          toast.loading(
            `Warnings for ${variables.modelId}: awaiting confirmation...`,
            { id: toastId },
          );
        }

        modal.open(WarningsModal, {
          warnings: error.warnings,
          onContinue: handleWarningsContinue,
          isLoading: false,
        });
        return;
      }

      hasWarningsRef.current = false;
      const modelId = variables.modelId;
      // Finish toast is handled in the SSE callback. If we failed before streaming starts,
      // fall back to a plain error toast.
      if (!toastIdsRef.current[modelId]) {
        const message = `Failed to install model: ${error.message}`;
        toast.error(renderMarkdownLinks(message), {
          action: needsConfigPageLink(message)
            ? configPageToastAction(navigate)
            : undefined,
        });
      }
      clearModelInstallProgress(serviceId, modelId);
    },
    onSettled: (_data, _error, variables) => {
      cancelledInstallsRef.current.delete(variables.modelId);
      delete installAbortControllersRef.current[variables.modelId];
      // Only reset if not showing warnings modal
      if (!hasWarningsRef.current) {
        setInstallingModelId(null);
      }
    },
  });

  const startOauthAfterInstallMutation = useMutation({
    mutationFn: (modelId: string) =>
      apiClient.startMcpOAuth(serviceId, modelId),
    onSuccess: (data, modelId) => {
      // Triggered by a direct click on the modal's "Authorize" button, so this is a
      // same-gesture window.open — browsers won't treat it as an unsolicited popup.
      window.open(data.authorize_url, "_blank", "noopener,noreferrer");
      queryClient.invalidateQueries({
        queryKey: [
          "admin",
          "services",
          serviceId,
          "models",
          modelId,
          "oauth-status",
        ],
      });
      setOauthPromptModelId(null);
    },
    onError: (error) => {
      toast.error(`Failed to start authorization: ${error.message}`);
    },
  });

  const uninstallMutation = useMutation({
    mutationFn: (modelId: string) =>
      apiClient.uninstallAdminServiceModel(serviceId, modelId, false),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["admin", "services", serviceId, "models"],
      });
      modal.close();
      toast.success("Model uninstalled successfully");
    },
    onError: (error) => {
      toast.error(`Failed to uninstall model: ${error.message}`);
    },
  });

  const purgeMutation = useMutation({
    mutationFn: (modelId: string) =>
      apiClient.uninstallAdminServiceModel(serviceId, modelId, true),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["admin", "services", serviceId, "models"],
      });
      modal.close();
      toast.success("Model purged successfully");
    },
    onError: (error) => {
      toast.error(`Failed to purge model: ${error.message}`);
    },
  });

  const testMutation = useMutation({
    mutationFn: async (registrationId: string) => {
      // Create a new AbortController for this test run
      const abortController = new AbortController();
      testAbortControllerRef.current = abortController;
      return apiClient.testModel(registrationId, abortController.signal);
    },
    onMutate: () => {
      // Open modal immediately with loading state
      modal.open(TestResultModal, {
        result: {},
        isLoading: true,
        onCancel: () => {
          testAbortControllerRef.current?.abort();
          testMutation.reset();
          testAbortControllerRef.current = null;
          modal.close();
        },
      });
    },
    onSuccess: (result) => {
      // Clear the abort controller ref
      testAbortControllerRef.current = null;
      queryClient.invalidateQueries({
        queryKey: ["admin", "services", serviceId, "models"],
      });
      // Update modal with actual result
      modal.open(TestResultModal, {
        result,
        isLoading: false,
        onCancel: () => {
          testMutation.reset();
          modal.close();
        },
      });
    },
    onError: (error) => {
      // onCancel already handled all cleanup for user-initiated cancels.
      // Don't interfere with any subsequent test that may have started since.
      if (error instanceof Error && error.name === "AbortError") {
        return;
      }
      // Clear the abort controller ref for genuine errors
      testAbortControllerRef.current = null;
      queryClient.invalidateQueries({
        queryKey: ["admin", "services", serviceId, "models"],
      });
      // Show error state in modal
      modal.open(TestResultModal, {
        result: {
          error: true,
          details: { message: error.message },
        },
        isLoading: false,
        onCancel: () => {
          testMutation.reset();
          modal.close();
        },
      });
    },
  });

  const addCustomModelMutation = useMutation({
    mutationFn: async (spec: Record<string, unknown>) => {
      return apiClient.addCustomModel(serviceId, spec);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["admin", "services", serviceId, "models"],
      });
      modal.close();
      toast.success("Custom model added successfully");

      const addedId = lastAddedCustomModelIdRef.current;
      if (addedId) {
        setFilterText(addedId);
        setTimeout(() => {
          const input = document.getElementById(
            "search-models",
          ) as HTMLInputElement | null;
          input?.focus();
        }, 0);
      }
    },
    onError: (error) => {
      toast.error(`Failed to add custom model: ${error.message}`);
    },
  });

  const editCustomModelMutation = useMutation({
    mutationFn: async ({
      customModelId,
      spec,
    }: {
      customModelId: string;
      spec: Record<string, unknown>;
    }) => {
      return apiClient.editCustomModel(serviceId, customModelId, spec);
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({
        queryKey: ["admin", "services", serviceId, "models"],
      });
      setEditingModelId(null);
      toast.success(
        result.reinstalled
          ? "Settings saved. The service was reinstalled with the new settings."
          : "Settings saved.",
      );
    },
    onError: (error) => {
      setEditingModelId(null);
      toast.error(`Failed to save settings: ${error.message}`);
    },
  });

  const editModelInstallOptionsMutation = useMutation({
    mutationFn: async ({
      modelId,
      spec,
    }: {
      modelId: string;
      spec: Record<string, unknown>;
    }) => {
      return apiClient.editModelInstallOptions(serviceId, modelId, spec);
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({
        queryKey: ["admin", "services", serviceId, "models"],
      });
      setEditingModelId(null);
      toast.success(
        result.reinstalled
          ? "Settings saved. The service was reinstalled with the new settings."
          : "Settings saved.",
      );
    },
    onError: (error) => {
      setEditingModelId(null);
      toast.error(`Failed to save settings: ${error.message}`);
    },
  });

  const removeCustomModelMutation = useMutation({
    mutationFn: async ({ customModelId }: { customModelId: string }) => {
      return apiClient.removeCustomModel(serviceId, customModelId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["admin", "services", serviceId, "models"],
      });
      modal.close();
      toast.success("Custom model removed successfully");
    },
    onError: (error) => {
      // Check if this is a validation error about the model being in use
      const errorMessage = error.message || "";
      if (
        errorMessage.includes("Cannot remove custom model") ||
        errorMessage.includes("it is in use")
      ) {
        toast.error(
          "Cannot remove custom model: it is currently installed. Please uninstall it first.",
        );
      } else {
        toast.error(`Failed to remove custom model: ${errorMessage}`);
      }
    },
  });

  const addMcpServerMutation = useMutation({
    mutationFn: async (payload: AddMcpServerPayload) => {
      const spec =
        payload.kind === "docker"
          ? {
              ...payload.data,
              ...(payload.repository_url
                ? { repository_url: payload.repository_url }
                : {}),
              ...(payload.description
                ? { description: payload.description }
                : {}),
            }
          : (payload as unknown as Record<string, unknown>);
      return apiClient.addCustomModel(serviceId, spec);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["admin", "services", serviceId, "models"],
      });
      setAddMcpServerOpen(false);
      setMcpApiError(null);
      toast.success("MCP server added successfully");
    },
    onError: (error) => {
      setMcpApiError(error.message || "Failed to add MCP server");
      toast.error(error.message || "Failed to add MCP server");
    },
  });

  const editMcpServerMutation = useMutation({
    mutationFn: async ({
      customModelId,
      spec,
    }: {
      customModelId: string;
      spec: AddMcpServerSpec | ProxyMcpServerSpec;
    }) => {
      return apiClient.editCustomModel(
        serviceId,
        customModelId,
        spec as unknown as Record<string, unknown>,
      );
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({
        queryKey: ["admin", "services", serviceId, "models"],
      });
      setEditingModelId(null);
      toast.success(
        result.reinstalled
          ? "MCP server updated. It was reinstalled with the new settings."
          : "MCP server updated.",
      );
    },
    onError: (error) => {
      setEditingModelId(null);
      toast.error(error.message || "Failed to update MCP server");
    },
  });

  const hasCatalogRefresh = serviceHasCatalogRefresh(serviceInfo?.type);
  const syncIsCatalogRefresh = syncModelsTriggersCatalogRefresh(
    serviceInfo?.type,
  );
  const catalogRefreshUnavailableReason =
    serviceInfo?.catalog_refresh_unavailable_reason ?? null;

  const refreshMutation = useMutation({
    mutationFn: async () => {
      // vLLM/llama.cpp/SGLang's sync_models already delegates to the same background refresh as
      // refreshCatalog - calling both there would race two concurrent triggers against the
      // shared HuggingFace in-flight guard, so for those the sync call is skipped. Every other
      // service type keeps calling both: sync_models is either a real sync (ollama-external) or
      // a no-op, and never touches the catalog.
      const [syncResult, catalogResult] = await Promise.allSettled([
        syncIsCatalogRefresh
          ? Promise.resolve(null)
          : apiClient.syncModels(serviceId),
        hasCatalogRefresh
          ? apiClient.refreshCatalog(serviceId, true)
          : Promise.resolve(null),
      ]);
      return { syncResult, catalogResult };
    },
    onSuccess: ({ syncResult, catalogResult }) => {
      queryClient.invalidateQueries({
        queryKey: ["admin", "services", serviceId, "models"],
      });

      const toastMsg = getCatalogRefreshToast({
        syncFailed: syncResult.status === "rejected",
        catalogFailed: catalogResult.status === "rejected",
        isOllamaExternal,
        catalogAdded:
          catalogResult.status === "fulfilled" && catalogResult.value
            ? catalogResult.value.added
            : null,
        catalogFailureReason:
          catalogResult.status === "rejected" &&
          catalogResult.reason instanceof Error
            ? catalogResult.reason.message
            : undefined,
      });
      toast[toastMsg.variant](toastMsg.message);
    },
  });

  const dockerRestartMutation = useMutation({
    mutationFn: (modelId: string) =>
      apiClient.restartDocker(serviceId, modelId),
    onMutate: () => {
      const toastId = toast.loading("Restarting Docker...");
      restartDockerToastIdRef.current = toastId;
    },
    onSuccess: () => {
      if (restartDockerToastIdRef.current) {
        toast.success("Docker restarted successfully", {
          id: restartDockerToastIdRef.current,
        });
        restartDockerToastIdRef.current = null;
      } else {
        toast.success("Docker restarted successfully");
      }
    },
    onError: (error) => {
      if (restartDockerToastIdRef.current) {
        toast.error(`Failed to restart Docker: ${error.message}`, {
          id: restartDockerToastIdRef.current,
        });
        restartDockerToastIdRef.current = null;
      } else {
        toast.error(`Failed to restart Docker: ${error.message}`);
      }
    },
  });

  const handleShowDockerLogs = useCallback(
    async (modelId: string) => {
      // Open modal immediately with loading state
      modal.open(ContentModal, {
        title: "Docker Logs",
        content: "",
        wide: true,
        pre: true,
        isLoading: true,
        onCancel: () => {
          modal.close();
        },
      });

      try {
        const data = await apiClient.getDockerLogs(serviceId, modelId);
        // Update modal with actual content
        modal.open(ContentModal, {
          title: "Docker Logs",
          content: data.logs,
          wide: true,
          pre: true,
          isLoading: false,
        });
      } catch (error) {
        modal.close();
        toast.error(
          `Failed to fetch Docker logs: ${error instanceof Error ? error.message : "Unknown error"}`,
        );
      }
    },
    [modal, serviceId],
  );

  const handleShowDockerCompose = useCallback(
    async (modelId: string) => {
      // Open modal immediately with loading state
      modal.open(ContentModal, {
        title: "Docker Compose File",
        content: "",
        wide: true,
        pre: true,
        isLoading: true,
        onCancel: () => {
          modal.close();
        },
      });

      try {
        const data = await apiClient.getDockerCompose(serviceId, modelId);
        // Update modal with actual content
        modal.open(ContentModal, {
          title: "Docker Compose File",
          content: data.compose_file,
          wide: true,
          pre: true,
          isLoading: false,
        });
      } catch (error) {
        modal.close();
        toast.error(
          `Failed to fetch Docker compose file: ${error instanceof Error ? error.message : "Unknown error"}`,
        );
      }
    },
    [modal, serviceId],
  );

  const handleRestartDocker = useCallback(
    (modelId: string) => {
      modal.open(ConfirmModal, {
        title: "Restart Docker",
        description: `Are you sure you want to restart Docker for model ${modelId}?`,
        confirmText: "Restart",
        cancelText: "Cancel",
        onConfirm: () => {
          modal.close();
          dockerRestartMutation.mutate(modelId);
        },
        isLoading: dockerRestartMutation.isPending,
        variant: "warning",
      });
    },
    [modal, dockerRestartMutation.isPending, dockerRestartMutation.mutate],
  );

  const handleInstallClick = useCallback(
    async (model: ServiceModel) => {
      modal.open(DynamicFormModal, {
        title: `Install ${model.id}`,
        fields: [],
        isLoading: true,
        isSubmitting: false,
        onSubmit: () => {},
      });

      try {
        const modelDetail = await apiClient.getAdminServiceModel(
          serviceId,
          model.id,
        );
        modal.open(DynamicFormModal, {
          title: `Install ${modelDetail.id}`,
          fields: modelDetail.spec.fields,
          serviceId,
          modelId: modelDetail.id,
          onSubmit: (spec: Record<string, unknown>) => {
            const cleanedSpec = Object.fromEntries(
              Object.entries(spec).filter(
                ([_, value]) => value !== null && value !== undefined,
              ),
            ) as Record<string, unknown>;
            pendingInstallationRef.current = {
              modelId: modelDetail.id,
              spec: cleanedSpec,
              size: modelDetail.size,
            };
            modal.close();
            const toastId: string | number = toast.loading(
              `Starting installation for ${modelDetail.id}...`,
            );
            toastIdsRef.current[modelDetail.id] = toastId;
            installMutation.mutate({
              modelId: modelDetail.id,
              spec: cleanedSpec,
              size: modelDetail.size,
            });
          },
          // Keep modal interactive; don't disable because another install is running.
          isSubmitting: false,
        });
      } catch {
        modal.close();
        toast.error("Failed to load model details");
      }
    },
    [modal, serviceId, installMutation.mutate],
  );

  // A duplicated model installs itself immediately, using the definition's own default prefix/envs -
  // no separate "click Install" step, unlike a freshly hand-added custom model.
  const installNewlyDuplicatedModel = useCallback(
    (newModelId: string, size?: string, spec: Record<string, unknown> = {}) => {
      // Set the same way handleInstallClick does - if this install trips docker-image warnings,
      // the WarningsModal's "Continue" button (handleWarningsContinue) reads this ref to know which
      // install to retry with ignoreWarnings. Without it, "Continue" would either no-op or retry a
      // stale install left over from an earlier attempt in the same session.
      pendingInstallationRef.current = { modelId: newModelId, spec, size };
      const toastId = toast.loading(
        `Starting installation for ${newModelId}...`,
      );
      toastIdsRef.current[newModelId] = toastId;
      installMutation.mutate({ modelId: newModelId, spec, size });
    },
    [installMutation.mutate],
  );

  const handleEditModelOptionsClick = useCallback(
    async (model: ServiceModel) => {
      modal.open(DynamicFormModal, {
        title: `Edit settings for ${model.id}`,
        fields: [],
        isLoading: true,
        isSubmitting: false,
        onSubmit: () => {},
      });

      try {
        const modelDetail = await apiClient.getAdminServiceModel(
          serviceId,
          model.id,
        );
        const installedInfo =
          model.installed && typeof model.installed === "object"
            ? model.installed
            : null;

        // Catalog models (no `model.custom`) have no stored definition to duplicate from - fetch one
        // synthesized from the live model, so "duplicate" can offer the full add-model field set
        // (image, command, etc.), not just the narrower install-options fields shown for editing.
        let duplicateSpec: Record<string, unknown> | null = null;
        if (serviceInfo?.custom_model_spec) {
          try {
            const result = await apiClient.getDuplicateSpec(
              serviceId,
              model.id,
            );
            duplicateSpec = result.spec;
          } catch (error) {
            // Duplicate just won't be offered for this model; editing its install options still works.
            // Still log it - a silent failure here also silently disables the "skip reinstall when
            // nothing changed" optimization (DynamicFormModal's isUnchanged requires onDuplicate to be
            // present), so a transient error would otherwise go completely unnoticed.
            console.error(
              `Failed to prepare a duplicate spec for ${model.id}:`,
              error,
            );
          }
        }

        modal.open(DynamicFormModal, {
          title: `Edit settings for ${modelDetail.id}`,
          duplicateTitle: `Duplicate ${modelDetail.id}`,
          fields: modelDetail.spec.fields,
          initialData: (installedInfo?.spec ?? {}) as Record<string, unknown>,
          selfModelId: modelDetail.id,
          serviceId,
          modelId: modelDetail.id,
          deferRender: true,
          submitLabel: "Save",
          onSubmit: (spec: Record<string, unknown>) => {
            const cleanedSpec = Object.fromEntries(
              Object.entries(spec).filter(
                ([_, value]) => value !== null && value !== undefined,
              ),
            ) as Record<string, unknown>;
            modal.close();
            setEditingModelId(modelDetail.id);
            editModelInstallOptionsMutation.mutate({
              modelId: modelDetail.id,
              spec: cleanedSpec,
            });
          },
          ...(duplicateSpec && serviceInfo?.custom_model_spec
            ? {
                duplicateFields: serviceInfo.custom_model_spec.fields,
                duplicateInitialData: duplicateSpec,
                onDuplicate: (spec: Record<string, unknown>) => {
                  const cleanedSpec = Object.fromEntries(
                    Object.entries(spec).filter(
                      ([_, value]) => value !== null && value !== undefined,
                    ),
                  ) as Record<string, unknown>;
                  const maybeId = cleanedSpec.id;
                  const newModelId =
                    typeof maybeId === "string" && maybeId.trim()
                      ? maybeId.trim()
                      : null;
                  // No search-and-focus after duplicating: the new model installs itself
                  // immediately below, so there's no need to jump to it in the list.
                  lastAddedCustomModelIdRef.current = null;
                  addCustomModelMutation.mutate(cleanedSpec, {
                    onSuccess: () => {
                      if (newModelId) {
                        installNewlyDuplicatedModel(
                          newModelId,
                          typeof cleanedSpec.size === "string"
                            ? cleanedSpec.size
                            : undefined,
                          buildDuplicateInstallSpec(cleanedSpec),
                        );
                      }
                    },
                  });
                },
              }
            : {}),
          existingModels: existingModelRefs,
          isSubmitting: addCustomModelMutation.isPending,
        });
      } catch {
        modal.close();
        toast.error("Failed to load model details");
      }
    },
    [
      modal,
      serviceId,
      serviceInfo,
      existingModelRefs,
      editModelInstallOptionsMutation.mutate,
      addCustomModelMutation.mutate,
      addCustomModelMutation.isPending,
      installNewlyDuplicatedModel,
    ],
  );

  const handleWarningsContinue = useCallback(() => {
    if (pendingInstallationRef.current) {
      hasWarningsRef.current = false;
      installMutation.mutate({
        modelId: pendingInstallationRef.current.modelId,
        spec: pendingInstallationRef.current.spec,
        size: pendingInstallationRef.current.size,
        ignoreWarnings: true,
      });
    }
  }, [installMutation.mutate]);

  const handleUninstallClick = useCallback(
    (modelId: string) => {
      modal.open(UninstallWithPurgeModal, {
        title: "Uninstall Model",
        description: `Are you sure you want to uninstall ${modelId}? This action cannot be undone.`,
        confirmText: "Uninstall",
        cancelText: "Cancel",
        purgeLabel: "Purge",
        purgeDescription:
          "Also remove downloaded model files and local data. This cannot be undone.",
        onConfirm: (purge) => {
          modal.close();
          if (purge) {
            purgeMutation.mutate(modelId);
          } else {
            uninstallMutation.mutate(modelId);
          }
        },
        isLoading: uninstallMutation.isPending || purgeMutation.isPending,
        variant: "destructive",
      });
    },
    [
      modal,
      uninstallMutation.isPending,
      purgeMutation.isPending,
      uninstallMutation.mutate,
      purgeMutation.mutate,
    ],
  );

  const handlePurgeClick = useCallback(
    (modelId: string) => {
      modal.open(ConfirmModal, {
        title: "Purge Model",
        description:
          "This will remove all downloaded files for the model. This action cannot be undone.",
        confirmText: "Purge",
        cancelText: "Cancel",
        onConfirm: () => {
          modal.close();
          purgeMutation.mutate(modelId);
        },
        isLoading: purgeMutation.isPending,
        variant: "destructive",
      });
    },
    [modal, purgeMutation.isPending, purgeMutation.mutate],
  );

  const handleTestClick = useCallback(
    (model: ServiceModel) => {
      const installedInfo =
        model.installed && typeof model.installed === "object"
          ? model.installed
          : null;

      if (!installedInfo?.registration_id) {
        toast.error("Model registration ID not found. Cannot test model.");
        return;
      }
      testMutation.mutate(installedInfo.registration_id);
    },
    [testMutation.mutate],
  );

  const handleAddCustomModel = () => {
    if (!serviceInfo?.custom_model_spec) return;
    const fields = serviceInfo.custom_model_spec.fields;

    modal.open(DynamicFormModal, {
      title: "Add custom model",
      fields,
      serviceId,
      deferRender: true,
      submitLabel: "Add",
      submittingLabel: "Adding...",
      onSubmit: (spec: Record<string, unknown>) => {
        const cleanedSpec = Object.fromEntries(
          Object.entries(spec).filter(
            ([_, value]) => value !== null && value !== undefined,
          ),
        ) as Record<string, unknown>;

        const maybeId = cleanedSpec.id;
        lastAddedCustomModelIdRef.current =
          typeof maybeId === "string" && maybeId.trim() ? maybeId.trim() : null;
        addCustomModelMutation.mutate(cleanedSpec);
      },
      isSubmitting: addCustomModelMutation.isPending,
    });
  };

  const handleRemoveCustomModelClick = useCallback(
    (model: ServiceModel) => {
      const customModelId = model.custom;
      if (!customModelId) {
        toast.error("Model is not a custom model.");
        return;
      }
      if (model.installed) {
        toast.error(
          "Cannot remove custom model: it is currently installed. Please uninstall it first.",
        );
        return;
      }
      modal.open(ConfirmModal, {
        title: "Remove Custom Model",
        description: `Are you sure you want to remove the custom model ${model.id}? Only uninstalled custom models can be removed. This action cannot be undone.`,
        confirmText: "Remove",
        cancelText: "Cancel",
        onConfirm: () => removeCustomModelMutation.mutate({ customModelId }),
        isLoading: removeCustomModelMutation.isPending,
        variant: "destructive",
      });
    },
    [
      modal,
      removeCustomModelMutation.isPending,
      removeCustomModelMutation.mutate,
    ],
  );

  const handleEditMcpServerClick = useCallback(
    (model: ServiceModel) => {
      const customModelId = model.custom;
      if (!customModelId) return;
      setMcpApiError(null);
      const rawSpec = model.custom_spec;
      const kind =
        rawSpec?.kind === "proxy"
          ? "proxy"
          : rawSpec?.kind === "user"
            ? "user"
            : "docker";
      if (kind === "docker") {
        // A plain Docker-image MCP server (no stdio command, no remote URL) - e.g. a duplicated
        // catalog model like `open-websearch` once it's persisted as its own CustomModel. It has
        // no `command`/`server_url` of its own, so AddMcpServerModal's stdio/proxy shape doesn't
        // fit it (its edit form never shows Docker fields); edit it the same way a "custom"
        // service model is edited instead - the full add-model field set, prefilled from its own
        // stored definition.
        if (!serviceInfo?.custom_model_spec) return;
        modal.open(DynamicFormModal, {
          title: `Edit MCP server ${model.id}`,
          duplicateTitle: `Duplicate ${model.id}`,
          fields: serviceInfo.custom_model_spec.fields,
          initialData: (rawSpec ?? {}) as Record<string, unknown>,
          selfModelId: model.id,
          deferRender: true,
          submitLabel: "Save",
          onSubmit: (spec: Record<string, unknown>) => {
            const cleanedSpec = Object.fromEntries(
              Object.entries(spec).filter(
                ([_, value]) => value !== null && value !== undefined,
              ),
            ) as Record<string, unknown>;
            modal.close();
            setEditingModelId(model.id);
            editCustomModelMutation.mutate({
              customModelId,
              spec: cleanedSpec,
            });
          },
          onDuplicate: (spec: Record<string, unknown>) => {
            const cleanedSpec = Object.fromEntries(
              Object.entries(spec).filter(
                ([_, value]) => value !== null && value !== undefined,
              ),
            ) as Record<string, unknown>;
            const maybeId = cleanedSpec.id;
            const newModelId =
              typeof maybeId === "string" && maybeId.trim()
                ? maybeId.trim()
                : null;
            lastAddedCustomModelIdRef.current = null;
            addCustomModelMutation.mutate(cleanedSpec, {
              onSuccess: () => {
                if (newModelId) {
                  installNewlyDuplicatedModel(
                    newModelId,
                    typeof cleanedSpec.size === "string"
                      ? cleanedSpec.size
                      : undefined,
                    buildDuplicateInstallSpec(cleanedSpec),
                  );
                }
              },
            });
          },
          existingModels: existingModelRefs,
          isSubmitting: addCustomModelMutation.isPending,
        });
        return;
      }
      if (kind === "proxy") {
        setEditMcpServer({
          customModelId,
          spec: {
            kind: "proxy",
            id: model.id,
            name: String(rawSpec?.name ?? model.id),
            server_url: String(rawSpec?.server_url ?? ""),
            transport:
              (rawSpec?.transport as ProxyMcpServerSpec["transport"]) ??
              "streamable_http",
            default_prefix:
              rawSpec?.default_prefix != null
                ? String(rawSpec.default_prefix)
                : undefined,
            headers: rawSpec?.headers as Record<string, string> | undefined,
            oauth: rawSpec?.oauth as ProxyMcpServerSpec["oauth"] | undefined,
          },
        });
      } else {
        setEditMcpServer({
          customModelId,
          spec: {
            kind: "user",
            id: model.id,
            name: model.id,
            variant: (model.variant ??
              "node-headless") as AddMcpServerSpec["variant"],
            command: String(rawSpec?.command ?? model.command ?? ""),
            base_image:
              rawSpec?.base_image != null
                ? String(rawSpec.base_image)
                : (model.base_image ?? undefined),
            envs: rawSpec?.envs as Record<string, string> | undefined,
            default_prefix:
              rawSpec?.default_prefix != null
                ? String(rawSpec.default_prefix)
                : undefined,
          },
        });
      }
    },
    [
      modal,
      serviceInfo,
      existingModelRefs,
      editCustomModelMutation.mutate,
      addCustomModelMutation.mutate,
      addCustomModelMutation.isPending,
      installNewlyDuplicatedModel,
    ],
  );

  // A duplicated model installs itself immediately, using the definition's own default prefix/envs -
  // no separate "click Install" step, unlike a freshly hand-added custom model.
  const handleEditCustomModelClick = useCallback(
    (model: ServiceModel) => {
      const customModelId = model.custom;
      if (!customModelId) {
        toast.error("Model is not a custom model.");
        return;
      }
      if (!serviceInfo?.custom_model_spec) return;
      const fields = serviceInfo.custom_model_spec.fields;

      // Mirrors the Install flow: the dialog closes the moment the request is submitted, and the
      // row's Status column shows "Editing..." for the duration instead of a spinner inside the
      // dialog. That also sidesteps needing to keep isSubmitting reactive across the closed dialog.
      modal.open(DynamicFormModal, {
        title: `Edit custom model ${model.id}`,
        duplicateTitle: `Duplicate ${model.id}`,
        fields,
        serviceId,
        initialData: (model.custom_spec ?? {}) as Record<string, unknown>,
        selfModelId: model.id,
        deferRender: true,
        submitLabel: "Save",
        onSubmit: (spec: Record<string, unknown>) => {
          const cleanedSpec = Object.fromEntries(
            Object.entries(spec).filter(
              ([_, value]) => value !== null && value !== undefined,
            ),
          ) as Record<string, unknown>;
          modal.close();
          setEditingModelId(model.id);
          editCustomModelMutation.mutate({ customModelId, spec: cleanedSpec });
        },
        onDuplicate: (spec: Record<string, unknown>) => {
          const cleanedSpec = Object.fromEntries(
            Object.entries(spec).filter(
              ([_, value]) => value !== null && value !== undefined,
            ),
          ) as Record<string, unknown>;
          const maybeId = cleanedSpec.id;
          const newModelId =
            typeof maybeId === "string" && maybeId.trim()
              ? maybeId.trim()
              : null;
          // No search-and-focus after duplicating: the new model installs itself immediately
          // below, so there's no need to jump to it in the list.
          lastAddedCustomModelIdRef.current = null;
          addCustomModelMutation.mutate(cleanedSpec, {
            onSuccess: () => {
              if (newModelId) {
                installNewlyDuplicatedModel(
                  newModelId,
                  typeof cleanedSpec.size === "string"
                    ? cleanedSpec.size
                    : undefined,
                  buildDuplicateInstallSpec(cleanedSpec),
                );
              }
            },
          });
        },
        existingModels: existingModelRefs,
        isSubmitting: addCustomModelMutation.isPending,
      });
    },
    [
      modal,
      serviceId,
      serviceInfo,
      existingModelRefs,
      editCustomModelMutation.mutate,
      addCustomModelMutation.mutate,
      addCustomModelMutation.isPending,
      installNewlyDuplicatedModel,
    ],
  );

  const sortedModels = useMemo(() => {
    if (!modelsData?.list) return [];

    return [...modelsData.list].sort((a, b) => {
      if (a.installed !== b.installed) {
        return a.installed ? -1 : 1;
      }
      if (a.type !== b.type) {
        return a.type.localeCompare(b.type);
      }
      return a.id.localeCompare(b.id);
    });
  }, [modelsData]);

  const filteredModels = useMemo(() => {
    if (!sortedModels) return [];

    const normalizedFilterText = deferredFilterText.toLowerCase();

    return sortedModels.filter((model) => {
      const matchesText =
        !normalizedFilterText ||
        model.id.toLowerCase().includes(normalizedFilterText);
      const matchesType = filterType === "__all" || model.type === filterType;
      const matchesInstalled =
        filterInstalled === "__all" ||
        (filterInstalled === "installed" && model.installed) ||
        (filterInstalled === "notinstalled" && !model.installed);

      const matchesDownloaded =
        filterDownloaded === "__all" ||
        (filterDownloaded === "downloaded" && !!model.downloaded) ||
        (filterDownloaded === "notdownloaded" && !model.downloaded);
      const matchesCustom =
        filterCustom === "__all" ||
        (filterCustom === "onlycustom" && !!model.custom) ||
        (filterCustom === "onlynotcustom" && !model.custom);

      return (
        matchesText &&
        matchesType &&
        matchesInstalled &&
        matchesDownloaded &&
        matchesCustom
      );
    });
  }, [
    sortedModels,
    deferredFilterText,
    filterType,
    filterInstalled,
    filterDownloaded,
    filterCustom,
  ]);

  if (showEntrySkeleton) {
    return <ServiceModelsSkeleton />;
  }

  return (
    <div className="w-full mx-auto p-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold">Models for {serviceId}</h1>
          <div className="flex gap-2">
            {catalogRefreshUnavailableReason ? (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span>
                      <Button variant="outline" disabled>
                        ↺ Refresh
                      </Button>
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    {catalogRefreshUnavailableReason}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            ) : (
              <Button
                variant="outline"
                onClick={() => refreshMutation.mutate()}
                disabled={refreshMutation.isPending}
              >
                {getRefreshButtonLabel(
                  refreshMutation.isPending,
                  isOllamaExternal,
                )}
              </Button>
            )}
            {serviceId === "mcp" && serviceInfo?.custom_model_spec && (
              <Button
                onClick={() => {
                  setMcpApiError(null);
                  setAddMcpServerOpen(true);
                }}
              >
                Add MCP Server
              </Button>
            )}
            {serviceInfo?.custom_model_spec && serviceId !== "mcp" && (
              <Button onClick={handleAddCustomModel} variant="outline">
                Add custom model
              </Button>
            )}
          </div>
        </div>

        <div className="flex flex-col md:flex-row gap-4">
          <div className="md:flex-1">
            <Label htmlFor="search-models" className="sr-only">
              Search models
            </Label>
            <Input
              id="search-models"
              placeholder="Search models..."
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
            />
          </div>
          <div className="flex gap-4">
            <div className="space-y-1">
              <Select
                value={filterType}
                onValueChange={(v) => startTransition(() => setFilterType(v))}
              >
                <SelectTrigger id="filter-type" className="w-[200px]">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all">All types</SelectItem>
                  {Object.entries(MODEL_TYPES).map(([key, label]) => (
                    <SelectItem key={key} value={key}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1">
              <Select
                value={filterInstalled}
                onValueChange={(v) =>
                  startTransition(() => setFilterInstalled(v))
                }
              >
                <SelectTrigger id="filter-installed" className="w-[200px]">
                  <SelectValue placeholder="Installation status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all">All statuses</SelectItem>
                  <SelectItem value="installed">Installed</SelectItem>
                  <SelectItem value="notinstalled">Not installed</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1">
              <Select
                value={filterDownloaded}
                onValueChange={(v) =>
                  startTransition(() => setFilterDownloaded(v))
                }
              >
                <SelectTrigger id="filter-downloaded" className="w-[200px]">
                  <SelectValue placeholder="Downloaded" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all">All</SelectItem>
                  <SelectItem value="downloaded">Only downloaded</SelectItem>
                  <SelectItem value="notdownloaded">
                    Only not downloaded
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            {serviceInfo?.custom_model_spec && (
              <div className="space-y-1">
                <Select
                  value={filterCustom}
                  onValueChange={(v) =>
                    startTransition(() => setFilterCustom(v))
                  }
                >
                  <SelectTrigger id="filter-custom" className="w-[200px]">
                    <SelectValue placeholder="Custom" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__all">All models</SelectItem>
                    <SelectItem value="onlycustom">Only custom</SelectItem>
                    <SelectItem value="onlynotcustom">
                      Only not custom
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>
        </div>
      </div>

      {gpuStatsQuery.data != null && !gpuStatsQuery.isError && (
        <GpuStatsPanel stats={gpuStatsQuery.data} />
      )}

      <ModelsTable
        models={filteredModels}
        loadFailed={modelsQuery.isError && !modelsData}
        onRetryLoad={handleRetryLoadModels}
        serviceId={serviceId}
        isCpuOnly={isCpuOnly}
        installingModelId={installingModelId}
        editingModelId={editingModelId}
        isInstallingAny={installMutation.isPending}
        isPurgePending={purgeMutation.isPending}
        isRemoveCustomPending={removeCustomModelMutation.isPending}
        isTestPending={testMutation.isPending}
        hasRealProgressByModelRef={hasRealProgressByModelRef}
        onInstallClick={handleInstallClick}
        onCancelInstall={handleCancelInstall}
        onRemoveCustomModelClick={handleRemoveCustomModelClick}
        onEditMcpServerClick={handleEditMcpServerClick}
        onEditCustomModelClick={handleEditCustomModelClick}
        onEditModelOptionsClick={handleEditModelOptionsClick}
        onPurgeClick={handlePurgeClick}
        onTestClick={handleTestClick}
        onShowDockerLogs={handleShowDockerLogs}
        onShowDockerCompose={handleShowDockerCompose}
        onRestartDocker={handleRestartDocker}
        onUninstallClick={handleUninstallClick}
      />

      <AddMcpServerModal
        open={addMcpServerOpen}
        onOpenChange={setAddMcpServerOpen}
        onSubmit={(payload) => addMcpServerMutation.mutate(payload)}
        isSubmitting={addMcpServerMutation.isPending}
        dockerFields={serviceInfo?.custom_model_spec?.fields ?? []}
        apiError={mcpApiError}
      />

      {editMcpServer && (
        <AddMcpServerModal
          open={true}
          onOpenChange={(open) => {
            if (!open) {
              setEditMcpServer(null);
              setMcpApiError(null);
            }
          }}
          onSubmit={(payload) => {
            if (payload.kind === "user" || payload.kind === "proxy") {
              setEditMcpServer(null);
              setMcpApiError(null);
              setEditingModelId(editMcpServer.spec.id);
              editMcpServerMutation.mutate({
                customModelId: editMcpServer.customModelId,
                spec: payload,
              });
            }
          }}
          onDuplicate={(payload) => {
            if (payload.kind === "user" || payload.kind === "proxy") {
              setEditMcpServer(null);
              setMcpApiError(null);
              addMcpServerMutation.mutate(payload, {
                onSuccess: () =>
                  installNewlyDuplicatedModel(
                    payload.id,
                    undefined,
                    buildDuplicateInstallSpec(
                      payload as unknown as Record<string, unknown>,
                    ),
                  ),
              });
            }
          }}
          isSubmitting={false}
          dockerFields={[]}
          existingModels={existingModelRefs}
          initialValues={editMcpServer.spec}
          title="Edit MCP Server"
          notice="If the server is currently installed, saving will briefly stop and restart it to apply the new settings."
          apiError={mcpApiError}
        />
      )}

      <AlertDialog
        open={oauthPromptModelId !== null}
        onOpenChange={(open) => {
          if (!open) setOauthPromptModelId(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Authorization required</AlertDialogTitle>
            <AlertDialogDescription>
              "{oauthPromptModelId}" requires OAuth authorization before it can
              be used. Authorize now, or use the "Authorize" action on its row
              later.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Later</AlertDialogCancel>
            <AlertDialogAction
              onClick={(e) => {
                e.preventDefault();
                if (oauthPromptModelId) {
                  startOauthAfterInstallMutation.mutate(oauthPromptModelId);
                }
              }}
              disabled={startOauthAfterInstallMutation.isPending}
            >
              {startOauthAfterInstallMutation.isPending
                ? "Opening…"
                : "Authorize"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

const EMPTY_SPEC: Record<string, unknown> = {};

type ModelsTableProps = {
  models: ServiceModel[];
  loadFailed: boolean;
  onRetryLoad: () => void;
  serviceId: string;
  isCpuOnly: boolean;
  installingModelId: string | null;
  editingModelId: string | null;
  isInstallingAny: boolean;
  isRemoveCustomPending: boolean;
  isPurgePending: boolean;
  isTestPending: boolean;
  hasRealProgressByModelRef: RefObject<Record<string, boolean>>;
  onInstallClick: (model: ServiceModel) => void | Promise<void>;
  onCancelInstall: (modelId: string) => void | Promise<void>;
  onRemoveCustomModelClick: (model: ServiceModel) => void;
  onEditMcpServerClick: (model: ServiceModel) => void;
  onEditCustomModelClick: (model: ServiceModel) => void;
  onEditModelOptionsClick: (model: ServiceModel) => void | Promise<void>;
  onPurgeClick: (modelId: string) => void;
  onTestClick: (model: ServiceModel) => void;
  onShowDockerLogs: (modelId: string) => void | Promise<void>;
  onShowDockerCompose: (modelId: string) => void | Promise<void>;
  onRestartDocker: (modelId: string) => void;
  onUninstallClick: (modelId: string) => void;
};

const ModelsTable = memo(function ModelsTable({
  models,
  loadFailed,
  onRetryLoad,
  serviceId,
  isCpuOnly,
  installingModelId,
  editingModelId,
  isInstallingAny,
  isRemoveCustomPending,
  isPurgePending,
  isTestPending,
  hasRealProgressByModelRef,
  onInstallClick,
  onCancelInstall,
  onRemoveCustomModelClick,
  onEditMcpServerClick,
  onEditCustomModelClick,
  onEditModelOptionsClick,
  onPurgeClick,
  onTestClick,
  onShowDockerLogs,
  onShowDockerCompose,
  onRestartDocker,
  onUninstallClick,
}: ModelsTableProps) {
  return (
    <div className="border rounded-lg">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="min-w-[160px]">Model ID</TableHead>
            {serviceId !== "mcp" && <TableHead>Type</TableHead>}
            <TableHead className="min-w-[150px]">Status</TableHead>
            <TableHead>Size</TableHead>
            <TableHead>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="flex items-center gap-1 cursor-default">
                      {isCpuOnly ? "RAM (est.)" : "VRAM (est.)"}{" "}
                      <Info className="h-3 w-3 text-muted-foreground" />
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    {isCpuOnly
                      ? "Estimate based on the model architecture. Values are in GiB (binary gigabytes)."
                      : "Estimate based on the model architecture. Values are in GiB (binary gigabytes). Does not account for the CUDA context overhead on the GPU."}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </TableHead>
            <TableHead>Configuration</TableHead>
            <TableHead className="text-right min-w-[165px]">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loadFailed ? (
            <TableRow>
              <TableCell
                colSpan={serviceId === "mcp" ? 6 : 7}
                className="text-center text-muted-foreground"
              >
                <div className="flex flex-col items-center justify-center gap-2 py-4">
                  <AlertCircle className="h-6 w-6" />
                  <p className="text-sm">Failed to load models</p>
                  <Button variant="outline" size="sm" onClick={onRetryLoad}>
                    Retry
                  </Button>
                </div>
              </TableCell>
            </TableRow>
          ) : models.length === 0 ? (
            <TableRow>
              <TableCell
                colSpan={serviceId === "mcp" ? 6 : 7}
                className="text-center text-muted-foreground"
              >
                No models found
              </TableCell>
            </TableRow>
          ) : (
            models.map((model) => (
              <ModelRow
                key={model.id}
                model={model}
                serviceId={serviceId}
                isCpuOnly={isCpuOnly}
                installingModelId={installingModelId}
                editingModelId={editingModelId}
                isInstallingAny={isInstallingAny}
                isRemoveCustomPending={isRemoveCustomPending}
                isPurgePending={isPurgePending}
                isTestPending={isTestPending}
                hasRealProgressByModelRef={hasRealProgressByModelRef}
                onInstallClick={onInstallClick}
                onCancelInstall={onCancelInstall}
                onRemoveCustomModelClick={onRemoveCustomModelClick}
                onEditMcpServerClick={onEditMcpServerClick}
                onEditCustomModelClick={onEditCustomModelClick}
                onEditModelOptionsClick={onEditModelOptionsClick}
                onPurgeClick={onPurgeClick}
                onTestClick={onTestClick}
                onShowDockerLogs={onShowDockerLogs}
                onShowDockerCompose={onShowDockerCompose}
                onRestartDocker={onRestartDocker}
                onUninstallClick={onUninstallClick}
              />
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
});

type ModelRowProps = {
  model: ServiceModel;
  serviceId: string;
  isCpuOnly: boolean;
  installingModelId: string | null;
  editingModelId: string | null;
  isInstallingAny: boolean;
  isRemoveCustomPending: boolean;
  isPurgePending: boolean;
  isTestPending: boolean;
  hasRealProgressByModelRef: RefObject<Record<string, boolean>>;
  onInstallClick: (model: ServiceModel) => void | Promise<void>;
  onCancelInstall: (modelId: string) => void | Promise<void>;
  onRemoveCustomModelClick: (model: ServiceModel) => void;
  onEditMcpServerClick: (model: ServiceModel) => void;
  onEditCustomModelClick: (model: ServiceModel) => void;
  onEditModelOptionsClick: (model: ServiceModel) => void | Promise<void>;
  onPurgeClick: (modelId: string) => void;
  onTestClick: (model: ServiceModel) => void;
  onShowDockerLogs: (modelId: string) => void | Promise<void>;
  onShowDockerCompose: (modelId: string) => void | Promise<void>;
  onRestartDocker: (modelId: string) => void;
  onUninstallClick: (modelId: string) => void;
};

function OAuthStatusBadge({ status }: { status: McpOAuthStatusValue }) {
  if (status === "authorized") {
    return <Badge variant="default">OAuth: authorized</Badge>;
  }
  if (status === "pending") {
    return <Badge variant="outline">OAuth: pending</Badge>;
  }
  if (status === "expired") {
    return <Badge variant="destructive">OAuth: expired</Badge>;
  }
  if (status === "error") {
    return <Badge variant="destructive">OAuth: error</Badge>;
  }
  return <Badge variant="secondary">OAuth: not authorized</Badge>;
}

const ModelRow = memo(function ModelRow({
  model,
  serviceId,
  isCpuOnly,
  installingModelId,
  editingModelId,
  isInstallingAny,
  isRemoveCustomPending,
  isPurgePending,
  isTestPending,
  hasRealProgressByModelRef,
  onInstallClick,
  onCancelInstall,
  onRemoveCustomModelClick,
  onEditMcpServerClick,
  onEditCustomModelClick,
  onEditModelOptionsClick,
  onPurgeClick,
  onTestClick,
  onShowDockerLogs,
  onShowDockerCompose,
  onRestartDocker,
  onUninstallClick,
}: ModelRowProps) {
  const isInstalled = !!model.installed;
  const isDownloaded = !!model.downloaded;
  const installedInfo =
    model.installed && typeof model.installed === "object"
      ? model.installed
      : null;
  const installedSpec = installedInfo?.spec ?? EMPTY_SPEC;

  const currentProgress = useModelInstallProgress(serviceId, model.id);
  const isInProgress = !!currentProgress;
  const hasProgressStage =
    !!installedInfo?.stage && installedInfo?.value !== undefined;
  const isInstallingCurrent = isInstallingAny && installingModelId === model.id;
  const isEditingCurrent = editingModelId === model.id;

  const oauthConfig = (
    model.custom_spec as { oauth?: { enabled?: boolean } } | null
  )?.oauth;
  const oauthEnabled = serviceId === "mcp" && !!oauthConfig?.enabled;

  const oauthStatusQuery = useQuery({
    queryKey: [
      "admin",
      "services",
      serviceId,
      "models",
      model.id,
      "oauth-status",
    ],
    queryFn: () => apiClient.getMcpOAuthStatus(serviceId, model.id),
    enabled: oauthEnabled,
    refetchInterval: (query) =>
      query.state.data?.status === "pending" ? 2000 : false,
  });

  const startOauthMutation = useMutation({
    mutationFn: () => apiClient.startMcpOAuth(serviceId, model.id),
    onSuccess: (data) => {
      window.open(data.authorize_url, "_blank", "noopener,noreferrer");
      oauthStatusQuery.refetch();
    },
    onError: (error) => {
      toast.error(`Failed to start authorization: ${error.message}`);
    },
  });

  const installedSpecEntries = useMemo(() => {
    if (!isInstalled || hasProgressStage)
      return [] as Array<{ key: string; displayValue: string }>;
    const entries = Object.entries(installedSpec);
    if (entries.length === 0)
      return [] as Array<{ key: string; displayValue: string }>;

    const fieldTypeByName = new Map<string, string>();
    for (const f of model.spec.fields) fieldTypeByName.set(f.name, f.type);

    return entries.map(([key, value]) => {
      const type = fieldTypeByName.get(key);
      return {
        key,
        displayValue:
          type === "password"
            ? "•••••"
            : value && typeof value === "object"
              ? JSON.stringify(value)
              : String(value ?? ""),
      };
    });
  }, [installedSpec, isInstalled, hasProgressStage, model.spec.fields]);

  return (
    <TableRow>
      <TableCell className="font-semibold">
        <div className="flex items-center gap-2">
          <div className="truncate max-w-md" title={model.id}>
            {model.id}
          </div>
          {model.custom && <Badge variant="secondary">Custom</Badge>}
          {model.variant && <Badge variant="outline">{model.variant}</Badge>}
          {model.repository_url && (
            <a
              href={model.repository_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground"
              title="Repository"
              onClick={(e) => e.stopPropagation()}
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
          )}
        </div>
        {model.description && (
          <div
            className="text-xs text-muted-foreground font-normal mt-0.5"
            title={stripMarkdownLinks(model.description)}
          >
            {renderMarkdownLinks(model.description)}
          </div>
        )}
      </TableCell>
      {serviceId !== "mcp" && (
        <TableCell className="text-sm">
          {MODEL_TYPES[model.type] || model.type}
        </TableCell>
      )}
      <TableCell>
        {isEditingCurrent ? (
          // Takes priority over the raw install-progress view below: an edit's internal
          // uninstall->update->reinstall cycle genuinely populates server-side install progress
          // partway through, and a periodic models-list refetch landing in that window would
          // otherwise flip this row from "Editing..." to a raw progress bar mid-edit. From the
          // admin's perspective this is one atomic operation, so it stays "Editing..." throughout.
          <Badge variant="outline">Editing…</Badge>
        ) : isInProgress || hasProgressStage ? (
          <ProgressBadge
            stage={
              currentProgress?.stage ||
              (installedInfo?.stage as "install" | "download")
            }
            value={currentProgress?.value ?? installedInfo?.value ?? 0}
            variant="default"
            simulated={!hasRealProgressByModelRef.current?.[model.id]}
          />
        ) : (
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={isInstalled ? "default" : "secondary"}>
              {isInstalled ? "Installed" : "Not installed"}
            </Badge>
            {!isInstalled && isDownloaded && (
              <Badge variant="outline">Downloaded</Badge>
            )}
            {model.capabilities_resolved === false && (
              <Badge
                variant="destructive"
                title="This model's type/capabilities could not be resolved from the provider's live listing or known metadata - it cannot be installed."
              >
                Unresolved
              </Badge>
            )}
            {model.stale && (
              <Badge
                variant="outline"
                title="This model no longer appears in the provider's live listing."
              >
                Stale
              </Badge>
            )}
            {isInstalled && model.is_loaded === true && (
              <Badge variant="outline">
                {isCpuOnly ? "In RAM" : "In VRAM"}
              </Badge>
            )}
            {oauthEnabled && oauthStatusQuery.data && (
              <OAuthStatusBadge status={oauthStatusQuery.data.status} />
            )}
          </div>
        )}
      </TableCell>
      <TableCell className="font-mono text-sm">{model.size || "N/A"}</TableCell>
      <TableCell className="font-mono text-sm">
        {isInstalled
          ? model.vram_estimate_gb != null
            ? `${model.vram_estimate_gb.toFixed(1)}GB`
            : "—"
          : null}
      </TableCell>
      <TableCell>
        {isInstalled && installedSpecEntries.length > 0 && !hasProgressStage ? (
          <div className="space-y-1">
            {installedSpecEntries.map(({ key, displayValue }) => (
              <div key={key} className="text-xs truncate max-w-xs">
                <span className="font-medium">{key}:</span> {displayValue}
              </div>
            ))}
          </div>
        ) : (
          <span className="text-sm text-muted-foreground">—</span>
        )}
      </TableCell>
      <TableCell className="text-right" style={{ height: "49px" }}>
        {isEditingCurrent ? (
          // No actions at all while an edit's uninstall->update->reinstall cycle is in flight -
          // there's nothing to cancel (unlike a real install), and every other action here
          // (Test, Uninstall, Docker controls, a second Edit...) would race the in-flight request.
          <div className="flex justify-end">
            <Button variant="outline" size="sm" disabled>
              <MoreVertical className="h-4 w-4" />
            </Button>
          </div>
        ) : isInProgress || hasProgressStage ? (
          <div className="flex justify-end gap-2">
            <Button
              onClick={() => onCancelInstall(model.id)}
              variant="destructive"
              size="sm"
            >
              Cancel
            </Button>
          </div>
        ) : !isInstalled ? (
          <div className="flex justify-end gap-2">
            <Button
              onClick={() => onInstallClick(model)}
              size="sm"
              disabled={
                isInstallingCurrent || model.capabilities_resolved === false
              }
              title={
                model.capabilities_resolved === false
                  ? "This model's type/capabilities could not be resolved from the provider's live listing or known metadata - it cannot be installed."
                  : undefined
              }
            >
              {isInstallingCurrent ? "Installing..." : "Install"}
            </Button>
            {model.custom && (
              <Button
                onClick={() => onRemoveCustomModelClick(model)}
                variant="destructive"
                size="sm"
                disabled={isRemoveCustomPending}
              >
                {isRemoveCustomPending ? "Removing..." : "Remove custom model"}
              </Button>
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!isDownloaded && !model.custom}
                >
                  <MoreVertical className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {model.custom && (
                  <>
                    <DropdownMenuItem
                      onClick={() =>
                        serviceId === "mcp"
                          ? onEditMcpServerClick(model)
                          : onEditCustomModelClick(model)
                      }
                    >
                      Edit
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                  </>
                )}
                {oauthEnabled && (
                  <>
                    <DropdownMenuItem
                      onClick={() => startOauthMutation.mutate()}
                      disabled={startOauthMutation.isPending}
                    >
                      {startOauthMutation.isPending ? "Opening…" : "Authorize"}
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                  </>
                )}
                <DropdownMenuItem
                  onClick={() => onPurgeClick(model.id)}
                  disabled={!isDownloaded || isPurgePending}
                  variant="destructive"
                >
                  Purge
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        ) : (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem
                onClick={() => onTestClick(model)}
                disabled={isTestPending}
              >
                {isTestPending ? "Testing..." : "Test"}
              </DropdownMenuItem>
              {model.custom ? (
                <DropdownMenuItem
                  onClick={() =>
                    serviceId === "mcp"
                      ? onEditMcpServerClick(model)
                      : onEditCustomModelClick(model)
                  }
                >
                  Edit Settings
                </DropdownMenuItem>
              ) : (
                // Ticket scope is "MCP and custom services" only - catalog models on other
                // services (Ollama, vLLM, sglang, etc.) don't get this action, even though the
                // backend route is generic enough to support them too.
                (serviceId === "mcp" || serviceId === "custom") && (
                  <DropdownMenuItem
                    onClick={() => onEditModelOptionsClick(model)}
                  >
                    Edit Settings
                  </DropdownMenuItem>
                )
              )}
              {oauthEnabled && (
                <DropdownMenuItem
                  onClick={() => startOauthMutation.mutate()}
                  disabled={startOauthMutation.isPending}
                >
                  {startOauthMutation.isPending ? "Opening…" : "Authorize"}
                </DropdownMenuItem>
              )}
              {model.has_docker && (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => onShowDockerLogs(model.id)}>
                    Docker Logs
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onClick={() => onShowDockerCompose(model.id)}
                  >
                    Docker Compose
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => onRestartDocker(model.id)}>
                    Restart Docker
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                </>
              )}
              <DropdownMenuItem
                onClick={() => onUninstallClick(model.id)}
                variant="destructive"
              >
                Uninstall
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </TableCell>
    </TableRow>
  );
});

function GpuStatsPanel({ stats }: { stats: GpuStats }) {
  if (stats.gpus && stats.gpus.length > 1) {
    return (
      <div className="mb-4 rounded-lg border px-4 py-2 text-sm">
        <span className="font-medium">GPU VRAM:</span>
        <div className="mt-1 flex flex-col gap-0.5">
          {stats.gpus.map((gpu: GpuCardStats, i: number) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: GPUs have no stable identity
            <span key={i} className="text-muted-foreground">
              <span className="font-medium text-foreground truncate max-w-[200px] inline-block align-bottom">
                {gpu.name}
              </span>
              {" — "}
              {gpu.used_vram_gb.toFixed(1)} / {gpu.total_vram_gb.toFixed(1)} GB
            </span>
          ))}
        </div>
      </div>
    );
  }
  return (
    <div className="mb-4 flex items-center gap-2 rounded-lg border px-4 py-2 text-sm">
      <span className="font-medium">GPU VRAM:</span>
      <span>
        {stats.used_vram_gb.toFixed(1)} GB used /{" "}
        {stats.total_vram_gb.toFixed(1)} GB total
      </span>
    </div>
  );
}

function ServiceModelsSkeleton() {
  return (
    <div className="w-full mx-auto p-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-6">
          <Skeleton className="h-9 w-64" />
          <Skeleton className="h-10 w-40" />
        </div>
        <div className="flex flex-col md:flex-row gap-4">
          <div className="md:flex-1">
            <Skeleton className="h-10 w-full" />
          </div>
          <div className="flex gap-4">
            <Skeleton className="h-10 w-[200px]" />
            <Skeleton className="h-10 w-[200px]" />
            <Skeleton className="h-10 w-[200px]" />
          </div>
        </div>
      </div>
      <div className="border rounded-lg">
        <div className="p-4 space-y-3">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </div>
      </div>
    </div>
  );
}
