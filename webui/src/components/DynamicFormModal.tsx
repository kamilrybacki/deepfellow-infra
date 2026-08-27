import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import type { SpecField } from "@/deepfellow/types";
/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { proposePrefix } from "@/utils/prefix";
import { useEffect, useMemo, useState } from "react";
import {
  DynamicFormFields,
  type ExistingModelRef,
  filterExistingModelsForCollisionCheck,
  getPrefixFieldName,
  hasFormDataChanged,
  mergeInitialData,
  validateFields,
} from "./DynamicFormFields";

/**
 * Toggling "duplicate" clears the `id` field so the existing add-model collision check forces a new,
 * non-colliding value before submission; toggling it back off restores the id (and prefix) being
 * edited. Both must be restored from `original` - a snapshot taken before duplicate mode was ever
 * turned on - not read back off `formData`, which by the time this runs *is* the already-cleared
 * duplicate-mode state; reading `default_prefix` from there just restored the empty string it was
 * itself set to a moment earlier.
 */
export function applyDuplicateMode(
  formData: Record<string, unknown>,
  original: { id: unknown; default_prefix?: unknown },
  duplicateMode: boolean,
): Record<string, unknown> {
  return {
    ...formData,
    id: duplicateMode ? "" : original.id,
    // Only touch the prefix if this model type actually has one - stamping a stray
    // `default_prefix: ""` onto a model with no such field would leave it there for good.
    ...("default_prefix" in formData
      ? { default_prefix: duplicateMode ? "" : original.default_prefix }
      : {}),
  };
}

interface DynamicFormModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  fields: SpecField[];
  initialData?: Record<string, unknown>;
  onSubmit: (data: Record<string, unknown>) => void;
  /** When provided, an edit form also offers a "save as new (duplicate)" checkbox that routes
   * submission here instead, with the "id" field cleared so the user must supply a new one. */
  onDuplicate?: (data: Record<string, unknown>) => void;
  /** Title shown instead of `title` while the "duplicate" checkbox is checked - e.g. "Edit settings
   * for x" becomes "Duplicate x", since the form is no longer editing the original at that point. */
  duplicateTitle?: string;
  /** When the model being edited has a different (usually wider) field set for duplication than for
   * editing - e.g. a catalog model's edit form only exposes install-time options, but duplicating it
   * needs the full add-model fields (image, command, etc.) synthesized from its live definition -
   * toggling "duplicate" swaps to these fields/values instead of just clearing `id` within `fields`. */
  duplicateFields?: SpecField[];
  duplicateInitialData?: Record<string, unknown>;
  /** Other existing models (id/effective_prefix) to check the submitted id/prefix against,
   * surfacing a "this id/prefix is already used" error inline instead of failing on the server. Include
   * the model being edited itself here too - it's excluded automatically while not duplicating, but a
   * duplicate must not collide with its own source either. */
  existingModels?: ExistingModelRef[];
  /** The id of the model this form is editing, used to exclude it from the collision check while not
   * duplicating. Falls back to `initialData.id`, which works when "id" is itself a form field - but
   * install-options-only forms (e.g. a catalog model's Edit Settings) have no "id" field at all, so
   * this must be passed explicitly there or the model would falsely collide with itself. */
  selfModelId?: string;
  isSubmitting?: boolean;
  isLoading?: boolean;
  deferRender?: boolean;
  submitLabel?: string;
  submittingLabel?: string;
  serviceId?: string;
  modelId?: string;
}

export function DynamicFormModal({
  open,
  onOpenChange,
  title,
  fields,
  initialData: initialDataProp,
  onSubmit,
  onDuplicate,
  duplicateTitle,
  duplicateFields,
  duplicateInitialData: duplicateInitialDataProp,
  existingModels = [],
  selfModelId,
  isSubmitting = false,
  isLoading = false,
  deferRender = false,
  submitLabel = "Install",
  submittingLabel = "Installing...",
  serviceId,
  modelId,
}: DynamicFormModalProps) {
  const [renderFields, setRenderFields] = useState<SpecField[]>(
    deferRender ? [] : fields,
  );

  useEffect(() => {
    if (!open) return;
    if (!deferRender) {
      setRenderFields(fields);
      return;
    }
    setRenderFields([]);
    const timeoutId = window.setTimeout(() => setRenderFields(fields), 0);
    return () => window.clearTimeout(timeoutId);
  }, [open, deferRender, fields]);

  const effectiveIsLoading =
    isLoading ||
    (deferRender && open && renderFields.length === 0 && fields.length > 0);

  // Use `fields`, not `renderFields`, so formData holds every value before deferred fields mount.
  const initialData = useMemo(
    () => mergeInitialData(fields, initialDataProp),
    [fields, initialDataProp],
  );
  const duplicateInitialData = useMemo(
    () =>
      duplicateFields
        ? mergeInitialData(duplicateFields, duplicateInitialDataProp)
        : null,
    [duplicateFields, duplicateInitialDataProp],
  );
  const [formData, setFormData] =
    useState<Record<string, unknown>>(initialData);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [duplicateMode, setDuplicateMode] = useState(false);
  // Tracks whether the prefix field was ever typed into directly, so a fresh id no longer overwrites
  // a prefix the admin deliberately chose - mirrors useDockerForm/useStdioForm/useUrlForm's own
  // per-field tracking in use-mcp-server-form.ts (AddMcpServerModal's forms).
  const [prefixIsManual, setPrefixIsManual] = useState(false);
  // Only meaningful for a genuine edit (onDuplicate present) - an "Add" form has no prior state to
  // compare against, so it's never blocked here. Skips the pointless uninstall/reinstall cycle an
  // edit submission would otherwise trigger even when nothing was actually changed.
  const isUnchanged =
    !!onDuplicate &&
    !duplicateMode &&
    !hasFormDataChanged(formData, initialData);

  useEffect(() => {
    if (open) {
      setFormData(initialData);
      setErrors({});
      setDuplicateMode(false);
      setPrefixIsManual(false);
    }
  }, [open, initialData]);

  const handleDuplicateModeChange = (checked: boolean) => {
    setDuplicateMode(checked);
    // The id (and, while duplicating, the prefix) is cleared below, so the next id keystroke should
    // propose a fresh prefix rather than leave the source model's prefix in place unannounced.
    setPrefixIsManual(false);
    if (duplicateFields && duplicateInitialData) {
      // A genuinely different field set (e.g. a catalog model duplicating into a full custom
      // model) - swap wholesale rather than clearing `id` within the current fields.
      setRenderFields(checked ? duplicateFields : fields);
      setFormData(
        checked
          ? applyDuplicateMode(
              duplicateInitialData,
              {
                id: duplicateInitialData.id,
                default_prefix: duplicateInitialData.default_prefix,
              },
              true,
            )
          : initialData,
      );
      setErrors({});
      return;
    }
    setFormData((prev) =>
      applyDuplicateMode(
        prev,
        { id: initialData.id, default_prefix: initialData.default_prefix },
        checked,
      ),
    );
  };

  const focusField = (name: string) => {
    const container = document.querySelector(
      `[data-field-name="${CSS.escape(name)}"]`,
    ) as HTMLElement | null;
    if (!container) return;
    const focusable = container.querySelector<HTMLElement>(
      "input,button,textarea,select,[role='combobox'],[role='button']",
    );
    focusable?.focus();
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (effectiveIsLoading || isSubmitting) return;
    if (isUnchanged) {
      onOpenChange(false);
      return;
    }
    const collisionCheckModels = filterExistingModelsForCollisionCheck(
      existingModels,
      selfModelId ?? initialData.id,
      duplicateMode,
    );
    const nextErrors = validateFields(
      renderFields,
      formData,
      collisionCheckModels,
    );
    setErrors(nextErrors);
    const firstInvalid = Object.keys(nextErrors)[0];
    if (firstInvalid) {
      focusField(firstInvalid);
      return;
    }
    if (duplicateMode && onDuplicate) {
      onDuplicate(formData);
      return;
    }
    onSubmit(formData);
  };

  const handleChange = (name: string, value: unknown) => {
    const prefixFieldName = getPrefixFieldName(renderFields);
    if (name === prefixFieldName) {
      setPrefixIsManual(typeof value === "string" && value.trim() !== "");
    }
    setFormData((prev) => {
      const next = { ...prev, [name]: value };
      if (name === "id" && prefixFieldName && !prefixIsManual) {
        next[prefixFieldName] = proposePrefix(
          typeof value === "string" ? value : "",
        );
      }
      return next;
    });
    setErrors((prev) => {
      if (!prev[name]) return prev;
      const next = { ...prev };
      delete next[name];
      return next;
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-[600px]"
        onInteractOutside={(e) => {
          if (isSubmitting) e.preventDefault();
        }}
        onEscapeKeyDown={(e) => {
          if (isSubmitting) e.preventDefault();
        }}
      >
        <DialogHeader>
          <DialogTitle>
            {duplicateMode && duplicateTitle ? duplicateTitle : title}
          </DialogTitle>
          <DialogDescription>
            {effectiveIsLoading
              ? "Loading form..."
              : "Fill in the required information below."}
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 px-2 py-4 max-h-[70vh] overflow-auto">
            {onDuplicate && !effectiveIsLoading && (
              <label
                htmlFor="duplicate-as-new"
                className="flex items-start gap-3 cursor-pointer rounded-md border bg-muted/30 p-3"
              >
                <Checkbox
                  id="duplicate-as-new"
                  checked={duplicateMode}
                  disabled={isSubmitting}
                  onCheckedChange={(checked) =>
                    handleDuplicateModeChange(checked === true)
                  }
                />
                <div className="grid gap-1">
                  <Label
                    htmlFor="duplicate-as-new"
                    className="leading-none cursor-pointer select-none"
                  >
                    Save as a new service (duplicate)
                  </Label>
                  <p className="text-sm text-muted-foreground">
                    Creates a copy with these settings, instead of changing the
                    original. New id and prefix must be provided The original is
                    left untouched; the copy starts with installation.
                  </p>
                </div>
              </label>
            )}
            {effectiveIsLoading ? (
              <div className="grid gap-4">
                <div className="grid gap-2">
                  <Skeleton className="h-4 w-40" />
                  <Skeleton className="h-10 w-full" />
                </div>
                <div className="grid gap-2">
                  <Skeleton className="h-4 w-48" />
                  <Skeleton className="h-10 w-full" />
                </div>
                <div className="grid gap-2">
                  <Skeleton className="h-4 w-36" />
                  <Skeleton className="h-10 w-full" />
                </div>
              </div>
            ) : (
              <DynamicFormFields
                fields={renderFields}
                formData={formData}
                errors={errors}
                onChange={handleChange}
                serviceId={serviceId}
                modelId={modelId}
                disabledFields={onDuplicate && !duplicateMode ? ["id"] : []}
              />
            )}
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={isSubmitting || effectiveIsLoading || isUnchanged}
            >
              {isSubmitting
                ? duplicateMode
                  ? "Duplicating..."
                  : submittingLabel
                : duplicateMode
                  ? "Duplicate"
                  : submitLabel}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
