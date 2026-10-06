/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import * as SliderPrimitive from "@radix-ui/react-slider";

export interface GpuAllocationSegment {
  id: string;
  label: string;
  gb: number;
}

interface ComputedSegment extends GpuAllocationSegment {
  pct: number;
}

export interface GpuAllocationLayout {
  segments: ComputedSegment[];
  /** VRAM in use that isn't attributed to any listed segment (other services, non-DeepFellow processes). */
  otherGb: number;
  otherPct: number;
  freeGb: number;
  freePct: number;
  newModelGb?: number;
  newModelPct?: number;
  /** True when the share rounds below the backend's minimum utilization, which it rejects with a 422. */
  newModelBelowMinimum?: boolean;
}

/** Mirrors `_default_gpu_memory_utilization` / `_gpu_memory_free_safety_margin` in the vLLM/SGLang
 * services: the backend never reserves more than this fraction of total VRAM, nor more than this
 * fraction of what's currently free. Keep in sync with the backend. */
export const GPU_MAX_UTILIZATION = 0.9;
export const GPU_FREE_SAFETY_MARGIN = 0.9;
/** Mirrors `_min_gpu_memory_utilization`: a share rounding below this is rejected by the backend. */
export const GPU_MIN_UTILIZATION = 0.01;

/** Rounds to 2 decimals, like the backend's `round(..., 2)` on utilization fractions. */
const round2 = (value: number) => Math.round(value * 100) / 100;

export interface GpuAllocationLayoutOptions {
  /** Live VRAM in use across the whole GPU. When larger than the listed segments, the difference is
   * shown as "other"; when omitted, only the listed segments count as used. */
  usedGb?: number;
  /** 0-1 (clamped) share of the VRAM the backend would grant a new install right now. */
  newModelPriority?: number;
  /** When set, the new model's size is this many GB instead of one derived from `newModelPriority` -
   * for an edit that keeps an installed model's share unchanged, which the backend reuses as-is. */
  newModelFixedGb?: number;
}

/** Pure layout math for the gparted-style bar, kept separate from rendering so it's unit-testable
 * without a DOM. The new model's share is carved out of free VRAM exactly the way the backend sizes it
 * (`round(priority * round(min(max, free * margin), 2), 2)` of total) - it never shrinks an existing
 * (already-running) segment. */
export function computeGpuAllocationLayout(
  segments: GpuAllocationSegment[],
  totalGb: number,
  {
    usedGb,
    newModelPriority,
    newModelFixedGb,
  }: GpuAllocationLayoutOptions = {},
): GpuAllocationLayout {
  const safeTotal = Math.max(0, totalGb);
  const pct = (gb: number) => (safeTotal > 0 ? (gb / safeTotal) * 100 : 0);

  const computedSegments = segments.map((segment) => ({
    ...segment,
    gb: Math.max(0, segment.gb),
    pct: pct(Math.max(0, segment.gb)),
  }));
  const segmentsGb = computedSegments.reduce((sum, s) => sum + s.gb, 0);
  const usedTotalGb = Math.min(
    safeTotal,
    Math.max(segmentsGb, Math.max(0, usedGb ?? 0)),
  );
  const otherGb = Math.max(0, usedTotalGb - segmentsGb);
  const freeGbBeforeNewModel = Math.max(0, safeTotal - usedTotalGb);
  const base = {
    segments: computedSegments,
    otherGb,
    otherPct: pct(otherGb),
  };

  if (newModelPriority === undefined) {
    return {
      ...base,
      freeGb: freeGbBeforeNewModel,
      freePct: pct(freeGbBeforeNewModel),
    };
  }

  let newModelGb: number;
  let belowMinimum = false;
  if (newModelFixedGb !== undefined) {
    newModelGb = Math.min(freeGbBeforeNewModel, Math.max(0, newModelFixedGb));
  } else {
    const clampedPriority = Math.min(1, Math.max(0, newModelPriority));
    const grantableFraction =
      safeTotal > 0
        ? round2(
            Math.min(
              GPU_MAX_UTILIZATION,
              (freeGbBeforeNewModel / safeTotal) * GPU_FREE_SAFETY_MARGIN,
            ),
          )
        : 0;
    const utilization = round2(clampedPriority * grantableFraction);
    newModelGb = utilization * safeTotal;
    belowMinimum = safeTotal > 0 && utilization < GPU_MIN_UTILIZATION;
  }
  const freeGb = Math.max(0, freeGbBeforeNewModel - newModelGb);
  return {
    ...base,
    freeGb,
    freePct: pct(freeGb),
    newModelGb,
    newModelPct: pct(newModelGb),
    newModelBelowMinimum: belowMinimum,
  };
}

const SEGMENT_COLORS = [
  "bg-chart-1",
  "bg-chart-2",
  "bg-chart-3",
  "bg-chart-4",
  "bg-chart-5",
];

export interface GpuAllocationBarProps {
  segments: GpuAllocationSegment[];
  totalGb: number;
  /** Live VRAM in use across the GPU, see `GpuAllocationLayoutOptions.usedGb`. */
  usedGb?: number;
  /** Short caption rendered above the bar so it's self-describing wherever it's used. */
  heading?: string;
  /** When set, renders an extra segment for the model being installed/edited, sized by a slider below
   * the bar and carved out of the free space - existing segments stay fixed.
   * An `undefined` priority means "not set": no segment is drawn, since the backend then falls back
   * to a share this bar can't know (the model definition's default or a legacy absolute value). */
  newModel?: {
    label: string;
    priority: number | undefined;
    /** See `GpuAllocationLayoutOptions.newModelFixedGb`. */
    fixedGb?: number;
    /** A legacy absolute `gpu_memory_utilization` (0-1 of total VRAM) the model keeps while unset. */
    legacyUtilization?: number;
    /** Called with `undefined` when the user resets the priority back to "not set". */
    onPriorityChange: (priority: number | undefined) => void;
  };
  className?: string;
}

export function GpuAllocationBar({
  segments,
  totalGb,
  usedGb,
  heading,
  newModel,
  className,
}: GpuAllocationBarProps) {
  const layout = computeGpuAllocationLayout(segments, totalGb, {
    usedGb,
    newModelPriority: newModel?.priority,
    newModelFixedGb: newModel?.fixedGb,
  });
  const sliderLabel = `GPU share for ${newModel?.label ?? "new model"}`;

  return (
    <div className={cn("space-y-2", className)}>
      {heading && (
        <div className="text-xs font-medium text-muted-foreground">
          {heading}
        </div>
      )}
      <div className="flex h-6 w-full overflow-hidden rounded-md border">
        {layout.segments.map((segment, index) => (
          <div
            key={segment.id}
            className={cn(
              SEGMENT_COLORS[index % SEGMENT_COLORS.length],
              "h-full shrink-0 flex items-center overflow-hidden px-1",
            )}
            style={{ width: `${segment.pct}%` }}
            title={`${segment.label}: ${segment.gb.toFixed(1)} GB (${segment.pct.toFixed(0)}%)`}
          >
            <span className="truncate text-[10px] leading-none text-primary-foreground">
              {segment.label}
            </span>
          </div>
        ))}
        {layout.otherGb > 0 && (
          <div
            className="h-full shrink-0 bg-muted-foreground/40"
            style={{ width: `${layout.otherPct}%` }}
            title={`Other GPU usage: ${layout.otherGb.toFixed(1)} GB (${layout.otherPct.toFixed(0)}%)`}
          />
        )}
        {layout.newModelGb !== undefined && (
          <div
            className="h-full shrink-0 flex items-center overflow-hidden px-1 bg-primary/70"
            style={{ width: `${layout.newModelPct}%` }}
            title={`${newModel?.label ?? "New model"}: ${layout.newModelGb.toFixed(1)} GB (${(layout.newModelPct ?? 0).toFixed(0)}%)`}
          >
            <span className="truncate text-[10px] leading-none text-primary-foreground">
              {newModel?.label ?? "New model"}
            </span>
          </div>
        )}
        <div
          className="h-full shrink-0 bg-muted"
          style={{ width: `${layout.freePct}%` }}
          title={`Free: ${layout.freeGb.toFixed(1)} GB (${layout.freePct.toFixed(0)}%)`}
        />
      </div>
      {newModel && (
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{newModel.label}</span>
            <span>
              {newModel.priority !== undefined
                ? totalGb > 0
                  ? `${(layout.newModelGb ?? 0).toFixed(1)} GB (${Math.round(newModel.priority * 100)}% of free VRAM)`
                  : `${Math.round(newModel.priority * 100)}% of free VRAM`
                : newModel.legacyUtilization !== undefined
                  ? `Fixed at ${Math.round(newModel.legacyUtilization * 100)}% of total VRAM (legacy setting)`
                  : "Default share (not set)"}
            </span>
          </div>
          {newModel.priority === undefined ? (
            // No slider while unset: a thumb would have to sit at some value that isn't what the
            // backend will actually use, and picking exactly that value wouldn't register as a change.
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => newModel.onPriorityChange(1)}
            >
              Set share
            </Button>
          ) : (
            <div className="flex items-center gap-2">
              {/* Radix primitives directly, not the generated `ui/slider`, so the focusable thumb
                  (the element with role="slider") gets an accessible name. */}
              <SliderPrimitive.Root
                data-slot="slider"
                className="relative flex w-full touch-none items-center select-none"
                value={[Math.round(newModel.priority * 100)]}
                min={1}
                max={100}
                step={1}
                onValueChange={([value]) =>
                  newModel.onPriorityChange((value ?? 1) / 100)
                }
              >
                <SliderPrimitive.Track className="bg-muted relative h-1.5 w-full grow overflow-hidden rounded-full">
                  <SliderPrimitive.Range className="bg-primary absolute h-full" />
                </SliderPrimitive.Track>
                <SliderPrimitive.Thumb
                  aria-label={sliderLabel}
                  aria-valuetext={`${Math.round(newModel.priority * 100)}% of free VRAM`}
                  className="border-primary bg-background ring-ring/50 block size-4 shrink-0 rounded-full border shadow-sm transition-[color,box-shadow] hover:ring-4 focus-visible:ring-4 focus-visible:outline-hidden"
                />
              </SliderPrimitive.Root>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => newModel.onPriorityChange(undefined)}
              >
                Reset
              </Button>
            </div>
          )}
          {layout.newModelBelowMinimum && (
            <p className="text-xs text-destructive" role="alert">
              This share is below 1% of total VRAM and will be rejected. Raise
              it or free up GPU memory.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
