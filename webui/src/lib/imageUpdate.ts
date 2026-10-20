/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import type { NewerNeedsCuda, SpecField } from "@/deepfellow/types";

export type ImageUpdatePlan =
  | { kind: "update"; from: string | null; to: string; warning?: string }
  | { kind: "up-to-date"; version: string; warning?: string }
  | { kind: "no-release" };

/** Tell the user that a newer version exists but the host's NVIDIA driver is too old for it. */
export function cudaWarning(needsCuda: NewerNeedsCuda): string {
  return `${needsCuda.tag} is newer but needs CUDA ${needsCuda.required_cuda}, and the NVIDIA driver on this host supports CUDA ${needsCuda.host_cuda}. Update the NVIDIA driver to be able to use it.`;
}

/**
 * Decide what an update does. `newest` is picked by the backend (it keeps the variant of the running
 * tag, e.g. `-rocm`); `from` is the running tag: the installed tag or, when the install uses the bundled
 * default, `defaultTag`; null only when neither is known. `needsCuda` is a newer version held back by
 * the host's CUDA version; it becomes a `warning` on the plan.
 */
export function planImageUpdate(
  installedTag: string | null | undefined,
  defaultTag: string | null | undefined,
  newest: string | null | undefined,
  needsCuda?: NewerNeedsCuda | null,
): ImageUpdatePlan {
  if (!newest) return { kind: "no-release" };
  const warning = needsCuda ? cudaWarning(needsCuda) : undefined;
  const running = installedTag || defaultTag || null;
  if (running === newest)
    return { kind: "up-to-date", version: newest, warning };
  return { kind: "update", from: running, to: newest, warning };
}

/** The field that selects the Docker image version, if the form has one. */
export function findImageVersionField(
  fields: SpecField[] | undefined,
): SpecField | undefined {
  return fields?.find((field) => field.type === "docker-tags");
}
