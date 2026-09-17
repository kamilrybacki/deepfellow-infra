/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { Badge } from "@/components/ui/badge";
import type { InstallStage } from "@/state/install-progress-store";
import { getStageLabel } from "@/utils/sse-stream";

interface ProgressBadgeProps {
  stage: InstallStage;
  value: number; // 0.0 to 1.0
  variant?: "default" | "secondary";
  simulated?: boolean;
}

export function ProgressBadge({
  stage,
  value,
  variant = "default",
}: ProgressBadgeProps) {
  const percentage = (value * 100).toFixed(1);
  const label = getStageLabel(stage);

  return (
    <Badge variant={variant}>
      {label} {percentage}%
    </Badge>
  );
}
