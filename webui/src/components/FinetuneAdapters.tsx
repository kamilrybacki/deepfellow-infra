/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { apiClient } from "@/deepfellow/client";
import { useQuery } from "@tanstack/react-query";
import { AlertCircle } from "lucide-react";

interface FinetuneAdaptersProps {
  serviceId: string;
}

export function FinetuneAdapters({ serviceId }: FinetuneAdaptersProps) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["admin", "services", serviceId, "adapters"],
    queryFn: () => apiClient.listFinetuneAdapters(serviceId),
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Adapters</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex flex-col gap-2">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        ) : error ? (
          <Alert variant="destructive">
            <AlertCircle />
            <AlertTitle>Could not load adapters</AlertTitle>
            <AlertDescription>{error.message}</AlertDescription>
          </Alert>
        ) : !data || data.data.length === 0 ? (
          <p className="text-muted-foreground text-sm">
            No adapters yet. Fine-tune a model with df-finetune to see it here.
          </p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Base model</TableHead>
                <TableHead>Formats</TableHead>
                <TableHead>LoRA rank</TableHead>
                <TableHead>Created</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.data.map((adapter) => (
                <TableRow key={adapter.id}>
                  <TableCell className="font-medium">{adapter.id}</TableCell>
                  <TableCell>{adapter.base_model}</TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1">
                      {adapter.formats.map((format) => (
                        <Badge key={format} variant="secondary">
                          {format}
                        </Badge>
                      ))}
                    </div>
                  </TableCell>
                  <TableCell>
                    {typeof adapter.lora?.rank === "number" ||
                    typeof adapter.lora?.rank === "string"
                      ? adapter.lora.rank
                      : "—"}
                  </TableCell>
                  <TableCell>
                    {new Date(adapter.created * 1000).toLocaleString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
