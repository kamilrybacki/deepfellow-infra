/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { Checkbox } from "@/components/ui/checkbox";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { apiClient } from "@/deepfellow/client";
import type { SpecField } from "@/deepfellow/types";
import {
  CheckIcon,
  ChevronsUpDownIcon,
  Loader2Icon,
  RotateCcwIcon,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ListInput } from "./ListInput";
import { MapInput } from "./MapInput";

export interface DynamicFormFieldsProps {
  fields: SpecField[];
  formData: Record<string, unknown>;
  errors: Record<string, string>;
  onChange: (name: string, value: unknown) => void;
  serviceId?: string;
}

interface DockerTagsFieldProps {
  field: SpecField;
  value: string;
  onChange: (value: string) => void;
  serviceId: string;
  hardware: string | undefined;
}

function DockerTagsField({
  field,
  value,
  onChange,
  serviceId,
  hardware,
}: DockerTagsFieldProps) {
  const [open, setOpen] = useState(false);
  const [tags, setTags] = useState<string[]>([]);
  const [defaultTag, setDefaultTag] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const previousHardware = useRef(hardware);

  useEffect(() => {
    if (previousHardware.current !== hardware) {
      previousHardware.current = hardware;
      onChange("");
    }
  }, [hardware, onChange]);

  const fetchTags = useCallback(async () => {
    setLoading(true);
    setError(false);
    try {
      const result = await apiClient.getDockerTags(serviceId, hardware);
      setTags(result.tags);
      setDefaultTag(result.default);
    } catch {
      setError(true);
      setTags([]);
    } finally {
      setLoading(false);
    }
  }, [serviceId, hardware]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(false);
    void apiClient.getDockerTags(serviceId, hardware).then(
      (result) => {
        if (!cancelled) {
          setTags(result.tags);
          setDefaultTag(result.default);
          setLoading(false);
        }
      },
      () => {
        if (!cancelled) {
          setError(true);
          setTags([]);
          setLoading(false);
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, [serviceId, hardware]);

  const busy = loading;

  if (error && tags.length === 0) {
    return (
      <div className="flex gap-2">
        <Input
          id={field.name}
          placeholder={field.placeholder || "e.g. 0.20.4"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
        <button
          type="button"
          onClick={fetchTags}
          disabled={busy}
          className="shrink-0 p-2 rounded-md border border-input bg-background hover:bg-accent hover:text-accent-foreground disabled:opacity-50"
          title="Retry fetching tags"
        >
          {busy ? (
            <Loader2Icon className="size-4 animate-spin" />
          ) : (
            <RotateCcwIcon className="size-4" />
          )}
        </button>
      </div>
    );
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          role="combobox"
          aria-expanded={open}
          aria-controls="version-listbox"
          className="flex h-9 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
          disabled={busy}
        >
          <span className={value ? "" : "text-muted-foreground"}>
            {loading
              ? "Loading tags…"
              : value ||
                (defaultTag ? `${defaultTag} (current)` : "Select version…")}
          </span>
          {loading ? (
            <Loader2Icon className="ml-2 size-4 shrink-0 animate-spin opacity-50" />
          ) : (
            <ChevronsUpDownIcon className="ml-2 size-4 shrink-0 opacity-50" />
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-[300px] p-0" align="start">
        <Command>
          <CommandInput placeholder="Search tags…" />
          <CommandList
            id="version-listbox"
            onWheel={(e) => e.nativeEvent.stopPropagation()}
          >
            <CommandEmpty>No tags found.</CommandEmpty>
            <CommandGroup>
              {tags.map((tag) => (
                <CommandItem
                  key={tag}
                  value={tag}
                  onSelect={(selected) => {
                    onChange(selected === value ? "" : selected);
                    setOpen(false);
                  }}
                >
                  {tag}
                  {!value && tag === defaultTag && (
                    <span className="ml-auto text-muted-foreground text-xs">
                      current
                    </span>
                  )}
                  {value === tag && <CheckIcon className="ml-auto size-4" />}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

export function DynamicFormFields({
  fields,
  formData,
  errors,
  onChange,
  serviceId,
}: DynamicFormFieldsProps) {
  const visibleFields = useMemo(
    () =>
      fields.filter((field) => {
        if (!field.display) return true;
        const [name, value] = field.display.split("=");
        return name && value ? formData[name] === value : true;
      }),
    [fields, formData],
  );

  return (
    <>
      {visibleFields.map((field) => (
        <div
          key={field.name}
          className="grid gap-2"
          data-field-name={field.name}
        >
          <Label htmlFor={field.name}>
            {field.description}
            {field.required && (
              <span className="text-destructive text-sm ml-1">*</span>
            )}
            {!field.required && (
              <span className="text-muted-foreground text-sm ml-1">
                (optional)
              </span>
            )}
          </Label>
          {field.type === "docker-tags" && serviceId ? (
            <DockerTagsField
              field={field}
              value={(formData[field.name] as string | undefined) ?? ""}
              onChange={(v) => onChange(field.name, v)}
              serviceId={serviceId}
              hardware={
                field.depends_on
                  ? (formData[field.depends_on] as string | undefined)
                  : undefined
              }
            />
          ) : field.type === "bool" ? (
            <div className="flex items-center space-x-2">
              <Checkbox
                id={field.name}
                checked={formData[field.name] === true}
                onCheckedChange={(checked) =>
                  onChange(field.name, checked === true)
                }
              />
            </div>
          ) : field.type === "oneof" ? (
            <Select
              value={
                (formData[field.name] as string | undefined) ||
                (field.default as string | undefined) ||
                "__none__"
              }
              onValueChange={(value) =>
                onChange(field.name, value === "__none__" ? "" : value)
              }
            >
              <SelectTrigger id={field.name} className="w-full">
                <SelectValue placeholder="Select an option" />
              </SelectTrigger>
              <SelectContent>
                {!field.values?.length && (
                  <SelectItem value="__none__">
                    <span className="text-muted-foreground">None</span>
                  </SelectItem>
                )}
                {field.values
                  ?.filter(
                    (val) => (typeof val === "string" ? val : val.value) !== "",
                  )
                  .map((val) => (
                    <SelectItem
                      key={typeof val === "string" ? val : val.value}
                      value={typeof val === "string" ? val : val.value}
                    >
                      {typeof val === "string" ? val : val.label}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          ) : field.type === "list" ? (
            <ListInput
              value={(formData[field.name] as string[] | undefined) ?? []}
              onChange={(value) => onChange(field.name, value)}
              placeholder={field.placeholder}
            />
          ) : field.type === "map" ? (
            <MapInput
              value={
                (formData[field.name] as Record<string, string> | undefined) ??
                {}
              }
              onChange={(value) => onChange(field.name, value)}
              placeholder={field.placeholder}
            />
          ) : field.type === "textarea" ? (
            <Textarea
              id={field.name}
              placeholder={field.placeholder || ""}
              required={field.required}
              value={(formData[field.name] as string | undefined) ?? ""}
              onChange={(e) => onChange(field.name, e.target.value)}
              aria-invalid={!!errors[field.name]}
            />
          ) : (
            <Input
              id={field.name}
              type={field.type}
              placeholder={field.placeholder || ""}
              required={field.required}
              value={
                (formData[field.name] as string | number | undefined) ?? ""
              }
              onChange={(e) => {
                const value =
                  field.type === "number"
                    ? e.target.valueAsNumber
                    : e.target.value;
                onChange(field.name, value);
              }}
              aria-invalid={!!errors[field.name]}
            />
          )}
          {errors[field.name] && (
            <div className="text-sm text-destructive">{errors[field.name]}</div>
          )}
        </div>
      ))}
    </>
  );
}

export function initFormData(fields: SpecField[]): Record<string, unknown> {
  const initial: Record<string, unknown> = {};
  for (const field of fields) {
    if (field.type === "bool") {
      initial[field.name] = field.default === true || field.default === "true";
    } else if (field.type === "list") {
      if (typeof field.default === "string" && field.default.startsWith("[")) {
        try {
          initial[field.name] = JSON.parse(field.default) ?? [];
        } catch {
          initial[field.name] = [];
        }
      } else {
        initial[field.name] = Array.isArray(field.default) ? field.default : [];
      }
    } else if (field.type === "map") {
      if (typeof field.default === "string" && field.default.startsWith("{")) {
        try {
          initial[field.name] = JSON.parse(field.default) ?? {};
        } catch {
          initial[field.name] = {};
        }
      } else {
        initial[field.name] = field.default ?? {};
      }
    } else if (field.type === "oneof") {
      const firstVal = field.values?.[0];
      const firstValue = firstVal
        ? typeof firstVal === "string"
          ? firstVal
          : firstVal.value
        : undefined;
      initial[field.name] =
        field.default !== undefined && field.default !== null
          ? field.default
          : (firstValue ?? "");
    } else if (field.default !== undefined && field.default !== null) {
      initial[field.name] = field.default;
    }
  }
  return initial;
}

export function mergeInitialData(
  fields: SpecField[],
  initialData: Record<string, unknown> | undefined,
): Record<string, unknown> {
  const defaults = initFormData(fields);
  if (!initialData) return defaults;
  const merged = { ...defaults };
  for (const field of fields) {
    if (field.name in initialData) {
      merged[field.name] = initialData[field.name];
    }
  }
  return merged;
}

export function validateFields(
  fields: SpecField[],
  formData: Record<string, unknown>,
): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const field of fields) {
    if (field.type === "map" && field.required_keys?.length) {
      const map =
        formData[field.name] && typeof formData[field.name] === "object"
          ? (formData[field.name] as Record<string, unknown>)
          : {};
      const missing = field.required_keys.filter((key) => {
        const v = map[key];
        return v === null || v === undefined || String(v).trim() === "";
      });
      if (missing.length > 0) {
        errors[field.name] = `Required: ${missing.join(", ")}`;
        continue;
      }
    }
    if (!field.required) continue;
    const value = formData[field.name];
    switch (field.type) {
      case "oneof":
        if (typeof value !== "string" || !value.trim())
          errors[field.name] = "This field is required.";
        break;
      case "list": {
        const list = Array.isArray(value) ? value : [];
        if (list.filter((v) => typeof v === "string" && v.trim()).length === 0)
          errors[field.name] = "Please add at least one item.";
        break;
      }
      case "map": {
        const map =
          value && typeof value === "object"
            ? (value as Record<string, unknown>)
            : {};
        if (Object.keys(map).filter((k) => k.trim()).length === 0)
          errors[field.name] = "Please add at least one pair.";
        break;
      }
      case "number":
        if (typeof value !== "number" || Number.isNaN(value))
          errors[field.name] = "This field is required.";
        break;
      case "bool":
        break;
      default:
        if (
          value === null ||
          value === undefined ||
          String(value).trim() === ""
        )
          errors[field.name] = "This field is required.";
    }
  }
  return errors;
}
