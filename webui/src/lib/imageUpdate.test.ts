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
  cudaWarning,
  findImageVersionField,
  planImageUpdate,
} from "./imageUpdate";

describe("planImageUpdate", () => {
  it("updates from the installed tag to the newest one", () => {
    expect(planImageUpdate("v2.2.0", "v2.4.0", "v2.4.0")).toEqual({
      kind: "update",
      from: "v2.2.0",
      to: "v2.4.0",
    });
  });

  it("treats an install without a tag as running the default", () => {
    expect(planImageUpdate(undefined, "v2.4.0", "v2.4.0")).toEqual({
      kind: "up-to-date",
      version: "v2.4.0",
    });
    expect(planImageUpdate("", "v2.2.0", "v2.4.0")).toEqual({
      kind: "update",
      from: "v2.2.0",
      to: "v2.4.0",
    });
  });

  it("keeps variant tags as they come from the backend", () => {
    expect(
      planImageUpdate("server-intel-b5966", null, "server-intel-b6100"),
    ).toEqual({
      kind: "update",
      from: "server-intel-b5966",
      to: "server-intel-b6100",
    });
    expect(planImageUpdate("0.12.3-rocm", null, "0.13.0-rocm")).toEqual({
      kind: "update",
      from: "0.12.3-rocm",
      to: "0.13.0-rocm",
    });
  });

  it("reports when the backend found no newest tag", () => {
    expect(planImageUpdate("v1.0.0", null, null)).toEqual({
      kind: "no-release",
    });
  });
});

describe("CUDA warning", () => {
  const needsCuda = {
    tag: "v0.5.21-cu130",
    required_cuda: "13.0",
    host_cuda: "12.9",
  };

  it("explains which version needs which CUDA", () => {
    expect(cudaWarning(needsCuda)).toBe(
      "v0.5.21-cu130 is newer but needs CUDA 13.0, and the NVIDIA driver on this host supports CUDA 12.9. Update the NVIDIA driver to be able to use it.",
    );
  });

  it("adds the warning to an up-to-date plan", () => {
    expect(
      planImageUpdate("v0.5.19-cu129", null, "v0.5.19-cu129", needsCuda),
    ).toEqual({
      kind: "up-to-date",
      version: "v0.5.19-cu129",
      warning: cudaWarning(needsCuda),
    });
  });

  it("adds the warning to an update plan", () => {
    expect(
      planImageUpdate("v0.5.18-cu129", null, "v0.5.19-cu129", needsCuda),
    ).toEqual({
      kind: "update",
      from: "v0.5.18-cu129",
      to: "v0.5.19-cu129",
      warning: cudaWarning(needsCuda),
    });
  });

  it("has no warning without a held-back version", () => {
    expect(planImageUpdate("v1", null, "v2", null)).toEqual({
      kind: "update",
      from: "v1",
      to: "v2",
      warning: undefined,
    });
  });
});

describe("findImageVersionField", () => {
  it("returns the docker-tags field", () => {
    const field = {
      name: "image_version",
      description: "",
      type: "docker-tags" as const,
      required: false,
    };
    expect(findImageVersionField([field])).toBe(field);
    expect(
      findImageVersionField([{ ...field, type: "text" as const }]),
    ).toBeUndefined();
    expect(findImageVersionField(undefined)).toBeUndefined();
  });
});
