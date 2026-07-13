/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { describe, expect, it } from "vitest";
import { isBooleanValue } from "./config-helpers";

describe("isBooleanValue", () => {
  it.each(["True", "False"])(
    "treats Python's str(bool) %s as boolean",
    (value) => {
      expect(isBooleanValue(value)).toBe(true);
    },
  );

  it.each(["", "true", "false", "1", "0", "some-string"])(
    "treats %j as a non-boolean value",
    (value) => {
      expect(isBooleanValue(value)).toBe(false);
    },
  );
});
