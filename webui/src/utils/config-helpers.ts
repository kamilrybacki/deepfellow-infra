/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/

// A dynamic field's raw value is serialized with Python's str(bool), so this is how
// booleans are told apart from other editable fields (see design.md decision #2).
export function isBooleanValue(value: string): boolean {
  return value === "True" || value === "False";
}
