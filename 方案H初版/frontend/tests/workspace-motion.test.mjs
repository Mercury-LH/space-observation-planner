import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { transform } from "esbuild";
import { cssTimeToMilliseconds } from "../src/cssTime.js";

test("CSS seconds and milliseconds produce the same Web Animations duration", () => {
  for (const [value, expected] of [
    ["900ms", 900], [" .9s ", 900], ["600ms", 600], ["0.6s", 600],
    ["620ms", 620], [".62s", 620], ["500ms", 500], [".5s", 500],
    ["0s", 0], ["1.5s", 1500],
  ]) assert.equal(cssTimeToMilliseconds(value, 900), expected, value);
  for (const value of ["", "auto", "NaN", "-1s", "900", "0.9invalids"])
    assert.equal(cssTimeToMilliseconds(value, 900), 900, value);
});

test("minifying the real drawer CSS preserves normal and reduced flight timings", async () => {
  const css = await readFile(new URL("../src/WorkspaceDrawer.css", import.meta.url), "utf8");
  const { code } = await transform(css, { loader: "css", minify: true });
  const durations = source => [...source.matchAll(/--drawer-(enter|exit)-duration:\s*([^;}]+)/g)]
    .map(([, direction, value]) => [direction, cssTimeToMilliseconds(value, NaN)]);
  assert.deepEqual(durations(css), [["enter", 900], ["exit", 600], ["enter", 620], ["exit", 500]]);
  assert.deepEqual(durations(code), durations(css));
});
