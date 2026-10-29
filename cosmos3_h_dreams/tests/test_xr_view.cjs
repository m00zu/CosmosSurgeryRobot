// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Run with: node cosmos3_h_dreams/tests/test_xr_view.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const context = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../cosmos3_h_dreams/web/xr_view.js"), "utf8"), context);
const { xrPlaceAhead, xrTranslateScale, xrMatMul } = context;
const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const closeAll = (a, b) => b.forEach((v, i) => close(a[i], v));

// Viewer at (0, 1.6, 0) looking down -z: the screen is 1.2 m ahead, unrotated.
const ahead = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1.6, 0, 1];
closeAll(xrPlaceAhead(ahead, 1.2), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1.6, -1.2, 1]);

// Turned 90° left (looking down -x): the screen sits at -x and faces back along +x.
const left = [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1.6, 0, 1];
closeAll(xrPlaceAhead(left, 1.2), [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, -1.2, 1.6, 0, 1]);

// Looking 45° down: the screen stays level at head height.
const c = Math.SQRT1_2;
const down = [1, 0, 0, 0, 0, c, -c, 0, 0, c, c, 0, 0, 1.6, 0, 1];
closeAll(xrPlaceAhead(down, 1.2), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1.6, -1.2, 1]);

// Looking straight down: no yaw to read, so face -z rather than produce NaN.
const floor = [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 1.6, 0, 1];
closeAll(xrPlaceAhead(floor, 1.2), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1.6, -1.2, 1]);

// A panel offset composes with the placement.
const panel = xrMatMul(xrPlaceAhead(left, 1.2), xrTranslateScale(0, -0.47, 0, 1, 0.4));
closeAll(panel.slice(12), [-1.2, 1.6 - 0.47, 0, 1]);
close(panel[5], 0.4);
console.log("VR screens are placed level, ahead of the viewer, and panels follow them.");
