// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

// WebXR drawing shared by /headset and /train: textured quads drawn in both eyes.
// Page scripts share one global scope, so every name here starts with xr.

function xrMatMul(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++)
    for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return out;
}

function xrTranslateScale(x, y, z, sx = 1, sy = 1) {
  return new Float32Array([sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
}

// A room-fixed placement `distance` m ahead of the viewer, level and facing them: only the head's yaw counts, so a
// screen placed while looking down still stands upright at eye height.
// ponytail: the yaw is read from the gaze direction, which is noisy when looking almost straight down. Upgrade: take
// the yaw from the head's up vector (m[4], m[6]) when |forward.y| is large.
function xrPlaceAhead(viewerMatrix, distance) {
  const m = viewerMatrix;
  const n = Math.hypot(m[8], m[10]);
  const [fx, fz] = n > 1e-6 ? [-m[8] / n, -m[10] / n] : [0, -1];   // straight up or down: no yaw, face -z
  return new Float32Array([
    -fz, 0, fx, 0,  0, 1, 0, 0,  -fx, 0, -fz, 0,
    m[12] + fx * distance, m[13], m[14] + fz * distance, 1,
  ]);
}

function xrShader(gl, type, source) {
  const item = gl.createShader(type);
  gl.shaderSource(item, source);
  gl.compileShader(item);
  if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(item));
  return item;
}

function xrMakeQuad(gl) {
  const program = gl.createProgram();
  gl.attachShader(program, xrShader(gl, gl.VERTEX_SHADER, `
    attribute vec2 point; attribute vec2 uv; uniform mat4 transform;
    varying vec2 texcoord;
    void main() { texcoord = uv; gl_Position = transform * vec4(point, 0.0, 1.0); }
  `));
  gl.attachShader(program, xrShader(gl, gl.FRAGMENT_SHADER, `
    precision mediump float; varying vec2 texcoord; uniform sampler2D picture;
    void main() { gl_FragColor = texture2D(picture, texcoord); }
  `));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -0.50, -0.32, 0, 1,  0.50, -0.32, 1, 1,
    -0.50,  0.32, 0, 0,  0.50,  0.32, 1, 0,
  ]), gl.STATIC_DRAW);
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
  return { program, buffer, texture,
    point: gl.getAttribLocation(program, "point"),
    uv: gl.getAttribLocation(program, "uv"),
    transform: gl.getUniformLocation(program, "transform"),
    picture: gl.getUniformLocation(program, "picture") };
}

function xrSetImage(gl, quad, source) {
  gl.bindTexture(gl.TEXTURE_2D, quad.texture);
  try { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source); }
  catch (_) { /* Keep the previous image during a decoder transition. */ }
}

function xrDrawQuad(gl, quad, model, viewer, layer) {
  gl.useProgram(quad.program);
  gl.bindBuffer(gl.ARRAY_BUFFER, quad.buffer);
  gl.enableVertexAttribArray(quad.point);
  gl.vertexAttribPointer(quad.point, 2, gl.FLOAT, false, 16, 0);
  gl.enableVertexAttribArray(quad.uv);
  gl.vertexAttribPointer(quad.uv, 2, gl.FLOAT, false, 16, 8);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, quad.texture);
  gl.uniform1i(quad.picture, 0);
  for (const view of viewer.views) {
    const viewport = layer.getViewport(view);
    gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
    gl.uniformMatrix4fv(quad.transform, false,
      xrMatMul(view.projectionMatrix, xrMatMul(view.transform.inverse.matrix, model)));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

function xrTextCanvas(canvas, lines, { align = "center", font = 40 } = {}) {
  const context = canvas.getContext("2d");
  context.fillStyle = "#10131a";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#e6e9ef";
  context.font = `${font}px system-ui, sans-serif`;
  context.textAlign = align;
  context.textBaseline = "middle";
  const lineHeight = font * 1.25;
  const top = (canvas.height - lineHeight * (lines.length - 1)) / 2;
  const x = align === "center" ? canvas.width / 2 : 24;
  lines.forEach((line, i) => context.fillText(line, x, top + i * lineHeight, canvas.width - 48));
}

async function xrStartSession() {
  const session = await navigator.xr.requestSession("immersive-vr", { optionalFeatures: ["local-floor"] });
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl", { xrCompatible: true, alpha: false });
    if (!gl) throw new Error("WebGL unavailable");
    await gl.makeXRCompatible();
    session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
    const space = await session.requestReferenceSpace("local-floor")
      .catch(() => session.requestReferenceSpace("local"));
    return { session, gl, space };
  } catch (error) {
    await session.end();
    throw error;
  }
}

// Bind and clear this frame's layer; the viewer pose is null while head tracking is lost.
function xrBeginFrame(gl, session, space, frame) {
  const layer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  return frame.getViewerPose(space);
}
