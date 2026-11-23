import { stationEdges, stationEdgeDistance } from "./stationSilhouette";

// This renderer owns GPU resources only. The caller owns sizing, the clock,
// visibility, and the static image underneath the canvas.
const VERTEX_SHADER = `
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
  // Image coordinates throughout this module: (0, 0) is the TOP LEFT.
  v_uv = vec2(a_position.x * 0.5 + 0.5, 0.5 - a_position.y * 0.5);
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

// Precompute the outline's line equations so each GPU distance check only
// needs a dot product and maximum per edge, in the same space as Canvas 2D.
const stationDistanceShader = stationEdges.map(({ normalX, normalY, offset }) =>
  `distanceToEdge = max(distanceToEdge, dot(point, vec2(${normalX.toFixed(8)}, ${normalY.toFixed(8)})) - ${offset.toFixed(6)});`
).join("\n");

const FRAGMENT_SHADER = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D u_image;
uniform float u_seconds;
varying vec2 v_uv;

// Coordinates are normalized against the 1672 × 941 radio observatory scene.
// Keep these landmarks fixed if the display size or its cover crop changes.
const vec2 SOURCE_SIZE = vec2(1672.0, 941.0);
const float SKY_PERIOD = 24.0;
// The off-frame pivot follows the photograph's existing curved star trails.
const vec2 SKY_PIVOT = vec2(0.90, 1.27);
const float SKY_ANGLE = 0.095;
const float GRASS_SWAY_PX = 23.0;

float ellipseMask(vec2 p, vec2 center, vec2 radius) {
  float distanceToEdge = (length((p - center) / radius) - 1.0)
    * min(radius.x * SOURCE_SIZE.x, radius.y * SOURCE_SIZE.y);
  return 1.0 - smoothstep(0.0, 7.0, distanceToEdge);
}

float stationDistance(vec2 uv) {
  vec2 point = uv * SOURCE_SIZE;
  float distanceToEdge = -2048.0;
  ${stationDistanceShader}
  return distanceToEdge;
}

float skySourceAllowed(vec2 p) {
  float sourceHorizon = 0.807 + 0.053 * smoothstep(0.0, 0.58, p.x);
  // Check the SOURCE coordinate too: a safe destination alone cannot stop a
  // rotated dish or ridge from being copied upwards into apparently empty sky.
  return smoothstep(8.0, 16.0, stationDistance(p))
    * (1.0 - smoothstep(sourceHorizon - 0.038, sourceHorizon - 0.022, p.y));
}

float rockMask(vec2 p) {
  float mask = ellipseMask(p, vec2(0.106, 0.961), vec2(0.090, 0.065));
  mask = max(mask, ellipseMask(p, vec2(0.451, 0.897), vec2(0.035, 0.024)));
  mask = max(mask, ellipseMask(p, vec2(0.552, 0.893), vec2(0.027, 0.015)));
  mask = max(mask, ellipseMask(p, vec2(0.337, 0.844), vec2(0.044, 0.018)));
  mask = max(mask, ellipseMask(p, vec2(0.182, 0.864), vec2(0.027, 0.014)));
  mask = max(mask, ellipseMask(p, vec2(0.127, 0.821), vec2(0.040, 0.025)));
  mask = max(mask, ellipseMask(p, vec2(0.059, 0.813), vec2(0.034, 0.017)));
  return mask;
}

vec4 sourceAt(vec2 p) {
  // No repeat/tiling: never introduce a seam or a second landscape.
  return texture2D(u_image, clamp(p, vec2(0.0), vec2(1.0)));
}

vec2 skyAtPhase(vec2 p, float phase, float mask) {
  vec2 relative = (p - SKY_PIVOT) * SOURCE_SIZE;
  float angle = (phase - 0.5) * SKY_ANGLE * mask;
  float c = cos(angle);
  float s = sin(angle);
  return SKY_PIVOT + vec2(c * relative.x - s * relative.y,
    s * relative.x + c * relative.y) / SOURCE_SIZE;
}

float grassClump(vec2 p, vec2 root, vec2 tip, float width, float phase) {
  // Each narrow clump bends about its own fixed root. The surrounding ground
  // is never assigned a broad wave/displacement field.
  float along = clamp((root.y - p.y) / (root.y - tip.y), 0.0, 1.0);
  float axis = mix(root.x, tip.x, along);
  float side = 1.0 - smoothstep(width * 0.28, width, abs(p.x - axis));
  float ends = smoothstep(tip.y - 0.013, tip.y + 0.011, p.y)
    * (1.0 - smoothstep(root.y - 0.013, root.y, p.y));
  float wind = 0.76 * sin(u_seconds * 1.72 + phase)
    + 0.24 * sin(u_seconds * 2.71 + phase * 1.63);
  float gust = 0.88 + 0.12 * sin(u_seconds * 0.61 + phase);
  return side * ends * pow(along, 1.35) * wind * gust;
}

void main() {
  vec2 p = v_uv;
  vec4 original = sourceAt(p);
  float fromStation = stationDistance(p);
  float fixedBuilding = 1.0 - smoothstep(4.0, 12.0, fromStation);
  float horizon = 0.807 + 0.053 * smoothstep(0.0, 0.58, p.x);
  // Ease sky displacement to zero before reaching the receiver or reflector.
  float sky = (1.0 - smoothstep(horizon - 0.125, horizon - 0.095, p.y))
    * smoothstep(16.0, 80.0, fromStation);
  float bandCenter = 0.565 - 0.655 * p.y + 0.075 * p.y * p.y;
  float bandDistance = abs(p.x - bandCenter);
  float band = 1.0 - smoothstep(0.175, 0.370, bandDistance);
  float frameEdge = smoothstep(0.0, 0.024, p.x)
    * (1.0 - smoothstep(0.976, 1.0, p.x))
    * smoothstep(0.0, 0.030, p.y);
  float skyMotion = sky * band * frameEdge * (1.0 - fixedBuilding);

  // Two copies advect in the SAME direction; each is reset only at zero
  // weight. This keeps the finite photograph flowing without a visible wrap,
  // a ping-pong reversal, or translating the entire sky as one flat sheet.
  // At t = 0 the visible phase samples the original image exactly.
  float phaseA = fract(u_seconds / SKY_PERIOD);
  float phaseB = fract(u_seconds / SKY_PERIOD + 0.5);
  float weightA = 1.0 - abs(phaseA * 2.0 - 1.0);
  weightA = weightA * weightA * (3.0 - 2.0 * weightA);
  vec2 sampleA = skyAtPhase(p, phaseA, skyMotion);
  vec2 sampleB = skyAtPhase(p, phaseB, skyMotion);
  vec4 skyA = mix(original, sourceAt(sampleA), skySourceAllowed(sampleA));
  vec4 skyB = mix(original, sourceAt(sampleB), skySourceAllowed(sampleB));
  vec4 skyColor = mix(skyB, skyA, weightA);

  // Include the middle foreground: on mobile the very bottom of the photo is
  // cropped away. Its visible grass tips must not inherit a near-zero depth.
  float sway = grassClump(p, vec2(0.002, 1.020), vec2(0.044, 0.727), 0.031, 0.0);
  sway += grassClump(p, vec2(0.065, 1.020), vec2(0.098, 0.849), 0.026, 1.2);
  sway += grassClump(p, vec2(0.236, 0.977), vec2(0.279, 0.856), 0.025, 0.4);
  sway += grassClump(p, vec2(0.302, 0.991), vec2(0.340, 0.866), 0.027, 1.5);
  sway += grassClump(p, vec2(0.368, 1.014), vec2(0.402, 0.871), 0.028, 2.6);
  sway += grassClump(p, vec2(0.430, 1.020), vec2(0.476, 0.885), 0.026, 3.7);
  sway += grassClump(p, vec2(0.514, 1.020), vec2(0.550, 0.865), 0.024, 4.5);
  sway += grassClump(p, vec2(0.534, 1.025), vec2(0.625, 0.823), 0.032, 5.2);
  sway += grassClump(p, vec2(0.618, 1.030), vec2(0.683, 0.866), 0.030, 0.9);
  sway += grassClump(p, vec2(0.735, 1.035), vec2(0.797, 0.883), 0.034, 2.1);
  sway += grassClump(p, vec2(0.863, 1.035), vec2(0.910, 0.889), 0.033, 3.2);
  float clearOfHorizon = smoothstep(0.009, 0.024, abs(p.y - horizon));
  float protection = (1.0 - fixedBuilding) * (1.0 - rockMask(p)) * clearOfHorizon;
  float start = smoothstep(0.0, 0.22, u_seconds);
  vec2 bend = vec2(sway, sway * -0.045) * GRASS_SWAY_PX / SOURCE_SIZE
    * protection * start;
  vec4 grassColor = sourceAt(p + bend);

  // The shared source has a local galaxy exposure lift. Architecture and
  // foreground retain their original values; this pass adds no further grade.
  vec4 color = mix(original, skyColor, sky);
  color.rgb += grassColor.rgb - original.rgb;
  gl_FragColor = vec4(color.rgb, 1.0);
}
`;

/**
 * Prepare the same local galaxy exposure for both rendering paths, once per
 * initialization. The source file is unchanged; no per-frame pixel readback.
 */
export function createGalaxyExposureSource(image, ownerDocument) {
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  let target;
  try {
    target = ownerDocument
      ? ownerDocument.createElement("canvas")
      : new OffscreenCanvas(width, height);
    target.width = width;
    target.height = height;
    const context = target.getContext("2d", { willReadFrequently: true });
    if (!context) return image;
    context.drawImage(image, 0, 0, width, height);
    const pixels = context.getImageData(0, 0, width, height);
    const data = pixels.data;
    const smoothstep = (lower, upper, value) => {
      const t = Math.max(0, Math.min(1, (value - lower) / (upper - lower)));
      return t * t * (3 - 2 * t);
    };
    for (let y = 0; y < Math.ceil(height * 0.58); y += 1) {
      const py = (y + 0.5) / height;
      const bandCenter = 0.565 - 0.655 * py + 0.075 * py * py;
      // Exposure is independent of the larger MOTION exclusion region. Its
      // broad vertical fade avoids outlining that region as a dark cutout.
      const skyFade = 1 - smoothstep(0.37, 0.58, py);
      for (let x = 0; x < width; x += 1) {
        const px = (x + 0.5) / width;
        const band = 1 - smoothstep(0.105, 0.285, Math.abs(px - bandCenter));
        if (!band) continue;
        const photoY = py * 941;
        // Everything above 64px clear of the receiver takes the full exposure.
        const clear = photoY < 366 ? 1 :
          smoothstep(4, 64, stationEdgeDistance(px * 1672, photoY));
        const amount = band * skyFade * clear;
        if (amount <= 0) continue;
        const index = (y * width + x) * 4;
        const red = data[index] / 255;
        const green = data[index + 1] / 255;
        const blue = data[index + 2] / 255;
        const luminance = red * 0.2126 + green * 0.7152 + blue * 0.0722;
        const peak = Math.max(red, green, blue);
        if (!peak) continue;
        // The dusty middle tones and stars receive more exposure than empty
        // dark sky. A highlight shoulder preserves color ratios and detail.
        const gain = 1 + amount * (0.4 + 0.8 * smoothstep(0.025, 0.12, luminance));
        const scale = (1 - Math.pow(1 - peak, gain)) / peak;
        data[index] = Math.round(data[index] * scale);
        data[index + 1] = Math.round(data[index + 1] * scale);
        data[index + 2] = Math.round(data[index + 2] * scale);
      }
    }
    context.putImageData(pixels, 0, 0);
    return target;
  } catch {
    // For a tainted image or allocation failure, retain the existing visual.
    if (target) { target.width = 1; target.height = 1; }
    return image;
  }
}

/**
 * Create a renderer for a decoded HTMLImageElement/ImageBitmap.
 * draw(seconds) renders the whole image into the current canvas dimensions.
 * Returns null if WebGL, shader compilation, or texture upload is unavailable.
 */
export function createLandscapeRenderer(canvas, loadedImage) {
  if (!canvas || typeof canvas.getContext !== "function" || !loadedImage) return null;
  const imageWidth = loadedImage.naturalWidth ?? loadedImage.width;
  const imageHeight = loadedImage.naturalHeight ?? loadedImage.height;
  if (!(imageWidth > 0 && imageHeight > 0)) return null;

  let gl;
  let program;
  let buffer;
  let texture;
  const shaders = [];
  let destroyed = false;

  function destroy() {
    if (destroyed) return;
    destroyed = true;
    if (!gl) return;
    if (texture) gl.deleteTexture(texture);
    if (buffer) gl.deleteBuffer(buffer);
    if (program) gl.deleteProgram(program);
    shaders.forEach((shader) => gl.deleteShader(shader));
  }

  try {
    gl = canvas.getContext("webgl", {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      powerPreference: "low-power",
    });
    if (!gl || gl.isContextLost()) return null;
    const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    if (imageWidth > maxSize || imageHeight > maxSize) return null;

    function compile(type, source) {
      const shader = gl.createShader(type);
      if (!shader) throw new Error("Shader allocation failed");
      shaders.push(shader);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader) || "Shader compilation failed");
      }
      return shader;
    }

    const vertex = compile(gl.VERTEX_SHADER, VERTEX_SHADER);
    const fragment = compile(gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    program = gl.createProgram();
    if (!program) throw new Error("Program allocation failed");
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || "Program linking failed");
    }
    gl.useProgram(program);

    buffer = gl.createBuffer();
    texture = gl.createTexture();
    if (!buffer || !texture) throw new Error("GPU resource allocation failed");
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const position = gl.getAttribLocation(program, "a_position");
    if (position < 0) throw new Error("Position attribute unavailable");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    // The shader deliberately uses top-left image UVs; no upload flip needed.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    const exposedImage = createGalaxyExposureSource(loadedImage, canvas.ownerDocument);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, exposedImage);
    gl.uniform1i(gl.getUniformLocation(program, "u_image"), 0);
    const time = gl.getUniformLocation(program, "u_seconds");
    if (time === null || gl.getError() !== gl.NO_ERROR) {
      throw new Error("Texture upload or uniform initialization failed");
    }

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.DITHER);

    function draw(seconds) {
      if (destroyed || gl.isContextLost()) return false;
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      gl.useProgram(program);
      gl.uniform1f(time, Number.isFinite(seconds) ? Math.max(0, seconds) : 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      return true;
    }

    draw(0);
    return { draw, destroy };
  } catch {
    destroy();
    return null;
  }
}
