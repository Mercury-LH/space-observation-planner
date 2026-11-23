import { eraseStation } from "./stationSilhouette";
import { createGalaxyExposureSource } from "./landscapeRenderer";

/**
 * Small Canvas2D cinemagraph for the 1672 × 941 observatory photograph.
 * Coordinates stay in photograph space; the caller owns sizing and scheduling.
 * The photograph file is unchanged; exposure and masks are prepared once.
 */
export function createLandscapeFallback(canvas, image) {
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) return null;

  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  if (!sourceWidth || !sourceHeight) {
    throw new Error("The landscape image must finish loading first");
  }

  // Cap intermediate surfaces independently of devicePixelRatio. The static
  // photograph still uses its original resolution on the visible canvas.
  const width = Math.min(1280, sourceWidth);
  const height = Math.round((width * sourceHeight) / sourceWidth);
  const surfaces = [];
  let destroyed = false;
  const exposedImage = createGalaxyExposureSource(image, canvas.ownerDocument);
  if (exposedImage !== image) surfaces.push(exposedImage);

  function surface(w, h) {
    const ownerDocument = canvas.ownerDocument;
    const buffer = ownerDocument
      ? ownerDocument.createElement("canvas")
      : new OffscreenCanvas(w, h);
    buffer.width = Math.max(1, Math.ceil(w));
    buffer.height = Math.max(1, Math.ceil(h));
    const ctx = buffer.getContext("2d");
    if (!ctx) throw new Error("Cannot allocate a landscape buffer");
    surfaces.push(buffer);
    return { canvas: buffer, ctx };
  }

  // A solid inner ellipse protects the architecture/rocks; the outer edge
  // softly returns to animation, rather than creating a visible cutout seam.
  function protectEllipse(ctx, x, y, radiusX, radiusY, inner = 0.8) {
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(radiusX, radiusY);
    const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
    gradient.addColorStop(0, "rgba(0,0,0,1)");
    gradient.addColorStop(inner, "rgba(0,0,0,1)");
    gradient.addColorStop(1, "rgba(0,0,0,0)");
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillStyle = gradient;
    ctx.fillRect(-1, -1, 2, 2);
    ctx.restore();
  }

  const skyMask = surface(width, height);
  const skyGradient = skyMask.ctx.createLinearGradient(0, 0, 0, height);
  skyGradient.addColorStop(0, "rgba(255,255,255,0)");
  skyGradient.addColorStop(0.045, "#fff");
  skyGradient.addColorStop(0.65, "#fff");
  skyGradient.addColorStop(0.72, "rgba(255,255,255,0)");
  skyGradient.addColorStop(1, "rgba(255,255,255,0)");
  skyMask.ctx.fillStyle = skyGradient;
  skyMask.ctx.fillRect(0, 0, width, height);
  // A rotating finite photograph must fade back to the original at its frame
  // edges, otherwise an exposed triangular corner can reveal the moving crop.
  skyMask.ctx.globalCompositeOperation = "destination-in";
  const skyEdges = skyMask.ctx.createLinearGradient(0, 0, width, 0);
  skyEdges.addColorStop(0, "rgba(255,255,255,0)");
  skyEdges.addColorStop(0.045, "#fff");
  skyEdges.addColorStop(0.955, "#fff");
  skyEdges.addColorStop(1, "rgba(255,255,255,0)");
  skyMask.ctx.fillStyle = skyEdges;
  skyMask.ctx.fillRect(0, 0, width, height);
  skyMask.ctx.globalCompositeOperation = "source-over";
  eraseStation(skyMask.ctx, width, height, 16, 64);
  // Remove all architecture/terrain from the SOURCE before rotating it.
  // The expanded fixed destination guard above is a separate second defense.
  const skySource = surface(width, height);
  skySource.ctx.drawImage(exposedImage, 0, 0, width, height);
  skySource.ctx.globalCompositeOperation = "destination-in";
  const sourceFade = skySource.ctx.createLinearGradient(0, 0, 0, height);
  sourceFade.addColorStop(0, "#fff");
  sourceFade.addColorStop(0.71, "#fff");
  sourceFade.addColorStop(0.77, "rgba(255,255,255,0)");
  sourceFade.addColorStop(1, "rgba(255,255,255,0)");
  skySource.ctx.fillStyle = sourceFade;
  skySource.ctx.fillRect(0, 0, width, height);
  skySource.ctx.globalCompositeOperation = "source-over";
  eraseStation(skySource.ctx, width, height, 8, 8);
  const movingSky = surface(width, height);

  // Only the near foreground moves. This fixed destination mask keeps the
  // distant horizon and prominent foreground rocks absolutely stationary,
  // even when a neighboring grass patch bends across their boundary.
  const grassTop = Math.floor(height * 0.835);
  const grassHeight = height - grassTop;
  const grassMask = surface(width, grassHeight);
  const depthFade = grassMask.ctx.createLinearGradient(0, 0, 0, grassHeight);
  depthFade.addColorStop(0, "rgba(255,255,255,0)");
  depthFade.addColorStop(0.08, "rgba(255,255,255,0)");
  depthFade.addColorStop(0.21, "#fff");
  depthFade.addColorStop(1, "#fff");
  grassMask.ctx.fillStyle = depthFade;
  grassMask.ctx.fillRect(0, 0, width, grassHeight);
  protectEllipse(
    grassMask.ctx,
    width * 0.106,
    height * 0.961 - grassTop,
    width * 0.099,
    height * 0.074,
    0.91,
  );
  protectEllipse(
    grassMask.ctx,
    width * 0.451,
    height * 0.897 - grassTop,
    width * 0.041,
    height * 0.031,
    0.87,
  );
  // Low outcrops along the left foothill, below the observatory.
  protectEllipse(
    grassMask.ctx,
    width * 0.337,
    height * 0.844 - grassTop,
    width * 0.051,
    height * 0.026,
  );
  protectEllipse(grassMask.ctx, width * 0.552, height * 0.893 - grassTop,
    width * 0.032, height * 0.021, 0.86);
  eraseStation(grassMask.ctx, width, height, 4, 8, grassTop);

  // Preserve the distant ridge as a fixed destination mask, even when a
  // moving patch's upper edge crosses it. Its grass starts 14px lower.
  grassMask.ctx.save();
  grassMask.ctx.globalCompositeOperation = "destination-out";
  grassMask.ctx.beginPath();
  grassMask.ctx.moveTo(0, 0);
  for (let step = 0; step <= 40; step += 1) {
    const x = step / 40;
    const v = Math.min(1, x / 0.58);
    const horizon = 0.807 + 0.053 * v * v * (3 - 2 * v);
    grassMask.ctx.lineTo(x * width, (horizon + 0.015) * height - grassTop);
  }
  grassMask.ctx.lineTo(width, 0);
  grassMask.ctx.closePath();
  grassMask.ctx.fillStyle = "#000";
  grassMask.ctx.fill();
  grassMask.ctx.restore();

  const movingGrass = surface(width, grassHeight);
  // Fixed roots and short, narrow clumps replace the old whole-width bands.
  // The middle foreground remains visible in the mobile photograph crop.
  const clumps = [
    [0.236, 0.977, 0.279, 0.850, 0.025, 0.4],
    [0.302, 0.991, 0.340, 0.852, 0.027, 1.5],
    [0.368, 1.014, 0.402, 0.858, 0.028, 2.6],
    [0.430, 1.020, 0.476, 0.870, 0.026, 3.7],
    [0.514, 1.020, 0.550, 0.850, 0.024, 4.5],
    [0.534, 1.025, 0.625, 0.820, 0.032, 5.2],
    [0.618, 1.030, 0.683, 0.852, 0.030, 0.9],
    [0.735, 1.035, 0.797, 0.870, 0.034, 2.1],
    [0.863, 1.035, 0.910, 0.875, 0.033, 3.2],
  ];
  const patches = clumps.map(([rootX, rootY, tipX, tipY, halfWidth, phase]) => {
    const left = width * (Math.min(rootX, tipX) - halfWidth);
    const top = height * tipY;
    const patchWidth = width * (Math.abs(tipX - rootX) + halfWidth * 2);
    const patchHeight = height * (rootY - tipY);
    const patch = surface(patchWidth, patchHeight);
    patch.ctx.drawImage(image, -left, -top, width, height);
    patch.ctx.globalCompositeOperation = "destination-in";
    const feather = patch.ctx.createLinearGradient(0, 0, patchWidth, 0);
    feather.addColorStop(0, "rgba(255,255,255,0)");
    feather.addColorStop(0.25, "#fff");
    feather.addColorStop(0.75, "#fff");
    feather.addColorStop(1, "rgba(255,255,255,0)");
    patch.ctx.fillStyle = feather;
    patch.ctx.fillRect(0, 0, patchWidth, patchHeight);
    const ends = patch.ctx.createLinearGradient(0, 0, 0, patchHeight);
    ends.addColorStop(0, "rgba(255,255,255,0)");
    ends.addColorStop(0.13, "#fff");
    ends.addColorStop(0.80, "#fff");
    ends.addColorStop(1, "rgba(255,255,255,0)");
    patch.ctx.fillStyle = ends;
    patch.ctx.fillRect(0, 0, patchWidth, patchHeight);
    patch.ctx.globalCompositeOperation = "source-over";
    return { ...patch, left, top, patchHeight, phase };
  });

  function draw(seconds = 0) {
    if (destroyed || !canvas.width || !canvas.height) return;
    const time = Number.isFinite(seconds) ? seconds : 0;
    const sky = movingSky.ctx;
    sky.setTransform(1, 0, 0, 1, 0, 0);
    sky.clearRect(0, 0, width, height);
    // Two overlapping phases both flow right/up. Each phase wraps only while
    // fully transparent; complementary weights avoid a visible reset or a
    // global brightness pulse. At t=0 the visible phase is the unscaled photo.
    const cycle = 24;
    const phase = (((time / cycle + 0.5) % 1) + 1) % 1;
    sky.globalCompositeOperation = "lighter";
    for (let layer = 0; layer < 2; layer += 1) {
      const progress = (phase + layer * 0.5) % 1;
      const weight = Math.sin(Math.PI * progress) ** 2;
      sky.globalAlpha = weight;
      sky.save();
      sky.translate(width * 0.9, height * 1.27);
      sky.rotate(-(progress - 0.5) * 0.095);
      sky.drawImage(skySource.canvas, -width * 0.9, -height * 1.27, width, height);
      sky.restore();
    }
    sky.globalAlpha = 1;
    sky.globalCompositeOperation = "destination-in";
    sky.drawImage(skyMask.canvas, 0, 0);
    sky.globalCompositeOperation = "source-over";

    const grass = movingGrass.ctx;
    grass.setTransform(1, 0, 0, 1, 0, 0);
    grass.clearRect(0, 0, width, grassHeight);
    for (const patch of patches) {
      // A 23px source-space tip bend is visible in the cropped mobile scene.
      // The base of each clump stays anchored, with no vertical ground ripple.
      const start = Math.min(1, Math.max(0, time / 0.22));
      const wind = (Math.sin(time * 1.72 + patch.phase) * 0.76
        + Math.sin(time * 2.71 + patch.phase * 1.63) * 0.24)
        * (0.88 + 0.12 * Math.sin(time * 0.61 + patch.phase))
        * 23 * width / sourceWidth / patch.patchHeight * start;
      grass.save();
      grass.translate(patch.left, patch.top + patch.patchHeight - grassTop);
      grass.transform(1, 0, wind, 1, 0, 0);
      grass.drawImage(patch.canvas, 0, -patch.patchHeight);
      grass.restore();
    }
    grass.globalCompositeOperation = "destination-in";
    grass.drawImage(grassMask.canvas, 0, 0);
    grass.globalCompositeOperation = "source-over";

    context.save();
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.globalCompositeOperation = "source-over";
    context.globalAlpha = 1;
    context.drawImage(exposedImage, 0, 0, canvas.width, canvas.height);
    context.setTransform(canvas.width / width, 0, 0, canvas.height / height, 0, 0);
    context.drawImage(movingSky.canvas, 0, 0);
    context.drawImage(movingGrass.canvas, 0, grassTop);
    context.restore();
  }

  function destroy() {
    if (destroyed) return;
    destroyed = true;
    for (const buffer of surfaces) {
      buffer.width = 1;
      buffer.height = 1;
    }
    surfaces.length = 0;
    patches.length = 0;
  }

  return { draw, destroy };
}
