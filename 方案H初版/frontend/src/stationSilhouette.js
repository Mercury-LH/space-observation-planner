// Convex outline of the dish, receiver and pedestal in the 1672 × 941 source.
// Pixel coordinates are clockwise in the photograph's top-left coordinate system.
const outline = [
  [184, 455], [193, 442], [212, 436], [241, 435], [464, 430], [596, 560],
  [607, 577], [611, 595], [607, 615], [489, 796], [221, 793], [184, 476],
];

export const stationEdges = outline.map(([x, y], index) => {
  const [nextX, nextY] = outline[(index + 1) % outline.length];
  const length = Math.hypot(nextX - x, nextY - y);
  const normalX = (nextY - y) / length;
  const normalY = (x - nextX) / length;
  return { normalX, normalY, offset: normalX * x + normalY * y };
});

// Each edge is a half plane. Outside corners this conservatively includes
// a little extra sky, so linear filtering cannot pull a bright rim into it.
export function stationEdgeDistance(x, y) {
  let distance = -Infinity;
  for (const { normalX, normalY, offset } of stationEdges) {
    distance = Math.max(distance, normalX * x + normalY * y - offset);
  }
  return distance;
}

export function eraseStation(ctx, width, height, padding, feather, top = 0) {
  ctx.save();
  ctx.translate(0, -top);
  ctx.scale(width / 1672, height / 941);
  ctx.globalCompositeOperation = "destination-out";
  ctx.fillStyle = "#000";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(...outline[0]);
  for (const [x, y] of outline.slice(1)) ctx.lineTo(x, y);
  ctx.closePath();
  ctx.fill();
  // One-time mask preparation. These nested outlines give a linear alpha
  // ramp with no filter dependency on the Canvas 2D fallback.
  const steps = Math.ceil(feather);
  for (let step = steps; step >= 0; step -= 1) {
    ctx.lineWidth = 2 * (padding + step);
    ctx.strokeStyle = `rgba(0,0,0,${1 / (step + 1)})`;
    ctx.stroke();
  }
  ctx.restore();
}
