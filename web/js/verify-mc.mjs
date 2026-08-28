// Verify the browser marching cubes against a known sphere:
// mesh must be watertight (every edge shared by exactly 2 faces) and
// have sphere topology (Euler characteristic 2).
import { marchingCubes } from "./marching-cubes.js";

const grid = 32;
const vol = new Float32Array(grid * grid * grid);
for (let z = 0; z < grid; z++)
  for (let y = 0; y < grid; y++)
    for (let x = 0; x < grid; x++) {
      const px = (x / (grid - 1)) * 2 - 1, py = (y / (grid - 1)) * 2 - 1, pz = (z / (grid - 1)) * 2 - 1;
      vol[x + y * grid + z * grid * grid] = Math.hypot(px, py, pz) < 0.5 ? 1 : 0;
    }

const mesh = marchingCubes(vol, grid, 0.5, 1.0);
const V = mesh.vertices.length / 3, F = mesh.faces.length / 3;
console.log(`vertices=${V} faces=${F}`);

// edge counts
const edgeCount = new Map();
for (let i = 0; i < mesh.faces.length; i += 3) {
  const a = mesh.faces[i], b = mesh.faces[i + 1], c = mesh.faces[i + 2];
  for (const [p, q] of [[a, b], [b, c], [c, a]]) {
    const key = Math.min(p, q) + "_" + Math.max(p, q);
    edgeCount.set(key, (edgeCount.get(key) ?? 0) + 1);
  }
}
let boundary = 0, nonManifold = 0;
for (const c of edgeCount.values()) {
  if (c === 1) boundary++;
  if (c > 2) nonManifold++;
}
const E = edgeCount.size;
const euler = V - E + F;
console.log(`edges=${E} boundary=${boundary} nonManifold=${nonManifold} euler=${euler}`);

// bounds check: sphere r=0.5 in [-1,1] -> world r=0.25
let maxAbs = 0;
for (let i = 0; i < mesh.vertices.length; i++) maxAbs = Math.max(maxAbs, Math.abs(mesh.vertices[i]));
console.log(`maxAbsCoord=${maxAbs.toFixed(4)} (expect ~0.25)`);

// normals outward check
let outward = 0;
for (let i = 0; i < V; i++) {
  const dot = mesh.normals[i*3]*mesh.vertices[i*3] + mesh.normals[i*3+1]*mesh.vertices[i*3+1] + mesh.normals[i*3+2]*mesh.vertices[i*3+2];
  if (dot > 0) outward++;
}
console.log(`outwardNormals=${(outward / V * 100).toFixed(1)}%`);

const ok = boundary === 0 && nonManifold === 0 && euler === 2 && maxAbs < 0.3 && outward / V > 0.95;
console.log(ok ? "MC_PASS" : "MC_FAIL");
process.exit(ok ? 0 : 1);
