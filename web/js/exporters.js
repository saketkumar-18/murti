// Browser-side mesh exporters: GLB (glTF 2.0 binary), OBJ, STL.
// Mirrors murti.export so downloaded assets are identical in structure.

export function meshToGLB(mesh, name = "murti") {
  const pos = mesh.vertices; // Float32Array
  const nrm = mesh.normals;  // Float32Array
  const idx = mesh.faces;    // Uint32Array

  const pad4 = (n) => (4 - (n % 4)) % 4;
  const posB = new Uint8Array(pos.buffer, pos.byteOffset, pos.byteLength);
  const nrmB = new Uint8Array(nrm.buffer, nrm.byteOffset, nrm.byteLength);
  const idxB = new Uint8Array(idx.buffer, idx.byteOffset, idx.byteLength);

  const posPad = pad4(posB.length), nrmPad = pad4(nrmB.length), idxPad = pad4(idxB.length);
  const binLen = posB.length + posPad + nrmB.length + nrmPad + idxB.length + idxPad;
  const bin = new Uint8Array(binLen);
  let off = 0;
  bin.set(posB, off); off += posB.length + posPad;
  const nrmOff = off;
  bin.set(nrmB, off); off += nrmB.length + nrmPad;
  const idxOff = off;
  bin.set(idxB, off);

  // min/max for POSITION accessor
  const vmin = [Infinity, Infinity, Infinity], vmax = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      vmin[k] = Math.min(vmin[k], pos[i + k]);
      vmax[k] = Math.max(vmax[k], pos[i + k]);
    }
  }

  const gltf = {
    asset: { version: "2.0", generator: "Murti 1.0 (browser)" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, indices: 2, mode: 4 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: pos.length / 3, type: "VEC3", min: vmin, max: vmax },
      { bufferView: 1, componentType: 5126, count: nrm.length / 3, type: "VEC3" },
      { bufferView: 2, componentType: 5125, count: idx.length, type: "SCALAR" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: posB.length, target: 34962 },
      { buffer: 0, byteOffset: nrmOff, byteLength: nrmB.length, target: 34962 },
      { buffer: 0, byteOffset: idxOff, byteLength: idxB.length, target: 34963 },
    ],
    buffers: [{ byteLength: binLen }],
  };

  const jsonStr = JSON.stringify(gltf);
  const jsonEnc = new TextEncoder().encode(jsonStr);
  const jsonPad = pad4(jsonEnc.length);
  const jsonChunk = new Uint8Array(jsonEnc.length + jsonPad);
  jsonChunk.set(jsonEnc);
  for (let i = jsonEnc.length; i < jsonChunk.length; i++) jsonChunk[i] = 0x20; // space

  const total = 12 + 8 + jsonChunk.length + 8 + bin.length;
  const out = new ArrayBuffer(total);
  const dv = new DataView(out);
  const u8 = new Uint8Array(out);
  let p = 0;
  dv.setUint32(p, 0x46546c67, true); p += 4; // "glTF"
  dv.setUint32(p, 2, true); p += 4;
  dv.setUint32(p, total, true); p += 4;
  dv.setUint32(p, jsonChunk.length, true); p += 4;
  dv.setUint32(p, 0x4e4f534a, true); p += 4; // "JSON"
  u8.set(jsonChunk, p); p += jsonChunk.length;
  dv.setUint32(p, bin.length, true); p += 4;
  dv.setUint32(p, 0x004e4942, true); p += 4; // "BIN\0"
  u8.set(bin, p);
  return new Blob([out], { type: "model/gltf-binary" });
}

export function meshToOBJ(mesh, name = "murti") {
  const lines = [`# Murti generated mesh: ${mesh.vertices.length / 3} verts, ${mesh.faces.length / 3} faces`, `o ${name}`];
  for (let i = 0; i < mesh.vertices.length; i += 3) {
    lines.push(`v ${mesh.vertices[i].toFixed(6)} ${mesh.vertices[i + 1].toFixed(6)} ${mesh.vertices[i + 2].toFixed(6)}`);
  }
  for (let i = 0; i < mesh.normals.length; i += 3) {
    lines.push(`vn ${mesh.normals[i].toFixed(6)} ${mesh.normals[i + 1].toFixed(6)} ${mesh.normals[i + 2].toFixed(6)}`);
  }
  for (let i = 0; i < mesh.faces.length; i += 3) {
    const a = mesh.faces[i] + 1, b = mesh.faces[i + 1] + 1, c = mesh.faces[i + 2] + 1;
    lines.push(`f ${a}//${a} ${b}//${b} ${c}//${c}`);
  }
  return new Blob([lines.join("\n") + "\n"], { type: "text/plain" });
}

export function meshToSTL(mesh) {
  const nFaces = mesh.faces.length / 3;
  const buf = new ArrayBuffer(84 + nFaces * 50);
  const dv = new DataView(buf);
  dv.setUint32(80, nFaces, true);
  let p = 84;
  for (let i = 0; i < mesh.faces.length; i += 3) {
    const a = mesh.faces[i] * 3, b = mesh.faces[i + 1] * 3, c = mesh.faces[i + 2] * 3;
    const e1 = [mesh.vertices[b] - mesh.vertices[a], mesh.vertices[b + 1] - mesh.vertices[a + 1], mesh.vertices[b + 2] - mesh.vertices[a + 2]];
    const e2 = [mesh.vertices[c] - mesh.vertices[a], mesh.vertices[c + 1] - mesh.vertices[a + 1], mesh.vertices[c + 2] - mesh.vertices[a + 2]];
    let nx = e1[1] * e2[2] - e1[2] * e2[1];
    let ny = e1[2] * e2[0] - e1[0] * e2[2];
    let nz = e1[0] * e2[1] - e1[1] * e2[0];
    const l = Math.hypot(nx, ny, nz) || 1;
    dv.setFloat32(p, nx / l, true); dv.setFloat32(p + 4, ny / l, true); dv.setFloat32(p + 8, nz / l, true);
    for (let k = 0; k < 3; k++) {
      const vi = mesh.faces[i + k] * 3;
      dv.setFloat32(p + 12 + k * 12, mesh.vertices[vi], true);
      dv.setFloat32(p + 16 + k * 12, mesh.vertices[vi + 1], true);
      dv.setFloat32(p + 20 + k * 12, mesh.vertices[vi + 2], true);
    }
    dv.setUint16(p + 48, 0, true);
    p += 50;
  }
  return new Blob([buf], { type: "model/stl" });
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}
