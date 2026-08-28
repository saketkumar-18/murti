// Murti web app: wires inference + marching cubes + Three.js viewer + exports.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { MurtiInference, imageToVolume } from "./inference.js";
import { marchingCubes } from "./marching-cubes.js";
import { meshToGLB, meshToOBJ, meshToSTL, downloadBlob } from "./exporters.js";
import { EXAMPLE_PROMPTS } from "./tokenizer.js";

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Three.js viewer
// ---------------------------------------------------------------------------
const canvas = $("canvas3d");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(1.4, 1.1, 1.6);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.6;

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xfff2dd, 1.4);
key.position.set(2, 3, 2);
scene.add(key);
const rim = new THREE.DirectionalLight(0x88aaff, 0.7);
rim.position.set(-2, 1.5, -2);
scene.add(rim);

const gridHelper = new THREE.GridHelper(2, 20, 0x2a3346, 0x1a2030);
gridHelper.position.y = -0.62;
scene.add(gridHelper);

let meshObj = null;
let currentMesh = null;

function setMesh(mesh) {
  if (meshObj) {
    scene.remove(meshObj);
    meshObj.geometry.dispose();
    meshObj.material.dispose();
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(mesh.vertices, 3));
  geo.setAttribute("normal", new THREE.BufferAttribute(mesh.normals, 3));
  geo.setIndex(new THREE.BufferAttribute(mesh.faces, 1));
  const mat = new THREE.MeshStandardMaterial({
    color: 0xf59e0b,
    metalness: 0.15,
    roughness: 0.55,
    flatShading: false,
  });
  meshObj = new THREE.Mesh(geo, mat);
  scene.add(meshObj);
  currentMesh = mesh;
  $("empty-state").classList.add("hidden");
  ["dl-glb", "dl-obj", "dl-stl"].forEach((id) => ($(id).disabled = false));
  const nV = mesh.vertices.length / 3, nF = mesh.faces.length / 3;
  $("mesh-stats").innerHTML =
    `vertices <b>${nV.toLocaleString()}</b> · faces <b>${nF.toLocaleString()}</b>`;
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------
const inference = new MurtiInference("models");
const statusEl = $("model-status");

async function init() {
  try {
    const cfg = await inference.load((msg) => {
      statusEl.textContent = msg;
    });
    statusEl.textContent = `ready · ${cfg.eval ? "trained" : "untrained"} model`;
    statusEl.classList.add("ready");
    $("generate").disabled = false;
    $("image-generate").disabled = false;
  } catch (e) {
    console.error(e);
    statusEl.textContent = "failed to load models";
    statusEl.classList.add("error");
  }
}

function setProgress(pct, text) {
  $("progress-wrap").classList.remove("hidden");
  $("progress-fill").style.width = `${pct}%`;
  $("progress-text").textContent = text;
}
function clearProgress() {
  $("progress-wrap").classList.add("hidden");
}

async function generateFromText() {
  if (!inference.ready) return;
  const prompt = $("prompt").value.trim();
  if (!prompt) return;
  const seed = parseInt($("seed").value || "0", 10);
  const cfgScale = parseFloat($("cfg").value);
  const ddimSteps = parseInt($("steps").value, 10);

  $("generate").disabled = true;
  const t0 = performance.now();
  try {
    setProgress(2, "sampling latents…");
    const z = await inference.sample(prompt, {
      seed, cfgScale, ddimSteps,
      onStep: (i, n) => setProgress(2 + (i / n) * 80, `diffusion step ${i}/${n}`),
    });
    setProgress(85, "decoding volume…");
    const volume = await inference.decode(z);
    setProgress(92, "marching cubes…");
    const mesh = marchingCubes(volume, inference.config.grid, 0.5, 1.0);
    setMesh(mesh);
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    $("gen-time").textContent = `generated "${prompt}" in ${dt}s`;
    setProgress(100, `done in ${dt}s`);
    setTimeout(clearProgress, 1200);
  } catch (e) {
    console.error(e);
    setProgress(0, `error: ${e.message}`);
  } finally {
    $("generate").disabled = false;
  }
}

async function generateFromImage() {
  if (!inference.ready) return;
  const img = $("image-preview");
  if (!img.src) return;
  $("image-generate").disabled = true;
  const t0 = performance.now();
  try {
    setProgress(20, "carving visual hull…");
    const volume = imageToVolume(img, inference.config.grid);
    setProgress(70, "marching cubes…");
    const mesh = marchingCubes(volume, inference.config.grid, 0.5, 1.0);
    setMesh(mesh);
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    $("gen-time").textContent = `carved image hull in ${dt}s`;
    setProgress(100, `done in ${dt}s`);
    setTimeout(clearProgress, 1200);
  } catch (e) {
    console.error(e);
    setProgress(0, `error: ${e.message}`);
  } finally {
    $("image-generate").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------
const examplesEl = $("examples");
for (const p of EXAMPLE_PROMPTS) {
  const chip = document.createElement("button");
  chip.className = "chip";
  chip.textContent = p;
  chip.onclick = () => { $("prompt").value = p; };
  examplesEl.appendChild(chip);
}

$("generate").onclick = generateFromText;
$("prompt").addEventListener("keydown", (e) => { if (e.key === "Enter") generateFromText(); });

$("cfg").oninput = () => ($("cfg-val").textContent = parseFloat($("cfg").value).toFixed(1));
$("steps").oninput = () => ($("steps-val").textContent = $("steps").value);

$("image-input").onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  $("image-preview").src = url;
  $("image-preview-wrap").classList.remove("hidden");
};
$("image-generate").onclick = generateFromImage;

$("wireframe").onchange = (e) => {
  if (meshObj) meshObj.material.wireframe = e.target.checked;
};
$("autorotate").onchange = (e) => { controls.autoRotate = e.target.checked; };

function slug(s) { return s.toLowerCase().replace(/[^a-z0-9]+/g, "_").slice(0, 40); }
$("dl-glb").onclick = () => currentMesh && downloadBlob(meshToGLB(currentMesh, slug($("prompt").value)), `${slug($("prompt").value) || "murti"}.glb`);
$("dl-obj").onclick = () => currentMesh && downloadBlob(meshToOBJ(currentMesh, slug($("prompt").value)), `${slug($("prompt").value) || "murti"}.obj`);
$("dl-stl").onclick = () => currentMesh && downloadBlob(meshToSTL(currentMesh), `${slug($("prompt").value) || "murti"}.stl`);

resize();
animate();
init();
