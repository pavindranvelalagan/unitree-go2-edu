'use strict';
const $ = id => document.getElementById(id);
const canvas = $('splat-canvas'), loadBtn = $('load-btn'), fileInput = $('file-input');
const loadingOverlay = $('loading-overlay'), welcomeOverlay = $('welcome-overlay');
const progressBar = $('progress-bar'), progressPct = $('progress-pct'), progressLabel = $('progress-label');
const loadingFilename = $('loading-filename'), statusDot = $('status-dot'), statusText = $('status-text');
const infoSplats = $('info-splats'), infoFile = $('info-file'), infoSize = $('info-size'), infoFps = $('info-fps');
const toast = $('toast'), webglWarning = $('webgl-warning');
const sortStatusEl = $('sort-status'), maxSplatsVal = $('max-splats-val'), maxSplatsSlider = $('max-splats-slider');
const steps = [0, 1, 2, 3].map(i => $('step-' + ['read', 'parse', 'upload', 'render'][i]));

// ── State ─────────────────────────────────────────────────
let gl = null, prog = null, bufs = {}, splatCount = 0, renderReady = false, frameCount = 0;
let origData = null, sortPending = false, sortDebounce = null;
let maxSplats = 1000000;

// ── Movement / sort throttle state ────────────────────────
let movementActive = false;
let movementTimeout = null;

const cam = {
  theta: .5, phi: 1.1, radius: 5, panX: 0, panY: 0, target: [0, 0, 0],
  dragging: false, button: -1, lastX: 0, lastY: 0,
  lastSortTheta: 0, lastSortPhi: 0
};

// ── Math ──────────────────────────────────────────────────
const n3 = v => { const l = Math.hypot(...v); return l ? v.map(x => x / l) : v; };
const s3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const x3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const d3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function perspective(fov, asp, n, f) {
  const t = 1 / Math.tan(fov / 2), nf = 1 / (n - f);
  return new Float32Array([t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) * nf, -1, 0, 0, 2 * f * n * nf, 0]);
}
function lookAt(eye, c, up) {
  const f = n3(s3(c, eye)), s = n3(x3(f, up)), u = x3(s, f);
  return new Float32Array([s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0,
  -d3(s, eye), -d3(u, eye), d3(f, eye), 1]);
}
function camEye() {
  const [tx, ty, tz] = cam.target;
  return [cam.panX + tx + cam.radius * Math.sin(cam.phi) * Math.sin(cam.theta),
  cam.panY + ty + cam.radius * Math.cos(cam.phi),
  tz + cam.radius * Math.sin(cam.phi) * Math.cos(cam.theta)];
}
function camCenter() { return [cam.panX + cam.target[0], cam.panY + cam.target[1], cam.target[2]]; }

// ── Shaders (GLSL 300 es – WebGL 2 instanced quad pipeline) ──
// Each splat is a 4-vertex TRIANGLE_STRIP quad.
// Per-quad geometry (a_quad) comes from a shared 4-vertex buffer.
// Per-splat data (a_pos, a_col, a_cov_a, a_cov_b) are instance
// attributes forwarded via vertexAttribDivisor(loc, 1).
const VS = `#version 300 es
precision highp float;

// ── Per-quad (4 verts, divisor 0) ────────────────────────────
in vec2 a_quad;        // corner offset: one of (-1,-1),(1,-1),(-1,1),(1,1)

// ── Per-instance splat data (divisor 1) ──────────────────────
in vec3 a_pos;         // splat world position
in vec4 a_col;         // RGBA colour (pre-sigmoid opacity)
in vec3 a_cov_a;       // upper triangle of 3D cov [Σxx, Σxy, Σxz]
in vec3 a_cov_b;       // lower triangle of 3D cov [Σyy, Σyz, Σzz]

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec2 u_vp;     // viewport width, height in pixels

out vec4  v_col;
out vec3  v_conic;     // (A, B, C) = upper triangle of Σ2D⁻¹
out vec2  v_offset;    // pixel-space offset from splat centre (→ FS)

void main() {
  // ── 1. View-space position ────────────────────────────────
  vec4 vp = u_view * vec4(a_pos, 1.0);
  if (vp.z > -0.1) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    v_col    = vec4(0.0);
    v_conic  = vec3(0.0);
    v_offset = vec2(0.0);
    return;
  }

  float depth = -vp.z;

  // ── 2. View-space covariance  Σv = R · Σ3D · Rᵀ ──────────
  mat3 R = mat3(u_view[0].xyz, u_view[1].xyz, u_view[2].xyz);
  mat3 Sig = mat3(
    a_cov_a.x, a_cov_a.y, a_cov_a.z,
    a_cov_a.y, a_cov_b.x, a_cov_b.y,
    a_cov_a.z, a_cov_b.y, a_cov_b.z
  );
  // Manual transpose – GLSL ES 1.0/2.0 compatible (transpose() is GLSL 3.0+)
  mat3 Rt = mat3(
    R[0][0], R[1][0], R[2][0],
    R[0][1], R[1][1], R[2][1],
    R[0][2], R[1][2], R[2][2]
  );
  mat3 VS3 = R * Sig * Rt;

  // ── 3. EWA splatting Jacobian → Σ2D (pixel-space) ─────────
  float fx  = u_proj[0][0] * u_vp.x * 0.5;
  float fy  = u_proj[1][1] * u_vp.y * 0.5;
  float J00 =  fx / depth;
  float J02 = -fx * vp.x / (depth * depth);
  float J11 =  fy / depth;
  float J12 = -fy * vp.y / (depth * depth);

  // Σ2D = J · Σv · Jᵀ  (+0.3 anti-alias low-pass regulariser)
  float cxx = J00*J00*VS3[0][0] + 2.0*J00*J02*VS3[0][2] + J02*J02*VS3[2][2] + 0.3;
  float cxy = J00*J11*VS3[0][1] + J00*J12*VS3[1][2]
            + J02*J11*VS3[0][2] + J02*J12*VS3[2][2];
  float cyy = J11*J11*VS3[1][1] + 2.0*J11*J12*VS3[1][2] + J12*J12*VS3[2][2] + 0.3;

  // ── 4. Analytically invert Σ2D → conic (A, B, C) ──────────
  float det = max(cxx * cyy - cxy * cxy, 1e-6);
  float iDet = 1.0 / det;
  v_conic = vec3(cyy * iDet, -cxy * iDet, cxx * iDet);  // A, B, C

  // ── 5. Bounding radius in pixels — 3-sigma of the largest axis ──
  float mid  = 0.5 * (cxx + cyy);
  float disc = sqrt(max(0.0, mid * mid - det));
  // r_px = 3σ bounding radius.  No ×2 — that was for point-sprite diameter.
  float r_px = clamp(3.0 * sqrt(mid + disc), 1.0, 1024.0);

  // ── 6. Quad corner position in clip space ────────────────────
  vec4 clip = u_proj * vp;
  // quadOffset is the pixel-space displacement for this corner.
  vec2 quadOffset = a_quad * r_px;
  // Convert pixel offset → NDC offset (range [-1,1]) → clip-space offset
  // (multiply NDC by clip.w so perspective division gives the correct NDC).
  // This preserves clip.zw intact, keeping depth and w correct.
  gl_Position = vec4(
    clip.x + quadOffset.x * (2.0 / u_vp.x) * clip.w,
    clip.y + quadOffset.y * (2.0 / u_vp.y) * clip.w,
    clip.z,
    clip.w
  );

  // ── 7. Pixel offset forwarded to FS for conic evaluation ─────
  // Matches the pixel-space units used by cxx/cxy/cyy above.
  v_offset = quadOffset;

  v_col = a_col;
}`;

// Fragment shader: conic elliptical Gaussian, premultiplied alpha
const FS = `#version 300 es
precision highp float;

in  vec4 v_col;
in  vec3 v_conic;    // (A, B, C) = Σ2D⁻¹ upper triangle
in  vec2 v_offset;  // pixel offset from splat centre

out vec4 fragColor;

void main() {
  vec2 d = v_offset;  // pixel-space displacement from splat centre

  // True conic elliptical Gaussian: exp(-½ dᵀ Σ⁻¹ d)
  float power = -0.5 * (
      v_conic.x * d.x * d.x
    + 2.0 * v_conic.y * d.x * d.y
    + v_conic.z * d.y * d.y
  );

  // power is always ≤ 0 by definition (positive definite conic).
  // Discard only if somehow > 0 (degenerate / numerical noise).
  if (power > 0.0) discard;

  float alpha = v_col.a * exp(power);

  // 1/255 cull — removes invisible ambient haze completely
  if (alpha < 0.00392156862) discard;

  // Premultiplied alpha output (blends with ONE, ONE_MINUS_SRC_ALPHA)
  fragColor = vec4(v_col.rgb * alpha, alpha);
}`;

function mkShader(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
// Shared 4-vertex quad (TRIANGLE_STRIP): covers one splat footprint
// Corners in NDC-local space: BL, BR, TL, TR
const QUAD_VERTS = new Float32Array([-1, -1,  1, -1,  -1, 1,  1, 1]);
let quadBuf = null;   // GL buffer for the 4 quad corners
let vao     = null;   // WebGL 2 VAO

function initGL() {
  try {
    // ── Request WebGL 2 exclusively (GLSL 300 es) ────────────
    gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: true, alpha: false });
    if (!gl) throw new Error(
      'WebGL 2 is not available. Please enable hardware acceleration in your browser settings.'
    );

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);  // premultiplied alpha
    gl.disable(gl.DEPTH_TEST);

    // ── Compile + link program ───────────────────────────────
    const p = gl.createProgram();
    gl.attachShader(p, mkShader(gl.VERTEX_SHADER,   VS));
    gl.attachShader(p, mkShader(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(p));
    prog = p;

    // ── Upload the shared quad geometry ─────────────────────
    quadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, QUAD_VERTS, gl.STATIC_DRAW);

    resize();
    return true;
  } catch (e) {
    console.error(e);
    webglWarning && (webglWarning.style.display = 'block');
    setStatus('error', e.message);
    return false;
  }
}
function resize() {
  canvas.width = innerWidth; canvas.height = innerHeight;
  gl && gl.viewport(0, 0, canvas.width, canvas.height);
  positionPerfPanel();
}

// ── Position perf panel above controls panel (right side) ──
function positionPerfPanel() {
  const perfPanel = document.getElementById('perf-panel');
  const ctrlPanel = document.getElementById('controls-panel');
  if (!perfPanel || !ctrlPanel) return;
  const ctrlRect = ctrlPanel.getBoundingClientRect();
  const gap = 10;
  const bottomPos = window.innerHeight - ctrlRect.top + gap;
  perfPanel.style.bottom = bottomPos + 'px';
}

// ── Render (WebGL 2 instanced quad) ──────────────────────
function render() {
  requestAnimationFrame(render);
  if (!gl) return;
  gl.clearColor(0.027, 0.035, 0.059, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (!renderReady || !prog || !vao || splatCount === 0) return;
  frameCount++;
  gl.useProgram(prog);
  const eye  = camEye();
  const view = lookAt(eye, camCenter(), [0, 1, 0]);
  const proj = perspective(Math.PI / 3, canvas.width / canvas.height, 0.01, 5000);
  gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'u_view'), false, view);
  gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'u_proj'), false, proj);
  gl.uniform2f(gl.getUniformLocation(prog, 'u_vp'), canvas.width, canvas.height);

  // VAO carries all attribute state; one instanced draw renders all splats
  gl.bindVertexArray(vao);
  // 4 quad vertices × splatCount instances → one draw call
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, splatCount);
  gl.bindVertexArray(null);
}

// ── Z-Sort (async, back-to-front, chunked typed-array) ────
const tick = () => new Promise(r => setTimeout(r, 0));

async function sortByDepth() {
  if (!origData || sortPending) return;
  sortPending = true;
  setSortStatus('sorting…');

  const count = Math.min(origData.count, maxSplats);
  const eye = camEye(), view = lookAt(eye, camCenter(), [0, 1, 0]);

  // Row 2 of view matrix (z row, column-major): indices 2, 6, 10, 14
  const m2 = view[2], m6 = view[6], m10 = view[10], m14 = view[14];

  // ── Phase 1: Compute depth per splat in 350k-row chunks ──
  const CHUNK = 350000;
  const depths  = new Float32Array(count);
  const indices = new Int32Array(count);
  for (let i = 0; i < count; i += CHUNK) {
    const end = Math.min(i + CHUNK, count);
    for (let j = i; j < end; j++) {
      indices[j] = j;
      depths[j]  = m2  * origData.pos[j * 3]
                 + m6  * origData.pos[j * 3 + 1]
                 + m10 * origData.pos[j * 3 + 2]
                 + m14;
    }
    await tick(); // yield – keeps UI thread alive
  }

  // ── Phase 2: Sort typed Int32Array directly (no Array.from) ──
  // Build a helper Float64Array of packed (depth | index) for a
  // single-array sort, avoiding a bloated native JS array copy.
  const packed = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    // Store depth in upper bits via a large offset trick:
    // We sort a plain Int32Array using a closure over depths[]
    packed[i] = i; // placeholder; real sort below
  }
  // Sort indices in-place using a typed comparator over depths[]
  // Int32Array doesn't support .sort(comparator) in all engines,
  // so we use a lightweight auxiliary index array (Int32Array).
  const sortedIdx = new Int32Array(count);
  for (let i = 0; i < count; i++) sortedIdx[i] = i;
  // Chunked comparator sort – yields every CHUNK elements scanned
  sortedIdx.sort((a, b) => depths[a] - depths[b]); // back-to-front
  await tick();

  // ── Phase 3: Rearrange data into sorted typed arrays ─────
  const sp = new Float32Array(count * 3);
  const sc = new Float32Array(count * 4);
  const sa = new Float32Array(count * 3);
  const sb = new Float32Array(count * 3);
  for (let i = 0; i < count; i += CHUNK) {
    const end = Math.min(i + CHUNK, count);
    for (let j = i; j < end; j++) {
      const s = sortedIdx[j];
      const j3 = j * 3, s3 = s * 3, j4 = j * 4, s4 = s * 4;
      sp[j3]     = origData.pos[s3];     sp[j3 + 1] = origData.pos[s3 + 1]; sp[j3 + 2] = origData.pos[s3 + 2];
      sc[j4]     = origData.col[s4];     sc[j4 + 1] = origData.col[s4 + 1];
      sc[j4 + 2] = origData.col[s4 + 2]; sc[j4 + 3] = origData.col[s4 + 3];
      sa[j3]     = origData.covA[s3];    sa[j3 + 1] = origData.covA[s3 + 1]; sa[j3 + 2] = origData.covA[s3 + 2];
      sb[j3]     = origData.covB[s3];    sb[j3 + 1] = origData.covB[s3 + 1]; sb[j3 + 2] = origData.covB[s3 + 2];
    }
    await tick(); // yield after every chunk
  }

  // ── Phase 4: GPU upload ───────────────────────────────────
  const up = (buf, data) => {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
  };
  up(bufs.pos, sp); up(bufs.col, sc); up(bufs.covA, sa); up(bufs.covB, sb);
  splatCount = count;
  cam.lastSortTheta = cam.theta;
  cam.lastSortPhi   = cam.phi;
  sortPending = false;
  setSortStatus('sorted ✓');
}

// ── Angle-threshold + movement-downsampling sort trigger ──
const SORT_ANGLE_THRESHOLD = 0.08; // radians
const MOVEMENT_MAX_SPLATS  = 1500000;

function triggerSort() {
  // Check whether the camera has rotated enough to warrant a re-sort
  const dTheta = Math.abs(cam.theta - cam.lastSortTheta);
  const dPhi   = Math.abs(cam.phi   - cam.lastSortPhi);

  if (!movementActive) {
    // First movement event in this gesture – enter movement mode
    movementActive = true;
    // Temporarily cap splats for smoother Intel GPU frame-rates
    if (origData && maxSplats > MOVEMENT_MAX_SPLATS) {
      splatCount = Math.min(origData.count, MOVEMENT_MAX_SPLATS);
    }
  }

  // Cancel any pending final sort
  clearTimeout(sortDebounce);
  clearTimeout(movementTimeout);

  // Schedule a final full-resolution sort once movement stops
  movementTimeout = setTimeout(() => {
    movementActive = false;
    // Restore full splat cap and fire a definitive sort
    sortByDepth();
  }, 300);

  // During active movement, only re-sort if angle shifted significantly
  if (dTheta > SORT_ANGLE_THRESHOLD || dPhi > SORT_ANGLE_THRESHOLD) {
    sortDebounce = setTimeout(() => sortByDepth(), 80);
  }
}

// ── PLY Parser ────────────────────────────────────────────
async function parsePLY(buf, onProg) {
  const u8 = new Uint8Array(buf);
  let hEnd = 0;
  const EH = [101, 110, 100, 95, 104, 101, 97, 100, 101, 114];
  for (let i = 0; i < u8.length - 10; i++) {
    if (EH.every((v, k) => u8[i + k] === v)) { hEnd = i + 11; break; }
  }
  const hdr = new TextDecoder('ascii').decode(u8.slice(0, hEnd));
  let numV = 0; const props = []; let inV = false;
  for (const ln of hdr.split('\n')) {
    const t = ln.trim();
    if (t.startsWith('element vertex')) { numV = parseInt(t.split(' ')[2]); inV = true; }
    else if (t.startsWith('element') && !t.includes('vertex')) inV = false;
    else if (t.startsWith('property') && inV) { const p = t.split(' '); props.push({ type: p[1], name: p[2] }); }
  }
  if (!numV) throw new Error('No vertices in PLY');
  const tsz = { float: 4, uchar: 1, double: 8, int: 4, uint: 4, short: 2, ushort: 2, char: 1 };
  let stride = 0; const layout = {};
  for (const p of props) { layout[p.name] = { offset: stride, type: p.type }; stride += tsz[p.type] || 4; }
  const pos = new Float32Array(numV * 3), col = new Float32Array(numV * 4);
  const covA = new Float32Array(numV * 3), covB = new Float32Array(numV * 3);
  const dv = new DataView(buf, hEnd);
  const SH = 0.28209479177387814;
  const gf = (off, n) => {
    const p = layout[n]; if (!p) return 0;
    if (p.type === 'float') return dv.getFloat32(off + p.offset, true);
    if (p.type === 'uchar') return dv.getUint8(off + p.offset);
    if (p.type === 'double') return dv.getFloat64(off + p.offset, true);
    return dv.getFloat32(off + p.offset, true);
  };
  let mnX = 1e9, mxX = -1e9, mnY = 1e9, mxY = -1e9, mnZ = 1e9, mxZ = -1e9, sX = 0, sY = 0, sZ = 0;
  const B = 40000;
  for (let i = 0; i < numV; i++) {
    const o = i * stride;
    const x = gf(o, 'x'), y = gf(o, 'y'), z = gf(o, 'z');
    pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
    if (x < mnX) mnX = x; if (x > mxX) mxX = x; if (y < mnY) mnY = y; if (y > mxY) mxY = y; if (z < mnZ) mnZ = z; if (z > mxZ) mxZ = z;
    sX += x; sY += y; sZ += z;
    col[i * 4] = Math.max(0, Math.min(1, .5 + SH * gf(o, 'f_dc_0')));
    col[i * 4 + 1] = Math.max(0, Math.min(1, .5 + SH * gf(o, 'f_dc_1')));
    col[i * 4 + 2] = Math.max(0, Math.min(1, .5 + SH * gf(o, 'f_dc_2')));
    col[i * 4 + 3] = 1 / (1 + Math.exp(-gf(o, 'opacity')));
    const s0 = Math.exp(gf(o, 'scale_0')), s1 = Math.exp(gf(o, 'scale_1')), s2 = Math.exp(gf(o, 'scale_2'));
    const q0 = gf(o, 'rot_0'), q1 = gf(o, 'rot_1'), q2 = gf(o, 'rot_2'), q3 = gf(o, 'rot_3');
    const ql = Math.hypot(q0, q1, q2, q3) || 1;
    const [w, rx, ry, rz] = [q0 / ql, q1 / ql, q2 / ql, q3 / ql];
    const R = [1 - 2 * (ry * ry + rz * rz), 2 * (rx * ry - w * rz), 2 * (rx * rz + w * ry),
    2 * (rx * ry + w * rz), 1 - 2 * (rx * rx + rz * rz), 2 * (ry * rz - w * rx),
    2 * (rx * rz - w * ry), 2 * (ry * rz + w * rx), 1 - 2 * (rx * rx + ry * ry)];
    const M = [R[0] * s0, R[1] * s1, R[2] * s2, R[3] * s0, R[4] * s1, R[5] * s2, R[6] * s0, R[7] * s1, R[8] * s2];
    covA[i * 3] = M[0] * M[0] + M[1] * M[1] + M[2] * M[2];
    covA[i * 3 + 1] = M[0] * M[3] + M[1] * M[4] + M[2] * M[5];
    covA[i * 3 + 2] = M[0] * M[6] + M[1] * M[7] + M[2] * M[8];
    covB[i * 3] = M[3] * M[3] + M[4] * M[4] + M[5] * M[5];
    covB[i * 3 + 1] = M[3] * M[6] + M[4] * M[7] + M[5] * M[8];
    covB[i * 3 + 2] = M[6] * M[6] + M[7] * M[7] + M[8] * M[8];
    if (i % B === 0) { onProg(i / numV); await tick(); }
  }
  return {
    pos, col, covA, covB, count: numV,
    centroid: [sX / numV, sY / numV, sZ / numV],
    diag: Math.hypot(mxX - mnX, mxY - mnY, mxZ - mnZ)
  };
}

// ── GPU upload ────────────────────────────────────────────
function mkBuf(data) {
  const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW); return b;
}
function initBufs(data) {
  // Delete old GL resources
  if (vao) gl.deleteVertexArray(vao);
  Object.values(bufs).forEach(b => gl.deleteBuffer(b));

  // Create new per-instance GPU buffers
  bufs = {
    pos:  mkBuf(data.pos),
    col:  mkBuf(data.col),
    covA: mkBuf(data.covA),
    covB: mkBuf(data.covB)
  };

  // ── Build VAO: bind quad corners (divisor 0) + per-instance attrs (divisor 1) ──
  vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  // Helper: bind buffer, enable attrib, set divisor
  const bindAttr = (buf, name, size, divisor) => {
    const loc = gl.getAttribLocation(prog, name);
    if (loc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(loc, divisor); // 0 = per-vertex, 1 = per-instance
  };

  // Shared quad geometry: 4 verts, no divisor (advances every vertex)
  bindAttr(quadBuf,   'a_quad',  2, 0);

  // Per-splat instance data: divisor=1 advances once per instance
  bindAttr(bufs.pos,  'a_pos',   3, 1);
  bindAttr(bufs.col,  'a_col',   4, 1);
  bindAttr(bufs.covA, 'a_cov_a', 3, 1);
  bindAttr(bufs.covB, 'a_cov_b', 3, 1);

  gl.bindVertexArray(null);
}

// ── UI helpers ─────────────────────────────────────────────
function setStatus(s, t) { if (statusDot) statusDot.className = 'status-dot ' + s; if (statusText) statusText.textContent = t; }
function setStep(i) { steps.forEach((e, j) => { if (!e) return; e.classList.remove('active', 'done'); if (j < i) e.classList.add('done'); if (j === i) e.classList.add('active'); }); }
function setProgress(p, lbl) { const v = Math.round(p * 100); if (progressBar) progressBar.style.width = v + '%'; if (progressPct) progressPct.textContent = v + '%'; if (lbl && progressLabel) progressLabel.textContent = lbl; }
function setSortStatus(s) { if (sortStatusEl) sortStatusEl.textContent = s; }
function showToast(msg, type = '') { if (!toast) return; toast.textContent = msg; toast.className = 'show ' + type; clearTimeout(toast._t); toast._t = setTimeout(() => toast.className = '', 3500); }

// ── Load ──────────────────────────────────────────────────
async function loadFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.ply')) { showToast('Select a .ply file', 'error'); return; }
  welcomeOverlay && welcomeOverlay.classList.add('hidden');
  loadingOverlay && loadingOverlay.classList.add('active');
  loadBtn.disabled = true; renderReady = false;
  setStatus('loading', 'Loading…');
  if (loadingFilename) loadingFilename.textContent = file.name;
  if (infoFile) infoFile.textContent = file.name.slice(0, 20);
  if (infoSize) infoSize.textContent = (file.size / 1048576).toFixed(1) + ' MB';
  setProgress(0, 'Reading…'); setStep(0);
  try {
    const buf = await file.arrayBuffer(); setProgress(.12, 'Parsing PLY…'); setStep(1);
    const data = await parsePLY(buf, p => setProgress(.12 + p * .68, `Parsing ${Math.round(p * 100)}%`));
    if (infoSplats) infoSplats.textContent = data.count.toLocaleString();
    origData = data;
    cam.target = data.centroid; cam.radius = Math.max(data.diag * .7, .5);
    cam.panX = 0; cam.panY = 0; cam.theta = .5; cam.phi = 1.1;
    setProgress(.82, 'Uploading GPU…'); setStep(2);
    await tick(); initBufs(data);
    setProgress(.97, 'Sorting…'); setStep(3);
    // Initial sort
    splatCount = Math.min(data.count, maxSplats); renderReady = true;
    await sortByDepth();
    setProgress(1, 'Done!'); await new Promise(r => setTimeout(r, 400));
    loadingOverlay && loadingOverlay.classList.remove('active');
    setStatus('ready', 'Model loaded');
    document.body.classList.add('viewer-active');
    showToast(`Loaded ${data.count.toLocaleString()} splats`, 'success');
  } catch (e) {
    console.error(e); loadingOverlay && loadingOverlay.classList.remove('active');
    setStatus('error', 'Failed'); showToast('Error: ' + e.message, 'error');
    welcomeOverlay && welcomeOverlay.classList.remove('hidden');
  } finally { loadBtn.disabled = false; }
}

// ── Controls ──────────────────────────────────────────────
canvas.addEventListener('mousedown', e => { cam.dragging = true; cam.button = e.button; cam.lastX = e.clientX; cam.lastY = e.clientY; e.preventDefault(); });
window.addEventListener('mousemove', e => {
  if (!cam.dragging) return;
  const dx = e.clientX - cam.lastX, dy = e.clientY - cam.lastY;
  cam.lastX = e.clientX; cam.lastY = e.clientY;
  if (cam.button === 0) { cam.theta -= dx * .007; cam.phi = Math.max(.05, Math.min(Math.PI - .05, cam.phi - dy * .007)); }
  else if (cam.button === 2) { const sp = cam.radius * .001; cam.panX -= dx * sp; cam.panY += dy * sp; }
  triggerSort();
});
window.addEventListener('mouseup', () => { cam.dragging = false; cam.button = -1; });
canvas.addEventListener('wheel', e => { cam.radius = Math.max(.01, cam.radius * (1 + e.deltaY * .001)); e.preventDefault(); triggerSort(); }, { passive: false });
canvas.addEventListener('contextmenu', e => e.preventDefault());

// ── Slider ────────────────────────────────────────────────
maxSplatsSlider && maxSplatsSlider.addEventListener('input', e => {
  maxSplats = parseInt(e.target.value);
  if (maxSplatsVal) maxSplatsVal.textContent = maxSplats.toLocaleString();
  if (origData) triggerSort();
});

// ── File wiring ───────────────────────────────────────────
loadBtn.addEventListener('click', () => fileInput.click());
$('welcome-load-btn')?.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => { if (e.target.files[0]) loadFile(e.target.files[0]); e.target.value = ''; });
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => { e.preventDefault(); const f = [...e.dataTransfer.files].find(f => f.name.toLowerCase().endsWith('.ply')); if (f) loadFile(f); else showToast('Drop a .ply file', 'error'); });

// ── FPS + Init ────────────────────────────────────────────
setInterval(() => { if (infoFps) infoFps.textContent = frameCount + ' fps'; frameCount = 0; }, 1000);
window.addEventListener('resize', resize);
(function init() { if (!initGL()) return; setStatus('', 'Ready'); positionPerfPanel(); requestAnimationFrame(render); })();

