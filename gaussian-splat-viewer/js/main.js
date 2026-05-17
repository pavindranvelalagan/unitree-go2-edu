'use strict';
const $=id=>document.getElementById(id);
const canvas=$('splat-canvas'),loadBtn=$('load-btn'),fileInput=$('file-input');
const loadingOverlay=$('loading-overlay'),welcomeOverlay=$('welcome-overlay');
const progressBar=$('progress-bar'),progressPct=$('progress-pct'),progressLabel=$('progress-label');
const loadingFilename=$('loading-filename'),statusDot=$('status-dot'),statusText=$('status-text');
const infoSplats=$('info-splats'),infoFile=$('info-file'),infoSize=$('info-size'),infoFps=$('info-fps');
const toast=$('toast'),webglWarning=$('webgl-warning');
const sortStatusEl=$('sort-status'),maxSplatsVal=$('max-splats-val'),maxSplatsSlider=$('max-splats-slider');
const steps=[0,1,2,3].map(i=>$('step-'+['read','parse','upload','render'][i]));

// ── State ─────────────────────────────────────────────────
let gl=null,prog=null,bufs={},splatCount=0,renderReady=false,frameCount=0;
let origData=null,sortPending=false,sortDebounce=null;
let maxSplats=1000000;

const cam={theta:.5,phi:1.1,radius:5,panX:0,panY:0,target:[0,0,0],
           dragging:false,button:-1,lastX:0,lastY:0,
           lastSortTheta:0,lastSortPhi:0};

// ── Math ──────────────────────────────────────────────────
const n3=v=>{const l=Math.hypot(...v);return l?v.map(x=>x/l):v;};
const s3=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const x3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const d3=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];

function perspective(fov,asp,n,f){
  const t=1/Math.tan(fov/2),nf=1/(n-f);
  return new Float32Array([t/asp,0,0,0,0,t,0,0,0,0,(f+n)*nf,-1,0,0,2*f*n*nf,0]);
}
function lookAt(eye,c,up){
  const f=n3(s3(c,eye)),s=n3(x3(f,up)),u=x3(s,f);
  return new Float32Array([s[0],u[0],-f[0],0,s[1],u[1],-f[1],0,s[2],u[2],-f[2],0,
    -d3(s,eye),-d3(u,eye),d3(f,eye),1]);
}
function camEye(){
  const[tx,ty,tz]=cam.target;
  return[cam.panX+tx+cam.radius*Math.sin(cam.phi)*Math.sin(cam.theta),
         cam.panY+ty+cam.radius*Math.cos(cam.phi),
         tz+cam.radius*Math.sin(cam.phi)*Math.cos(cam.theta)];
}
function camCenter(){return[cam.panX+cam.target[0],cam.panY+cam.target[1],cam.target[2]];}

// ── Shaders ───────────────────────────────────────────────
// FIX: Full EWA Jacobian with view-space covariance rotation → correct splat size
const VS=`
precision highp float;
attribute vec3 a_pos;attribute vec4 a_col;
attribute vec3 a_cov_a;attribute vec3 a_cov_b;
uniform mat4 u_view;uniform mat4 u_proj;uniform vec2 u_vp;
varying vec4 v_col;
void main(){
  vec4 vp=u_view*vec4(a_pos,1.0);
  if(vp.z>-0.1){gl_Position=vec4(0,0,2,1);gl_PointSize=0.0;v_col=vec4(0);return;}
  gl_Position=u_proj*vp;
  float depth=-vp.z;
  // Extract view rotation (upper-left 3x3 of view matrix, column-major)
  mat3 R=mat3(u_view[0].xyz,u_view[1].xyz,u_view[2].xyz);
  // Build symmetric 3D cov matrix
  mat3 S=mat3(a_cov_a.x,a_cov_a.y,a_cov_a.z,
              a_cov_a.y,a_cov_b.x,a_cov_b.y,
              a_cov_a.z,a_cov_b.y,a_cov_b.z);
  // Transform cov to view space: R*S*Rt
  mat3 Rt=mat3(R[0][0],R[1][0],R[2][0],R[0][1],R[1][1],R[2][1],R[0][2],R[1][2],R[2][2]);
  mat3 VS3=R*S*Rt;
  // EWA Jacobian
  float fx=u_proj[0][0]*u_vp.x*0.5;
  float fy=u_proj[1][1]*u_vp.y*0.5;
  float tx=vp.x,ty2=vp.y,tz=depth;
  float J00=fx/tz,J02=-fx*tx/(tz*tz);
  float J11=fy/tz,J12=-fy*ty2/(tz*tz);
  // 2D cov
  float c2xx=J00*J00*VS3[0][0]+2.0*J00*J02*VS3[0][2]+J02*J02*VS3[2][2]+0.3;
  float c2xy=J00*J11*VS3[0][1]+J00*J12*VS3[1][2]+J02*J11*VS3[0][2]+J02*J12*VS3[2][2];
  float c2yy=J11*J11*VS3[1][1]+2.0*J11*J12*VS3[1][2]+J12*J12*VS3[2][2]+0.3;
  float mid=0.5*(c2xx+c2yy);
  float disc=sqrt(max(0.0,mid*mid-(c2xx*c2yy-c2xy*c2xy)));
  gl_PointSize=clamp(3.0*sqrt(mid+disc)*2.0,1.0,30.0);
  v_col=a_col;
}`;
// FIX: premultiplied alpha output (pairs with ONE, ONE_MINUS_SRC_ALPHA blend)
const FS=`
precision highp float;
varying vec4 v_col;
void main(){
  vec2 uv=gl_PointCoord*2.0-1.0;
  float r2=dot(uv,uv);
  if(r2>1.0)discard;
  float a=v_col.a*exp(-4.0*r2);
  if(a<0.003)discard;
  gl_FragColor=vec4(v_col.rgb*a,a);
}`;

function mkShader(type,src){
  const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function initGL(){
  try{
    gl=canvas.getContext('webgl',{antialias:false,premultipliedAlpha:true,alpha:false})
      ||canvas.getContext('experimental-webgl',{antialias:false,alpha:false});
    if(!gl)throw new Error('No WebGL');
    gl.enable(gl.BLEND);
    // FIX: premultiplied alpha blending
    gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);
    const p=gl.createProgram();
    gl.attachShader(p,mkShader(gl.VERTEX_SHADER,VS));
    gl.attachShader(p,mkShader(gl.FRAGMENT_SHADER,FS));
    gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));
    prog=p;resize();return true;
  }catch(e){console.error(e);webglWarning&&(webglWarning.style.display='block');setStatus('error',e.message);return false;}
}
function resize(){canvas.width=innerWidth;canvas.height=innerHeight;gl&&gl.viewport(0,0,canvas.width,canvas.height);}

// ── Render ────────────────────────────────────────────────
function render(){
  requestAnimationFrame(render);
  if(!gl)return;
  gl.clearColor(0.027,0.035,0.059,1);gl.clear(gl.COLOR_BUFFER_BIT);
  if(!renderReady||!prog)return;
  frameCount++;
  gl.useProgram(prog);
  const eye=camEye(),view=lookAt(eye,camCenter(),[0,1,0]);
  const proj=perspective(Math.PI/3,canvas.width/canvas.height,0.01,5000);
  gl.uniformMatrix4fv(gl.getUniformLocation(prog,'u_view'),false,view);
  gl.uniformMatrix4fv(gl.getUniformLocation(prog,'u_proj'),false,proj);
  gl.uniform2f(gl.getUniformLocation(prog,'u_vp'),canvas.width,canvas.height);
  const bind=(buf,name,sz)=>{const loc=gl.getAttribLocation(prog,name);if(loc<0)return;
    gl.bindBuffer(gl.ARRAY_BUFFER,buf);gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc,sz,gl.FLOAT,false,0,0);};
  bind(bufs.pos,'a_pos',3);bind(bufs.col,'a_col',4);
  bind(bufs.covA,'a_cov_a',3);bind(bufs.covB,'a_cov_b',3);
  const C=500000;
  for(let i=0;i<splatCount;i+=C)gl.drawArrays(gl.POINTS,i,Math.min(C,splatCount-i));
}

// ── Z-Sort (async, back-to-front) ─────────────────────────
async function sortByDepth(){
  if(!origData||sortPending)return;
  sortPending=true;
  setSortStatus('sorting…');
  const count=Math.min(origData.count,maxSplats);
  const eye=camEye(),view=lookAt(eye,camCenter(),[0,1,0]);
  // Row 2 of view matrix (z row, column-major): indices 2,6,10,14
  const[m2,m6,m10,m14]=[view[2],view[6],view[10],view[14]];
  // Compute depths
  const depths=new Float32Array(count);
  const indices=new Int32Array(count);
  const CHUNK=200000;
  for(let i=0;i<count;i+=CHUNK){
    const e=Math.min(i+CHUNK,count);
    for(let j=i;j<e;j++){
      indices[j]=j;
      depths[j]=m2*origData.pos[j*3]+m6*origData.pos[j*3+1]+m10*origData.pos[j*3+2]+m14;
    }
    await tick();
  }
  // Sort indices back-to-front (most negative view-z = farthest away)
  const arr=Array.from(indices);
  arr.sort((a,b)=>depths[a]-depths[b]);
  await tick();
  // Rearrange data into sorted typed arrays
  const sp=new Float32Array(count*3),sc=new Float32Array(count*4);
  const sa=new Float32Array(count*3),sb=new Float32Array(count*3);
  for(let i=0;i<count;i+=CHUNK){
    const e=Math.min(i+CHUNK,count);
    for(let j=i;j<e;j++){
      const s=arr[j];
      sp[j*3]=origData.pos[s*3];sp[j*3+1]=origData.pos[s*3+1];sp[j*3+2]=origData.pos[s*3+2];
      sc[j*4]=origData.col[s*4];sc[j*4+1]=origData.col[s*4+1];sc[j*4+2]=origData.col[s*4+2];sc[j*4+3]=origData.col[s*4+3];
      sa[j*3]=origData.covA[s*3];sa[j*3+1]=origData.covA[s*3+1];sa[j*3+2]=origData.covA[s*3+2];
      sb[j*3]=origData.covB[s*3];sb[j*3+1]=origData.covB[s*3+1];sb[j*3+2]=origData.covB[s*3+2];
    }
    await tick();
  }
  // Upload
  const up=(buf,data)=>{gl.bindBuffer(gl.ARRAY_BUFFER,buf);gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);};
  up(bufs.pos,sp);up(bufs.col,sc);up(bufs.covA,sa);up(bufs.covB,sb);
  splatCount=count;
  cam.lastSortTheta=cam.theta;cam.lastSortPhi=cam.phi;
  sortPending=false;
  setSortStatus('sorted ✓');
}
const tick=()=>new Promise(r=>setTimeout(r,0));

function triggerSort(){
  clearTimeout(sortDebounce);
  sortDebounce=setTimeout(()=>sortByDepth(),300);
}

// ── PLY Parser ────────────────────────────────────────────
async function parsePLY(buf,onProg){
  const u8=new Uint8Array(buf);
  let hEnd=0;
  const EH=[101,110,100,95,104,101,97,100,101,114];
  for(let i=0;i<u8.length-10;i++){
    if(EH.every((v,k)=>u8[i+k]===v)){hEnd=i+11;break;}
  }
  const hdr=new TextDecoder('ascii').decode(u8.slice(0,hEnd));
  let numV=0;const props=[];let inV=false;
  for(const ln of hdr.split('\n')){
    const t=ln.trim();
    if(t.startsWith('element vertex')){numV=parseInt(t.split(' ')[2]);inV=true;}
    else if(t.startsWith('element')&&!t.includes('vertex'))inV=false;
    else if(t.startsWith('property')&&inV){const p=t.split(' ');props.push({type:p[1],name:p[2]});}
  }
  if(!numV)throw new Error('No vertices in PLY');
  const tsz={float:4,uchar:1,double:8,int:4,uint:4,short:2,ushort:2,char:1};
  let stride=0;const layout={};
  for(const p of props){layout[p.name]={offset:stride,type:p.type};stride+=tsz[p.type]||4;}
  const pos=new Float32Array(numV*3),col=new Float32Array(numV*4);
  const covA=new Float32Array(numV*3),covB=new Float32Array(numV*3);
  const dv=new DataView(buf,hEnd);
  const SH=0.28209479177387814;
  const gf=(off,n)=>{const p=layout[n];if(!p)return 0;
    if(p.type==='float')return dv.getFloat32(off+p.offset,true);
    if(p.type==='uchar')return dv.getUint8(off+p.offset);
    if(p.type==='double')return dv.getFloat64(off+p.offset,true);
    return dv.getFloat32(off+p.offset,true);};
  let mnX=1e9,mxX=-1e9,mnY=1e9,mxY=-1e9,mnZ=1e9,mxZ=-1e9,sX=0,sY=0,sZ=0;
  const B=40000;
  for(let i=0;i<numV;i++){
    const o=i*stride;
    const x=gf(o,'x'),y=gf(o,'y'),z=gf(o,'z');
    pos[i*3]=x;pos[i*3+1]=y;pos[i*3+2]=z;
    if(x<mnX)mnX=x;if(x>mxX)mxX=x;if(y<mnY)mnY=y;if(y>mxY)mxY=y;if(z<mnZ)mnZ=z;if(z>mxZ)mxZ=z;
    sX+=x;sY+=y;sZ+=z;
    col[i*4]=Math.max(0,Math.min(1,.5+SH*gf(o,'f_dc_0')));
    col[i*4+1]=Math.max(0,Math.min(1,.5+SH*gf(o,'f_dc_1')));
    col[i*4+2]=Math.max(0,Math.min(1,.5+SH*gf(o,'f_dc_2')));
    col[i*4+3]=1/(1+Math.exp(-gf(o,'opacity')));
    const s0=Math.exp(gf(o,'scale_0')),s1=Math.exp(gf(o,'scale_1')),s2=Math.exp(gf(o,'scale_2'));
    const q0=gf(o,'rot_0'),q1=gf(o,'rot_1'),q2=gf(o,'rot_2'),q3=gf(o,'rot_3');
    const ql=Math.hypot(q0,q1,q2,q3)||1;
    const[w,rx,ry,rz]=[q0/ql,q1/ql,q2/ql,q3/ql];
    const R=[1-2*(ry*ry+rz*rz),2*(rx*ry-w*rz),2*(rx*rz+w*ry),
             2*(rx*ry+w*rz),1-2*(rx*rx+rz*rz),2*(ry*rz-w*rx),
             2*(rx*rz-w*ry),2*(ry*rz+w*rx),1-2*(rx*rx+ry*ry)];
    const M=[R[0]*s0,R[1]*s1,R[2]*s2,R[3]*s0,R[4]*s1,R[5]*s2,R[6]*s0,R[7]*s1,R[8]*s2];
    covA[i*3]=M[0]*M[0]+M[1]*M[1]+M[2]*M[2];
    covA[i*3+1]=M[0]*M[3]+M[1]*M[4]+M[2]*M[5];
    covA[i*3+2]=M[0]*M[6]+M[1]*M[7]+M[2]*M[8];
    covB[i*3]=M[3]*M[3]+M[4]*M[4]+M[5]*M[5];
    covB[i*3+1]=M[3]*M[6]+M[4]*M[7]+M[5]*M[8];
    covB[i*3+2]=M[6]*M[6]+M[7]*M[7]+M[8]*M[8];
    if(i%B===0){onProg(i/numV);await tick();}
  }
  return{pos,col,covA,covB,count:numV,
    centroid:[sX/numV,sY/numV,sZ/numV],
    diag:Math.hypot(mxX-mnX,mxY-mnY,mxZ-mnZ)};
}

// ── GPU upload ────────────────────────────────────────────
function mkBuf(data){
  const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
  gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);return b;
}
function initBufs(data){
  Object.values(bufs).forEach(b=>gl.deleteBuffer(b));
  bufs={pos:mkBuf(data.pos),col:mkBuf(data.col),covA:mkBuf(data.covA),covB:mkBuf(data.covB)};
}

// ── UI helpers ─────────────────────────────────────────────
function setStatus(s,t){if(statusDot)statusDot.className='status-dot '+s;if(statusText)statusText.textContent=t;}
function setStep(i){steps.forEach((e,j)=>{if(!e)return;e.classList.remove('active','done');if(j<i)e.classList.add('done');if(j===i)e.classList.add('active');});}
function setProgress(p,lbl){const v=Math.round(p*100);if(progressBar)progressBar.style.width=v+'%';if(progressPct)progressPct.textContent=v+'%';if(lbl&&progressLabel)progressLabel.textContent=lbl;}
function setSortStatus(s){if(sortStatusEl)sortStatusEl.textContent=s;}
function showToast(msg,type=''){if(!toast)return;toast.textContent=msg;toast.className='show '+type;clearTimeout(toast._t);toast._t=setTimeout(()=>toast.className='',3500);}

// ── Load ──────────────────────────────────────────────────
async function loadFile(file){
  if(!file)return;
  if(!file.name.toLowerCase().endsWith('.ply')){showToast('Select a .ply file','error');return;}
  welcomeOverlay&&welcomeOverlay.classList.add('hidden');
  loadingOverlay&&loadingOverlay.classList.add('active');
  loadBtn.disabled=true;renderReady=false;
  setStatus('loading','Loading…');
  if(loadingFilename)loadingFilename.textContent=file.name;
  if(infoFile)infoFile.textContent=file.name.slice(0,20);
  if(infoSize)infoSize.textContent=(file.size/1048576).toFixed(1)+' MB';
  setProgress(0,'Reading…');setStep(0);
  try{
    const buf=await file.arrayBuffer();setProgress(.12,'Parsing PLY…');setStep(1);
    const data=await parsePLY(buf,p=>setProgress(.12+p*.68,`Parsing ${Math.round(p*100)}%`));
    if(infoSplats)infoSplats.textContent=data.count.toLocaleString();
    origData=data;
    cam.target=data.centroid;cam.radius=Math.max(data.diag*.7,.5);
    cam.panX=0;cam.panY=0;cam.theta=.5;cam.phi=1.1;
    setProgress(.82,'Uploading GPU…');setStep(2);
    await tick();initBufs(data);
    setProgress(.97,'Sorting…');setStep(3);
    // Initial sort
    splatCount=Math.min(data.count,maxSplats);renderReady=true;
    await sortByDepth();
    setProgress(1,'Done!');await new Promise(r=>setTimeout(r,400));
    loadingOverlay&&loadingOverlay.classList.remove('active');
    setStatus('ready','Model loaded');
    showToast(`Loaded ${data.count.toLocaleString()} splats`,'success');
  }catch(e){
    console.error(e);loadingOverlay&&loadingOverlay.classList.remove('active');
    setStatus('error','Failed');showToast('Error: '+e.message,'error');
    welcomeOverlay&&welcomeOverlay.classList.remove('hidden');
  }finally{loadBtn.disabled=false;}
}

// ── Controls ──────────────────────────────────────────────
canvas.addEventListener('mousedown',e=>{cam.dragging=true;cam.button=e.button;cam.lastX=e.clientX;cam.lastY=e.clientY;e.preventDefault();});
window.addEventListener('mousemove',e=>{
  if(!cam.dragging)return;
  const dx=e.clientX-cam.lastX,dy=e.clientY-cam.lastY;
  cam.lastX=e.clientX;cam.lastY=e.clientY;
  if(cam.button===0){cam.theta-=dx*.007;cam.phi=Math.max(.05,Math.min(Math.PI-.05,cam.phi-dy*.007));}
  else if(cam.button===2){const sp=cam.radius*.001;cam.panX-=dx*sp;cam.panY+=dy*sp;}
  triggerSort();
});
window.addEventListener('mouseup',()=>{cam.dragging=false;cam.button=-1;});
canvas.addEventListener('wheel',e=>{cam.radius=Math.max(.01,cam.radius*(1+e.deltaY*.001));e.preventDefault();triggerSort();},{passive:false});
canvas.addEventListener('contextmenu',e=>e.preventDefault());

// ── Slider ────────────────────────────────────────────────
maxSplatsSlider&&maxSplatsSlider.addEventListener('input',e=>{
  maxSplats=parseInt(e.target.value);
  if(maxSplatsVal)maxSplatsVal.textContent=maxSplats.toLocaleString();
  if(origData)triggerSort();
});

// ── File wiring ───────────────────────────────────────────
loadBtn.addEventListener('click',()=>fileInput.click());
$('welcome-load-btn')?.addEventListener('click',()=>fileInput.click());
fileInput.addEventListener('change',e=>{if(e.target.files[0])loadFile(e.target.files[0]);e.target.value='';});
document.addEventListener('dragover',e=>e.preventDefault());
document.addEventListener('drop',e=>{e.preventDefault();const f=[...e.dataTransfer.files].find(f=>f.name.toLowerCase().endsWith('.ply'));if(f)loadFile(f);else showToast('Drop a .ply file','error');});

// ── FPS + Init ────────────────────────────────────────────
setInterval(()=>{if(infoFps)infoFps.textContent=frameCount+' fps';frameCount=0;},1000);
window.addEventListener('resize',resize);
(function init(){if(!initGL())return;setStatus('','Ready');requestAnimationFrame(render);})();
