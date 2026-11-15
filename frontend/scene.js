
(() => {
  const photo = document.querySelector('.nightscape');
  const station = document.querySelector('#station');
  const sky = document.querySelector('#sky');
  const ctx = station.getContext('2d');
  const skyCtx = sky.getContext('2d');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  // Coordinates of the pier's top in the original 1672 × 941 background.
  const pier = {x:393, y:595};
  const telescopeScale = 65;
  const gimbalHeight = .65;
  const mount = [];
  const tube = [];
  const meteorTrails = [
    {start:450, duration:1150, x:.86, y:.09, run:123, rise:54},
    {start:4390, duration:980, x:.71, y:.18, run:91, rise:43}
  ];
  let width = 0, height = 0, scale = 1;
  let anchorX = 0, anchorY = 0, imageScale = 1;
  let yaw = .89, pitch = .46;
  let targetYaw = yaw, targetPitch = pitch;
  let previous = performance.now();
  let meteorWasVisible = false;

  function lathe(mesh, rings, color, vertical = false) {
    const segments = 48;
    const loop = rings.map(([axis,radius]) => Array.from({length:segments}, (_, i) => {
      const angle = i*Math.PI*2/segments;
      const x = radius*Math.cos(angle);
      const y = radius*Math.sin(angle);
      return vertical ? [x,axis,y] : [x,y,axis];
    }));
    for (let k = 0; k < loop.length-1; k++) {
      for (let i = 0; i < segments; i++) {
        const next = (i+1)%segments;
        const vertices = [loop[k][i],loop[k+1][i],loop[k+1][next],loop[k][next]];
        mesh.push({vertices:vertical ? vertices : vertices.reverse(),color});
      }
    }
  }

  function arm(center, dimensions, color) {
    const [cx,cy,cz] = center;
    const [w,h,d] = dimensions.map(value=>value/2);
    const vertices = [[-w,-h,-d],[w,-h,-d],[w,h,-d],[-w,h,-d],[-w,-h,d],[w,-h,d],[w,h,d],[-w,h,d]]
      .map(([x,y,z])=>[cx+x,cy+y,cz+z]);
    for (const indices of [[0,3,2,1],[4,5,6,7],[0,4,7,3],[1,2,6,5],[3,7,6,2],[0,1,5,4]]) {
      mount.push({vertices:indices.map(i=>vertices[i]),color});
    }
  }

  const graphite = [55,58,61];
  const metal = [139,143,148];
  const enamel = [172,173,166];
  lathe(mount, [[-.04,.36],[.02,.36],[.07,.29],[.22,.28],[.27,.34],[.33,.34]], graphite,true);
  arm([0,.32,0],[.82,.16,.40],metal);
  for (const side of [-1,1]) {
    arm([side*.37,.51,0],[.16,.40,.35],metal);
    arm([side*.46,.65,0],[.055,.19,.25],graphite);
  }
  lathe(tube, [[-.57,0],[-.57,.19],[-.53,.25],[-.39,.25],[-.35,.285],[.39,.285],[.47,.325],[.97,.325],[1.02,.30]],enamel);
  lathe(tube, [[-.46,.253],[-.40,.253]],graphite);
  lathe(tube, [[-.23,.289],[-.17,.289],[.12,.289],[.15,.289]],metal);
  lathe(tube, [[.48,.329],[.54,.329]],graphite);
  lathe(tube, [[.84,.329],[.91,.329]],metal);
  lathe(tube, [[1.02,.303],[1.025,.246],[.99,.223]],graphite);
  lathe(tube, [[.99,.223],[.98,.176],[.973,0]], [52,87,97]);

  function rotate([x,y,z], tilt) {
    const forward = z*Math.cos(tilt)-y*Math.sin(tilt);
    return [x*Math.cos(yaw)+forward*Math.sin(yaw), y*Math.cos(tilt)+z*Math.sin(tilt), forward*Math.cos(yaw)-x*Math.sin(yaw)];
  }

  function project([x,y,z]) {
    const depth = z*.993 + y*.117;
    const perspective = telescopeScale*imageScale/(1-depth*.045);
    return [anchorX+x*perspective,anchorY-(y*.993-z*.117)*perspective,depth];
  }

  function projectFace({vertices,color},tilt,offsetY) {
    const points = vertices.map(vertex=>{
      const [x,y,z]=rotate(vertex,tilt);
      return [x,y+offsetY,z];
    });
    const [a,b,c,d] = points;
    const u=b.map((value,i)=>value-a[i]);
    let v=c.map((value,i)=>value-a[i]);
    const cross=(f,g)=>[f[1]*g[2]-f[2]*g[1],f[2]*g[0]-f[0]*g[2],f[0]*g[1]-f[1]*g[0]];
    let normal=cross(u,v);
    if (Math.hypot(...normal)<.00001) {
      v=d.map((value,i)=>value-a[i]);
      normal=cross(u,v);
    }
    const length=Math.hypot(...normal)||1;
    const [x,y,z]=normal.map(value=>value/length);
    const cool=Math.max(0,-x*.29+y*.84+z*.34);
    const warm=Math.max(0,-x*.91+y*.12+z*.21);
    const gloss=Math.pow(Math.max(0,-x*.40+y*.43+z*.78),24);
    const rgb=color.map((channel,i)=>Math.round(Math.min(255,channel*(.23+cool*[.43,.47,.51][i])+warm*[58,37,19][i]+gloss*[22,23,24][i])));
    const projected=points.map(project);
    return {projected,depth:projected.reduce((sum,point)=>sum+point[2],0)/projected.length,fill:`rgb(${rgb.join(',')})`};
  }

  function drawTelescope() {
    ctx.clearRect(0,0,width,height);
    const faces=mount.map(face=>projectFace(face,0,0)).concat(tube.map(face=>projectFace(face,pitch,gimbalHeight)));
    faces.sort((a,b)=>a.depth-b.depth);
    ctx.lineWidth=.4;
    for (const {projected,fill} of faces) {
      ctx.beginPath();
      projected.forEach(([x,y],i)=>i ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
      ctx.closePath();
      ctx.fillStyle=ctx.strokeStyle=fill;
      ctx.fill();
      ctx.stroke();
    }
  }

  function drawMeteor(time, force=false) {
    const cycle=(time+650)%8300;
    const trail=motion.matches ? undefined : meteorTrails.find(({start,duration}) => cycle>=start && cycle<start+duration);
    const visible=Boolean(trail);
    if (!force && !visible && !meteorWasVisible) return;
    meteorWasVisible=visible;
    skyCtx.clearRect(0,0,width,height);
    if (!trail) return;
    const progress=(cycle-trail.start)/trail.duration;
    const opacity=Math.sin(progress*Math.PI);
    const x=width*trail.x-progress*trail.run;
    const y=height*trail.y+progress*trail.rise;
    const tail=skyCtx.createLinearGradient(x,y,x+trail.run*.58,y-trail.rise*.58);
    tail.addColorStop(0,`rgba(232,230,243,${opacity*.92})`);
    tail.addColorStop(1,'#e8e6f300');
    skyCtx.strokeStyle=tail;
    skyCtx.beginPath();
    skyCtx.moveTo(x,y);
    skyCtx.lineTo(x+trail.run*.58,y-trail.rise*.58);
    skyCtx.stroke();
    skyCtx.fillStyle=`rgba(247,241,231,${opacity*.9})`;
    skyCtx.beginPath();
    skyCtx.arc(x,y,1,0,Math.PI*2);
    skyCtx.fill();
  }

  function resize() {
    width=station.clientWidth;
    height=station.clientHeight;
    if (!width || !height) return;
    const sourceWidth=photo.naturalWidth || 1672;
    const sourceHeight=photo.naturalHeight || 941;
    imageScale=Math.max(width/sourceWidth,height/sourceHeight);
    const horizontalAlignment=parseFloat(getComputedStyle(photo).objectPosition)/100;
    anchorX=pier.x*imageScale+(width-sourceWidth*imageScale)*horizontalAlignment;
    anchorY=pier.y*imageScale+height-sourceHeight*imageScale;
    scale=Math.min(devicePixelRatio || 1,3);
    for (const canvas of [station,sky]) {
      canvas.width=Math.round(width*scale);
      canvas.height=Math.round(height*scale);
    }
    ctx.setTransform(scale,0,0,scale,0,0);
    skyCtx.setTransform(scale,0,0,scale,0,0);
    drawTelescope();
    drawMeteor(performance.now(),true);
  }

  new ResizeObserver(resize).observe(station);
  photo.addEventListener('load',resize);
  window.addEventListener('pointermove',event=>{
    if (event.pointerType==='touch' || motion.matches) return;
    const bounds=station.getBoundingClientRect();
    const x=event.clientX-bounds.left-anchorX;
    const y=anchorY-gimbalHeight*telescopeScale*imageScale-event.clientY+bounds.top;
    targetYaw=1.44*Math.tanh(x/Math.max(110,width*.34));
    targetPitch=.19+.88*Math.tanh(Math.max(0,y)/Math.max(100,height*.52));
  },{passive:true});

  function frame(time) {
    const step=1-Math.exp(-Math.min(time-previous,65)/80);
    previous=time;
    if (width && height) {
      if (!motion.matches && Math.abs(targetYaw-yaw)+Math.abs(targetPitch-pitch)>.001) {
        yaw+=(targetYaw-yaw)*step;
        pitch+=(targetPitch-pitch)*step;
        drawTelescope();
      }
      drawMeteor(time);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

