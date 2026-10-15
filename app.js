'use strict';

// Table 1 in the supplied LoopNav manuscript (page 8).
// Model order: Oasis, NWM, MineWorld, DIAMOND, WorldMem.
const benchmark = {
  TURN: {
    ranges: ['60–90°', '90–120°', '120–150°', '150–180°'],
    rows: [
      [[.555,.21],[.344,.23],[.721,.17],[.597,.23],[.839,.14]],
      [[.458,.22],[.293,.19],[.665,.21],[.484,.23],[.846,.12]],
      [[.472,.24],[.291,.20],[.597,.20],[.470,.26],[.839,.13]],
      [[.458,.20],[.253,.16],[.593,.18],[.510,.23],[.829,.11]]
    ],
    insight: 'WorldMem has the highest mean SGCS in every TURN range, scoring 0.829–0.846. Rotation tests whether previously observed views remain consistent.'
  },
  ABA: {
    ranges: ['5', '15', '30', '50'],
    rows: [
      [[.308,.24],[.398,.26],[.413,.24],[.311,.26],[.530,.25]],
      [[.263,.20],[.382,.24],[.367,.23],[.306,.23],[.373,.21]],
      [[.184,.15],[.376,.21],[.295,.24],[.198,.19],[.195,.15]],
      [[.110,.09],[.352,.21],[.325,.23],[.181,.18],[.108,.07]]
    ],
    insight: 'WorldMem has the highest SGCS at 5 blocks; NWM leads at 15, 30, and 50 blocks. Maintaining consistency over longer loops remains challenging.'
  },
  ABCA: {
    ranges: ['5', '15', '30', '50'],
    rows: [
      [[.352,.28],[.390,.30],[.451,.26],[.349,.26],[.551,.27]],
      [[.273,.22],[.402,.25],[.421,.24],[.276,.22],[.415,.25]],
      [[.205,.21],[.365,.25],[.330,.23],[.237,.23],[.291,.25]],
      [[.204,.21],[.391,.22],[.376,.26],[.218,.23],[.329,.27]]
    ],
    insight: 'WorldMem leads at 5 blocks, MineWorld at 15, and NWM at 30 and 50. Longer paths may reach simpler grass-and-sky scenes, so scores should be read alongside scene complexity.'
  }
};
const $ = (selector) => document.querySelector(selector);

function renderResults(task) {
  const data = benchmark[task];
  document.querySelectorAll('[data-result-task]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.resultTask === task));
  });
  $('#range-heading').textContent = task === 'TURN' ? 'Angle range' : 'Range (blocks)';
  $('#results-rows').innerHTML = data.rows.map((row, i) => {
    const best = Math.max(...row.map(([mean]) => mean));
    return `<tr><th scope="row">${data.ranges[i]}</th>${row.map(([mean, std]) => `<td${mean === best ? ' class="best"' : ''}>${mean === best ? '<strong>' : ''}${mean.toFixed(3)}${mean === best ? '</strong>' : ''}<span> ± ${std.toFixed(2)}</span></td>`).join('')}</tr>`;
  }).join('');
  $('#results-insight').textContent = data.insight;
}
document.querySelectorAll('[data-result-task]').forEach(button => button.addEventListener('click', () => renderResults(button.dataset.resultTask)));
renderResults('ABA');

const samples = {
  turn: {title:'Turn away. Look back.', description:'Rotate away from the initial view, then reverse the turn. Returning to a familiar direction tests whether the observed scene is preserved.', range:'120–150°', first:0,last:47,returnFrame:24,speed:1,name:'TURN',route:'rotate and return',explorePath:'M 96 93 C 96 15 232 15 232 93',returnPath:'M 232 93 C 232 164 96 164 96 93',nodes:[[96,93,'A'],[232,93,'B']]},
  aba: {title:'Leave, then retrace.', description:'Travel from A to B, then return along the forward waypoints in reverse. The return tests whether the model can reconstruct previously observed places.',range:'5 blocks',first:46,last:181,returnFrame:113,speed:1,name:'ABA',route:'A → B → A',explorePath:'M 48 83 C 117 45 203 45 272 83',returnPath:'M 272 97 C 203 135 117 135 48 97',nodes:[[48,90,'A'],[272,90,'B']]},
  abca: {title:'A different route home.',description:'Visit B and C before returning to A. The final leg tests reconstruction after a longer loop and a different approach to the starting location.',range:'30 blocks',first:20,last:841,returnFrame:572,speed:3,name:'ABCA',route:'A → B → C → A',explorePath:'M 57 137 L 160 35 L 263 137',returnPath:'M 263 137 L 57 137',nodes:[[57,137,'A'],[160,35,'B'],[263,137,'C']]}
};
let currentTask = 'aba';
const video = $('#trajectory-video');
function returnTime() { const sample=samples[currentTask]; return (sample.returnFrame-sample.first)/20/sample.speed; }
function drawMap(sample) {
  let svg='<defs><pattern id="map-grid" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="10" cy="10" r=".8" fill="#e3e8e5"/></pattern></defs><rect width="320" height="180" fill="url(#map-grid)"/>';
  svg+=`<path id="explore-path" d="${sample.explorePath}" fill="none" stroke="#40856b" stroke-width="3" stroke-linecap="round"/><path id="return-path" d="${sample.returnPath}" fill="none" stroke="#b77b40" stroke-width="3" stroke-linecap="round" stroke-dasharray="5 5"/>`;
  sample.nodes.forEach(([x,y,label]) => {svg+=`<circle cx="${x}" cy="${y}" r="12" fill="white" stroke="#d2ddd6"/><text x="${x}" y="${y+4}" text-anchor="middle" fill="#5e7265" font-size="11" font-family="Inter,sans-serif">${label}</text>`;});
  $('#trajectory-map').innerHTML=svg;
  $('#trajectory-map').setAttribute('aria-label',`Schematic of ${sample.name}: ${sample.route}. Solid green is exploration; dashed amber is the return.`);
}
function updatePlayback() {
  const sample=samples[currentTask];
  const duration=Number.isFinite(video.duration) && video.duration>0 ? video.duration : (sample.last-sample.first+1)/20/sample.speed;
  $('#timeline-cursor').style.left=`${Math.max(0,Math.min(100,video.currentTime/duration*100))}%`;
  const returning = video.currentTime + 1e-6 >= returnTime();
  $('#phase-badge').textContent=returning ? 'Return / evaluated' : 'Exploration';
  $('#phase-badge').style.background=returning ? '#795326d9' : '#173329c9';
}
function selectSample(task, initial=false) {
  currentTask=task;
  const sample=samples[task];
  document.querySelectorAll('[data-task]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.task===task)));
  $('#trajectory-title').textContent=sample.title;
  $('#trajectory-description').textContent=sample.description;
  $('#sample-range').textContent=sample.range;
  $('#sample-frames').textContent=`${sample.first}–${sample.last}`;
  $('#speed-badge').textContent=sample.speed===1 ? 'Original speed' : `${sample.speed}× playback`;
  $('#sample-caption').textContent=`${sample.name} · ${sample.route} · ${sample.range}${task==='turn'?' turn':' range'}. Displayed interval: frames ${sample.first}–${sample.last}; return starts at frame ${sample.returnFrame}.`;
  $('#exploration-segment').style.width=`${(sample.returnFrame-sample.first)/(sample.last-sample.first+1)*100}%`;
  video.setAttribute('aria-label',`${sample.name} ground-truth navigation recording at ${sample.speed} times original speed`);
  if (!initial) {
    video.pause();
    video.poster=`assets/${task}-poster.webp`;
    video.querySelector('source[type="video/mp4"]').src=`assets/${task}.mp4`;
    video.querySelector('source[type="video/webm"]').src=`assets/${task}.webm`;
    const download=video.querySelector('a');
    download.href=`assets/${task}.mp4`;
    download.textContent=`Download the ${sample.name} video.`;
    video.load();
  }
  drawMap(sample);
  updatePlayback();
}
document.querySelectorAll('[data-task]').forEach(button=>button.addEventListener('click',()=>selectSample(button.dataset.task)));
video.addEventListener('timeupdate',updatePlayback);
video.addEventListener('loadedmetadata',updatePlayback);
$('#jump-return').addEventListener('click',()=>{
  const selectedTask=currentTask;
  // Seek to the first encoded frame at or after the source return point.
  // A 3× replay may omit the exact source frame; avoid rounding just before it.
  const seek=()=> {if(selectedTask===currentTask){video.currentTime=Math.ceil(returnTime()*20)/20;updatePlayback();}};
  if(video.readyState>=1) seek(); else video.addEventListener('loadedmetadata',seek,{once:true});
});
selectSample('aba',true);

// Native dialog supplies focus trapping and Escape handling.
const dialog=$('#figure-dialog');
document.querySelectorAll('[data-zoom]').forEach(button=>button.addEventListener('click',()=>{
  $('#dialog-image').src=button.dataset.zoom;
  $('#dialog-image').alt=button.querySelector('img').alt;
  dialog.showModal();
  document.body.classList.add('dialog-open');
}));
$('.dialog-close').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',event=>{
  if(event.target!==dialog) return;
  const bounds=dialog.getBoundingClientRect();
  if(event.clientX<bounds.left || event.clientX>bounds.right || event.clientY<bounds.top || event.clientY>bounds.bottom) dialog.close();
});
dialog.addEventListener('close',()=>document.body.classList.remove('dialog-open'));

$('#copy-citation').addEventListener('click',async()=>{
  const text=$('#bibtex').textContent;
  try {
    if(navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else {
      const field=document.createElement('textarea');
      field.value=text;field.style.position='fixed';field.style.opacity='0';document.body.append(field);field.select();
      const copied=document.execCommand('copy');field.remove();
      if(!copied) throw new Error('Clipboard unavailable');
    }
    $('#copy-citation').textContent='Copied ✓';
    $('#copy-status').textContent='BibTeX copied to clipboard.';
    setTimeout(()=>{$('#copy-citation').textContent='Copy BibTeX';},2200);
  } catch {
    const range=document.createRange();range.selectNodeContents($('#bibtex'));
    const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);
    $('#copy-status').textContent='Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});

const navLinks=[...document.querySelectorAll('[data-section-link]')];
const sections=navLinks.map(link=>$(link.getAttribute('href')));
let scrollPending=false;
function updateSection(){
  const active=sections.filter(section=>section.getBoundingClientRect().top<=window.innerHeight*.38).at(-1);
  navLinks.forEach(link=>{if(active && link.hash===`#${active.id}`)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});
  scrollPending=false;
}
window.addEventListener('scroll',()=>{if(!scrollPending){scrollPending=true;requestAnimationFrame(updateSection);}},{passive:true});
window.addEventListener('resize',updateSection);
updateSection();
