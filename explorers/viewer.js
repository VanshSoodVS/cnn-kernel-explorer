'use strict';
const D=window.KERNEL_DATA, $=id=>document.getElementById(id);
function typed(text,Type){const s=atob(text),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return new Type(a.buffer)}
const raw=typed(D.raw,D.raw_type==='f32'?Float32Array:Float64Array), N=D.n;
const coords={};for(const [k,v] of Object.entries(D.coordinates))coords[k]=typed(v,Int32Array);
const metrics={};for(const [name,m] of Object.entries(D.metrics))metrics[name]={i:typed(m.i,Int32Array),j:typed(m.j,Int32Array),distance:typed(m.distance,Float64Array),a:typed(m.children1,Int32Array),b:typed(m.children2,Int32Array),height:typed(m.height,Float64Array),minima:typed(m.node_min_id,Int32Array),valid:typed(m.valid,Uint8Array)};
let metric='shape',cutoff=.05,query=0,labels,counts,matches=[],csvText='';
let mapView='partners';
let overviewColumns=7;
const partnerIds=new Set();
let matchingKernels=[];
const metricNames={shape:'Signed shape',weights:'Original weight values',centered:'Centered shape'};
const metricNotes={shape:'Signed shape compares the pattern of the nine weights after removing overall magnitude. Signs and spatial positions are preserved.',weights:'Original weight values compares the actual nine weights, relative to the smaller kernel magnitude. Two kernels with the same shape but different strengths can fail this test.',centered:'Centered shape removes each kernel\'s average weight and magnitude before comparing patterns. It helps separate a shared average level from a shared spatial pattern.'};
const layerPlot={x:64,y:20,size:512};
const norms=new Float64Array(N),means=new Float64Array(N);for(let i=0;i<N;i++){let s=0,n=0;for(let p=0;p<9;p++){const x=raw[i*9+p];s+=x;n+=x*x}norms[i]=Math.sqrt(n);means[i]=s/9}
$('title').textContent=D.label+' · '+(D.stage==='initial'?'initial':'trained')+' kernels';document.title=$('title').textContent;
const otherStage=D.stage==='initial'?'trained':'initial';
const switchLink=document.createElement('a');switchLink.href=D.snapshot.replace(/_(initial|trained)$/, '_'+otherStage)+'.html';switchLink.textContent='Compare with '+otherStage+' snapshot →';$('snapshot-switch').append(switchLink);
$('snapshot-note').textContent=(D.stage==='initial'?'Initial snapshot: random starting weights before training.':'Trained snapshot: the checkpoint with the best validation accuracy, which may be earlier than the final epoch.')+' '+N.toLocaleString()+' kernels across '+D.layers.length+' convolutions.';
$('id-help').textContent='Enter a whole-number ID from 0 to '+(N-1)+', then select it or press Enter. This selects a location in this snapshot, even if it has no partners.';
$('kernel').max=N-1;for(const l of D.layers){const o=document.createElement('option');o.value=l.index;o.textContent=l.index+' '+l.name;$('layer').append(o)}
function integer(value,low,high){let n=Math.trunc(Number(value));return Math.max(low,Math.min(high,Number.isFinite(n)?n:low))}
function find(parent,x){let r=x;while(parent[r]!==r)r=parent[r];while(parent[x]!==x){const next=parent[x];parent[x]=r;x=next}return r}
function partition(){const m=metrics[metric],p=new Int32Array(N);for(let i=0;i<N;i++)p[i]=i;for(let k=0;k<m.height.length&&m.height[k]<=cutoff;k++){const a=find(p,m.minima[m.a[k]]),b=find(p,m.minima[m.b[k]]);if(a!==b)p[Math.max(a,b)]=Math.min(a,b)}labels=new Int32Array(N);counts=new Int32Array(N);for(let i=0;i<N;i++){labels[i]=m.valid[i]?find(p,i):-1;if(labels[i]>=0)counts[labels[i]]++}const active=new Set();for(let k=0;k<m.distance.length;k++){if(m.distance[k]<=cutoff){active.add(m.i[k]);active.add(m.j[k])}}matchingKernels=Array.from(active).sort((a,b)=>a-b);$('metric-help').textContent=metricNotes[metric]}
function matchPosition(id){let lo=0,hi=matchingKernels.length;while(lo<hi){const mid=(lo+hi)>>>1;if(matchingKernels[mid]<id)lo=mid+1;else hi=mid}return lo}
function navigationStatus(){const n=matchingKernels.length,pos=matchPosition(query);for(const id of ['first-match','previous-match','next-match'])$(id).disabled=n===0;$('navigation-status').textContent=n===0?'No kernels have a partner with this comparison and cutoff. Choose another comparison or cutoff to explore.':matchingKernels[pos]===query?'Matching kernel '+(pos+1).toLocaleString()+' of '+n.toLocaleString()+' (ID '+query+').':'Kernel '+query+' has no partners here. Browse '+n.toLocaleString()+' kernels that do using First, Previous or Next.'}
function stepMatch(direction){const n=matchingKernels.length;if(!n)return;const pos=matchPosition(query),onMatch=matchingKernels[pos]===query;let next=direction>0?pos+(onMatch?1:0):pos-1;const wrapped=next<0||next>=n;next=(next+n)%n;select(matchingKernels[next]);if(wrapped)$('navigation-status').textContent+=' Wrapped to the '+(direction>0?'first':'last')+' matching kernel.'}
function color(id){if(mapView==='partners'){if(id===query)return [208,32,32];if(partnerIds.has(id))return [0,120,160];return [235,239,242]}const label=labels[id];if(label<0)return [245,245,245];if(counts[label]<2)return [225,231,235];let h=(label*2654435761>>>0)%360,s=.62,l=.58;const a=s*Math.min(l,1-l);return [0,8,4].map(n=>{const k=(n+h/30)%12;return Math.round(255*(l-a*Math.max(-1,Math.min(k-3,9-k,1))))})}
function layerImage(layer){const image=new ImageData(layer.channels[1],layer.channels[0]);for(let o=0;o<layer.channels[0];o++)for(let i=0;i<layer.channels[1];i++){const id=layer.offset+o*layer.channels[1]+i, rgb=color(id),p=(o*layer.channels[1]+i)*4;image.data[p]=rgb[0];image.data[p+1]=rgb[1];image.data[p+2]=rgb[2];image.data[p+3]=255}return image}
function paintMap(ctx,layer,x,y,w,h){const off=document.createElement('canvas');off.width=layer.channels[1];off.height=layer.channels[0];off.getContext('2d').putImageData(layerImage(layer),0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(off,x,y,w,h);if(coords.layer[query]===layer.index){const o=coords.output_channel[query],i=coords.input_channel[query];ctx.strokeStyle='#d02020';ctx.lineWidth=2;ctx.strokeRect(x+i*w/layer.channels[1],y+o*h/layer.channels[0],w/layer.channels[1],h/layer.channels[0])}}
function overview(){
 const c=$('overview'),container=c.parentElement;
 const available=container.clientWidth||Math.min(980,window.innerWidth-74);
 overviewColumns=Math.max(2,Math.min(7,Math.floor(available/140)));
 c.width=overviewColumns*140;c.height=Math.ceil(D.layers.length/overviewColumns)*150+12;
 const ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.font='13px system-ui';
 for(const l of D.layers){const x=(l.index%overviewColumns)*140+4,y=Math.floor(l.index/overviewColumns)*150+20;ctx.fillStyle='#26323b';ctx.fillText(l.name,x,y-5);paintMap(ctx,l,x,y,128,128)}
}
function weights(){const c=$('weights'),ctx=c.getContext('2d'),values=Array.from(raw.slice(query*9,query*9+9)),max=Math.max(...values.map(Math.abs),1e-30);ctx.font='13px monospace';ctx.textAlign='center';for(let k=0;k<9;k++){const v=values[k]/max,light=95-Math.abs(v)*45;ctx.fillStyle=`hsl(${v<0?213:5} 65% ${light}%)`;ctx.fillRect((k%3)*100,Math.floor(k/3)*100,100,100);ctx.fillStyle='#15222c';ctx.fillText(values[k].toPrecision(6),(k%3)*100+50,Math.floor(k/3)*100+54)}$('weight-values').textContent=Array.from({length:3},(_,r)=>values.slice(r*3,r*3+3).map(x=>x.toPrecision(17)).join(', ')).join('\n');$('kernel-info').textContent='Norm '+norms[query].toPrecision(17)+'; mean '+means[query].toPrecision(17)}
function coordinateHelp(){const l=D.layers[Number($('layer').value)];$('coordinate-help').textContent=l.name+': output channel 0–'+(l.channels[0]-1)+' (row), input channel 0–'+(l.channels[1]-1)+' (column). Choose the three coordinates, then click Select coordinates.'}
function drawLayer(layer){const c=$('layer-map'),ctx=c.getContext('2d'),p=layerPlot;ctx.clearRect(0,0,c.width,c.height);paintMap(ctx,layer,p.x,p.y,p.size,p.size);ctx.fillStyle='#26323b';ctx.font='24px system-ui';ctx.textAlign='center';ctx.fillText('Columns: input channel →',p.x+p.size/2,586);ctx.save();ctx.translate(18,p.y+p.size/2);ctx.rotate(-Math.PI/2);ctx.fillText('Rows: output channel',0,0);ctx.restore();for(let axis=0;axis<2;axis++){const n=layer.channels[axis],ticks=n<=4?Array.from({length:n},(_,i)=>i):[0,16,32,48,n-1];ctx.font='22px system-ui';for(const t of ticks){const position=(t+.5)*p.size/n;ctx.textAlign=axis===0?'right':'center';ctx.fillText(String(t),axis===0?p.x-9:p.x+position,axis===0?p.y+position+5:p.y+p.size+24)}}}
function coordinates(){const l=D.layers[coords.layer[query]];$('kernel').value=query;$('layer').value=l.index;$('output').max=l.channels[0]-1;$('input').max=l.channels[1]-1;$('output').value=coords.output_channel[query];$('input').value=coords.input_channel[query];$('coordinates').textContent='Kernel '+query+' — '+l.name;$('layer-help').textContent='Selected cell: output row '+coords.output_channel[query]+', input column '+coords.input_channel[query]+'. This layer has '+l.channels[0]+' rows × '+l.channels[1]+' columns. '+matches.filter(item=>coords.layer[item.id]===l.index).length+' of the selected kernel’s '+matches.length+' direct partners are in this layer.';coordinateHelp();drawLayer(l);weights()}
const header=['query_kernel_id','matched_kernel_id','layer','block','role','output_channel','input_channel','comparison','cutoff','distance','raw_weight_euclidean','raw_relative_difference','query_norm','matched_norm','query_mean','matched_mean',...Array.from({length:9},(_,i)=>'query_weight_'+i),...Array.from({length:9},(_,i)=>'matched_weight_'+i)];
function precise(v){return typeof v==='number'?(Number.isInteger(v)?String(v):v.toPrecision(17)):String(v)}
function csv(){const lines=[header.join(',')];for(const item of matches){const id=item.id;let sq=0;for(let k=0;k<9;k++)sq+=(raw[query*9+k]-raw[id*9+k])**2;const absolute=Math.sqrt(sq),relative=absolute/Math.min(norms[query],norms[id]);lines.push([query,id,coords.layer[id],coords.block[id],coords.role[id],coords.output_channel[id],coords.input_channel[id],metric,cutoff,item.distance,absolute,relative,norms[query],norms[id],means[query],means[id],...raw.slice(query*9,query*9+9),...raw.slice(id*9,id*9+9)].map(precise).join(','))}return lines.join('\r\n')+'\r\n'}
function queryMatches(){
 const m=metrics[metric];matches=[];
 for(let k=0;k<m.distance.length;k++){
  if(m.distance[k]<=cutoff&&(m.i[k]===query||m.j[k]===query))matches.push({id:m.i[k]===query?m.j[k]:m.i[k],distance:m.distance[k]});
 }
 matches.sort((a,b)=>a.distance-b.distance||a.id-b.id);
 const body=$('matches');body.replaceChildren();const cards=$('match-cards');cards.replaceChildren();
 for(const item of matches.slice(0,500)){
  const id=item.id,tr=document.createElement('tr'),layer=D.layers[coords.layer[id]];
  const values=[id,layer.index+': '+layer.name,coords.block[id]?'B'+String(coords.block[id]).padStart(2,'0'):'Stem',['stem','conv1','conv2'][coords.role[id]],coords.output_channel[id],coords.input_channel[id],item.distance.toPrecision(17),labels[id]===labels[query]?'Yes':'No'];
  for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td)}
  const td=document.createElement('td'),button=document.createElement('button');button.textContent='Select '+id;button.onclick=()=>select(id);td.append(button);tr.append(td);body.append(tr);
  const card=document.createElement('article');card.className='match-card';
  const head=document.createElement('div');head.className='match-card-head';
  const heading=document.createElement('h3');heading.textContent='Partner '+id;
  const pick=document.createElement('button');pick.textContent='Select '+id;pick.onclick=()=>select(id);head.append(heading,pick);card.append(head);
  const location=document.createElement('p');location.textContent=layer.name+' · output row '+coords.output_channel[id]+' · input column '+coords.input_channel[id];card.append(location);
  const facts=document.createElement('dl');
  for(const [name,value] of [['Distance',item.distance.toPrecision(17)],['Tight group',labels[id]===labels[query]?'Same as selected kernel':'Different from selected kernel'],['Block / role',values[2]+' / '+values[3]]]){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=name;dd.textContent=value;if(name==='Distance')dd.className='distance-value';facts.append(dt,dd)}
  card.append(facts);cards.append(card);

 }
 $('match-status').textContent=matches.length?matches.length+' qualifying '+(matches.length===1?'partner. ':'partners. ')+(matches.length>500?'Cards and table display 500; the complete CSV contains every partner.':'Cards, table and complete CSV contain every partner.'):'No matching partner for this kernel. '+(matchingKernels.length?'Use Next matching kernel to find one elsewhere. ':'No kernels match anywhere with these settings; choose another comparison or cutoff. ')+'The CSV contains its header only.';
 $('table-caption').textContent='Partners of kernel '+query+' — '+metricNames[metric]+', cutoff '+cutoff;
 csvText=csv();$('csv').value=csvText;$('csv').hidden=true;$('export-status').textContent='';
 const size=labels[query]>=0?counts[labels[query]]:0;
 $('summary').textContent='Selected kernel '+query+' · '+matches.length+' direct '+(matches.length===1?'partner':'partners')+' · '+(size?'tight group size '+size+' (including this kernel)':'comparison undefined for this kernel')+' · '+metricNames[metric]+' ≤ '+cutoff;
}
function maps(){coordinates();overview();$('partner-legend').hidden=mapView!=='partners';$('map-legend').textContent=mapView==='partners'?'Red = selected kernel; blue = its '+matches.length+' direct matching partners across all convolutions; gray = all other kernels. Highlights update when the kernel, comparison or cutoff changes.':'Group colors show all tight groups and stay the same when only the selection changes. Gray = singleton; red outline = selected kernel. Switch to Selected kernel and partners to highlight its direct matches.'}
function select(id){query=integer(id,0,N-1);queryMatches();partnerIds.clear();for(const item of matches)partnerIds.add(item.id);maps();navigationStatus()}
function controls(){metric=$('metric').value;cutoff=Number($('cutoff').value);partition();select(query)}
$('select-id').onclick=()=>select($('kernel').value);$('kernel').onkeydown=e=>{if(e.key==='Enter')select($('kernel').value)};
$('select-coordinates').onclick=()=>{const l=D.layers[Number($('layer').value)],o=integer($('output').value,0,l.channels[0]-1),i=integer($('input').value,0,l.channels[1]-1);select(l.offset+o*l.channels[1]+i)};
$('layer').onchange=()=>{const l=D.layers[Number($('layer').value)];$('output').max=l.channels[0]-1;$('input').max=l.channels[1]-1;$('output').value=integer($('output').value,0,l.channels[0]-1);$('input').value=integer($('input').value,0,l.channels[1]-1);coordinateHelp()};
$('metric').onchange=controls;$('cutoff').onchange=controls;
$('first-match').onclick=()=>{if(matchingKernels.length)select(matchingKernels[0])};
$('previous-match').onclick=()=>stepMatch(-1);$('next-match').onclick=()=>stepMatch(1);
$('map-view').onchange=()=>{mapView=$('map-view').value;maps()};
function canvasPoint(c,e){const r=c.getBoundingClientRect(),style=typeof getComputedStyle==='function'?getComputedStyle(c):{},left=parseFloat(style.borderLeftWidth)||0,right=parseFloat(style.borderRightWidth)||0,top=parseFloat(style.borderTopWidth)||0,bottom=parseFloat(style.borderBottomWidth)||0;return {x:(e.clientX-r.left-left)*c.width/(r.width-left-right),y:(e.clientY-r.top-top)*c.height/(r.height-top-bottom)}}
$('overview').onclick=e=>{const c=e.currentTarget,{x,y}=canvasPoint(c,e),col=Math.floor(x/140),row=Math.floor(y/150),l=D.layers[row*overviewColumns+col];if(col>=overviewColumns||!l)return;const xx=x-col*140-4,yy=y-row*150-20;if(xx<0||xx>=128||yy<0||yy>=128)return;select(l.offset+integer(Math.floor(yy/128*l.channels[0]),0,l.channels[0]-1)*l.channels[1]+integer(Math.floor(xx/128*l.channels[1]),0,l.channels[1]-1))};
$('overview-details').ontoggle=()=>{if($('overview-details').open)overview()};
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(overview,120)});

$('layer-map').onclick=e=>{const c=e.currentTarget,point=canvasPoint(c,e),l=D.layers[coords.layer[query]],p=layerPlot,x=point.x-p.x,y=point.y-p.y;if(x<0||y<0||x>=p.size||y>=p.size)return;const i=Math.floor(x/p.size*l.channels[1]),o=Math.floor(y/p.size*l.channels[0]);select(l.offset+o*l.channels[1]+i)};
$('preview').onclick=()=>{$('csv').hidden=false;$('csv').value=csvText;$('export-status').textContent='Complete preview: '+matches.length+' data rows and one header; all saved distance precision retained.'};
$('copy').onclick=async()=>{$('csv').hidden=false;$('csv').value=csvText;let ok=false;try{await navigator.clipboard.writeText(csvText);ok=true}catch(error){$('csv').focus();$('csv').select();try{ok=document.execCommand('copy')}catch(ignore){}}$('export-status').textContent=ok?'Complete CSV copied: '+matches.length+' data rows.':'Automatic copy was unavailable. The complete CSV is selected in the preview for manual copying.'};
$('download').onclick=()=>{const url=URL.createObjectURL(new Blob([csvText],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=D.snapshot+'_'+metric+'_'+cutoff+'_kernel'+query+'.csv';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);$('export-status').textContent='CSV download requested for '+matches.length+' data rows. Preview and copy provide the same complete text.'};
partition();select(matchingKernels.length?matchingKernels[0]:0);
window.KERNEL_VIEWER_READY=true;
