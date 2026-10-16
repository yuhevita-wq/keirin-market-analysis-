/** Card-only conditional head roles. Descriptive historical returns are not live EV. */
import {normalizeRace} from './cross_engine.mjs';
export const ROLE_LABELS=['先頭×先頭','A先頭×B番手','A番手×B先頭','番手×番手','３番手が１・２着'];
export function roleFeatures(input,indexes){
 const r=normalizeRace(input), ids=[...indexes].sort((i,j)=>r.lines[j].length-r.lines[i].length||r.lines[i][0]-r.lines[j][0]);
 const [a,b]=ids.map(i=>r.lines[i]), outside=r.lines.find((_,i)=>!ids.includes(i));
 if(r.entries.length!==7||r.lines.map(l=>l.length).sort().join()!=='2,2,3'||ids.length!==2||!outside)throw Error('役割条件は7車3–2–2が対象');
 const [af,ab,bf,bb,of]=[a[0],a[1],b[0],b[1],outside[0]].map(n=>r.riders.get(n));
 const v=(e,k)=>e[k]===null||e[k]===undefined||e[k]===''?NaN:Number(e[k]);
 return {lines:ids,fronts:[a[0],b[0]],bantes:[a[1],b[1]],features:[v(af,'score')-v(ab,'score'),v(bf,'score')-v(bb,'score'),v(af,'score')-v(bf,'score'),v(ab,'score')-v(bb,'score'),...[af,bf,of].map(e=>v(e,'b')),...[af,bf].map(e=>v(e,'nige')),...[af,bf].map(e=>v(e,'makuri')),...[ab,bb].map(e=>v(e,'sashi')),...[ab,bb].map(e=>v(e,'mark')),v(of,'score')-(v(af,'score')+v(bf,'score'))/2,a.length,b.length,...[af,bf,ab,bb].map(e=>v(e,'top3_rate'))]};
}
export function predictRoleConditions(input,indexes,bundle){
 const r=normalizeRace(input),cls=/チャレンジ|チ予|チ般|チ選|チ準|チ決/.test(r.race_type)||r.entries.some(e=>/A3/.test(e.class||''))?'A級チャレンジ':/[ＳS]級/.test(r.race_type)?'S級':/[ＡA]級/.test(r.race_type)?'A級1・2班':null;
 const date=input.race_date||input.raceDate||String(r.race_id).replace(/^\d{2}(\d{4})(\d{2})(\d{2}).*$/,'$1-$2-$3');
 if(/^\d{4}-\d{2}-\d{2}$/.test(date)&&date<=bundle.trainThrough)throw Error('学習終了日以前のレースに役割予測は使用できません');
 if(bundle.version!=='role-conditions-v1'||!bundle.models[cls])throw Error('役割条件の対象級を確認してください');
 const pack=roleFeatures(input,indexes),m=bundle.models[cls],missing=pack.features.map((v,i)=>Number.isFinite(v)?null:i).filter(v=>v!==null),x=pack.features.map((v,i)=>Number.isFinite(v)?v:m.medians[i]).map(Math.fround);
 let i=0;const conditions=[];
 while(m.nodes[i].left>=0){const n=m.nodes[i],left=x[n.feature]<=n.threshold;conditions.push({feature:bundle.features[n.feature],value:x[n.feature],threshold:n.threshold,operator:left?'≤':'>',imputed:missing.includes(n.feature)});i=left?n.left:n.right;}
 const n=m.nodes[i],shape=pack.features[16]===3?'3x2':'2x2';return {...pack,class:cls,leaf:i,conditions,missingFeatures:missing.map(k=>bundle.features[k]),trainCrosses:n.n,shape,trainShapeCrosses:m.trainCrossesByShape?.[i]?.[shape]??null,probabilities:n.probability,base:m.base,validation:m.validationByShape?.[i]?.[shape]??null,
  target:'この２ラインが１・２着を分け、３着もこの２ラインから出た場合の１・２着の役割',status:'CONDITION_COMPARISON',trainThrough:bundle.trainThrough};
}
/** Optional research challenger: preserve each two-line event mass, shift roles only. */
export function roleAdjustedForecast(input,forecast,bundle){
 const r=normalizeRace(input),p=forecast.probabilities,out={...p};
 for(let a=0;a<r.lines.length;a++)for(let b=a+1;b<r.lines.length;b++){
  const model=predictRoleConditions(input,[a,b],bundle),groups=Array.from({length:5},()=>[]),mass=Array(5).fill(0);
  for(const [key,v] of Object.entries(p)){
   const t=key.split('-').map(Number),ps=t.map(n=>r.position.get(n));
   if(ps[0].line===ps[1].line||ps.some(x=>![a,b].includes(x.line)))continue;
   const pa=ps.find((x,i)=>i<2&&x.line===model.lines[0]).pos-1,pb=ps.find((x,i)=>i<2&&x.line===model.lines[1]).pos-1,k=pa<2&&pb<2?2*pa+pb:4;
   groups[k].push(key);mass[k]+=v;
  }
  const total=mass.reduce((s,v)=>s+v,0);if(!total)continue;
  // Zero support cannot be repaired by multiplying: retain the original pair.
  if(mass.some((v,k)=>v===0&&model.probabilities[k]>0))continue;
  groups.forEach((keys,k)=>keys.forEach(key=>out[key]=p[key]*total*model.probabilities[k]/mass[k]));
 }
 return {...forecast,source:{...forecast.source,id:forecast.source.id+'+role-conditions-v1',status:'RESEARCH_CHALLENGER'},probabilities:out};
}
