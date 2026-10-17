/** Strength is relative, from the frozen pre-race joint forecast, never the result.
 * Learn a JOINT unordered top-two event, retaining the old conditional order/third.
 */
import {normalizeRace} from './cross_engine.mjs';
export const COEXISTENCE_VERSION='coexistence-v1';
export const FEATURE_NAMES=['pairP','additivePairP','winHigh','winLow','winProduct',
  'bothGivenEitherWins','lineWinHigh','lineWinLow','otherLineWin','sameLine',
  'highFront','highBante','lowFront','lowBante','highLineSize','lowLineSize',
  'highScoreCentered','lowScoreCentered','highWinRate','lowWinRate','highTop3','lowTop3',
  'highB','lowB','highNige','lowNige','highMakuri','lowMakuri','highSashi','lowSashi',
  'highMark','lowMark','frontBProduct','frontNigeProduct','otherFrontB',
  'lineRankHigh','lineRankLow','riderRankHigh','riderRankLow','lineWinGap','riderWinGap'];
export function coexistenceFeatures(input,forecast){
  const r=normalizeRace(input),cars=r.entries.map(e=>e.no).sort((a,b)=>a-b),win=new Map(cars.map(n=>[n,0]));
  const pair=new Map(),base=new Map();
  const key=(a,b)=>[a,b].sort((x,y)=>x-y).join('-');
  for(const [k,v] of Object.entries(forecast.probabilities)){
    const [a,b]=k.split('-').map(Number);win.set(a,win.get(a)+v);
    const p=key(a,b);pair.set(p,(pair.get(p)||0)+v);
    base.set(p,(base.get(p)||0)+(forecast.additiveControl?.probabilities?.[k]??v));
  }
  const lineWin=r.lines.map(l=>l.reduce((s,n)=>s+win.get(n),0));
  const rank=(v,all)=>1+all.filter(x=>x>v+1e-12).length;
  const lineRank=lineWin.map(v=>rank(v,lineWin)),riderRank=new Map(cars.map(n=>[n,rank(win.get(n),[...win.values()])]));
  const mean=r.entries.reduce((s,e)=>s+e.score,0)/cars.length,records=[];
  for(let i=0;i<cars.length;i++)for(let j=i+1;j<cars.length;j++){
    const ns=[cars[i],cars[j]].sort((a,b)=>win.get(b)-win.get(a)||a-b),[a,b]=ns;
    const x=r.riders.get(a),y=r.riders.get(b),pa=r.position.get(a),pb=r.position.get(b);
    const la=r.lines[pa.line],lb=r.lines[pb.line],fa=r.riders.get(la[0]),fb=r.riders.get(lb[0]);
    const other=r.lines.filter((_,li)=>li!==pa.line&&li!==pb.line),p=pair.get(key(a,b)),wa=win.get(a),wb=win.get(b);
    records.push({pair:[cars[i],cars[j]],key:key(a,b),probability:p,
      linePair:[pa.line,pb.line].sort((a,b)=>a-b),
      features:[p,base.get(key(a,b)),wa,wb,wa*wb,p/(wa+wb),lineWin[pa.line],lineWin[pb.line],
        other.reduce((s,l)=>s+l.reduce((t,n)=>t+win.get(n),0),0),+(pa.line===pb.line),
        +(pa.pos===1),+(pa.pos===2),+(pb.pos===1),+(pb.pos===2),la.length,lb.length,
        (x.score-mean)/5,(y.score-mean)/5,x.win_rate/100,y.win_rate/100,x.top3_rate/100,y.top3_rate/100,
        x.b/10,y.b/10,x.nige/10,y.nige/10,x.makuri/10,y.makuri/10,x.sashi/10,y.sashi/10,
        x.mark/10,y.mark/10,fa.b*fb.b/100,fa.nige*fb.nige/100,
        other.reduce((s,l)=>s+r.riders.get(l[0]).b,0)/10,
        lineRank[pa.line],lineRank[pb.line],riderRank.get(a),riderRank.get(b),
        Math.abs(lineWin[pa.line]-lineWin[pb.line]),wa-wb]});
  }
  return {records,lineWin,lineRank,riderRank,win};
}
function treeValue(tree,x){let i=0;while(!tree[i].leaf){const n=tree[i];i=x[n.feature]<=n.threshold?n.left:n.right;}return tree[i].value;}
export function applyCoexistence(input,forecast,bundle){
  if(!forecast||!bundle)return forecast;
  const r=normalizeRace(input),cls=/[ＳS]級/.test(r.race_type)?'S':'A',model=bundle.models?.[cls];
  if(bundle.version!==COEXISTENCE_VERSION||!model||bundle.featureNames?.join('|')!==FEATURE_NAMES.join('|'))throw Error('共存モデルの版・特徴が不正');
  if(forecast.source.id!=='conditional-pair-v3_'+cls||String(forecast.raceId)!==String(r.race_id))throw Error('共存モデルは同じレース・対象級の同時入着v3専用');
  const match=String(r.race_id).match(/^\d{2}(\d{4})(\d{2})(\d{2})/),date=match?.slice(1).join('-');
  if(!date||bundle.trainThrough>=date)throw Error('当日以降を学習した共存モデルは使えません');
  const pack=coexistenceFeatures(input,forecast);
  const probs=pack.records.map(z=>{const x=z.features.map(Math.fround);return 1/(1+Math.exp(-(model.intercept+model.trees.reduce((s,t)=>s+treeValue(t,x),0))));});
  const total=probs.reduce((s,v)=>s+v,0),weights=new Map(pack.records.map((z,i)=>[z.key,probs[i]/total/z.probability]));
  const probabilities={};for(const [k,v] of Object.entries(forecast.probabilities)){
    const [a,b]=k.split('-').map(Number);probabilities[k]=v*weights.get([a,b].sort((a,b)=>a-b).join('-'));
  }
  return {...forecast,probabilities,coexistenceBase:forecast,coexistenceGateEnabled:bundle.hardGateEnabled===true,
    source:{...forecast.source,id:COEXISTENCE_VERSION+'_'+cls,trainThrough:bundle.trainThrough,
      scope:cls==='S'?'S級7車（学習データはF1）':'A級1・2班7車',
      caveat:'ライン・選手の相対強度と同時1・2着を学習。条件付きの予測であり、非成立の断定ではない'}};
}

/** Exact disjoint event sums. A weak-strong alternative is not inferred from NOT SS. */
export function auditCoexistence(input,forecast){
  if(!forecast)return null;
  const r=normalizeRace(input),base=forecast.coexistenceBase||forecast,pack=coexistenceFeatures(input,base);
  const pairs=new Map();let sameLine=0;
  for(const [key,v] of Object.entries(forecast.probabilities)){
    const [a,b]=key.split('-').map(Number),la=r.position.get(a).line,lb=r.position.get(b).line;
    if(la===lb)sameLine+=v;
    const k=[a,b].sort((a,b)=>a-b).join('-');pairs.set(k,(pairs.get(k)||0)+v);
  }
  const orderedLines=r.lines.map((cars,line)=>({line,cars,win:pack.lineWin[line],rank:pack.lineRank[line]})).sort((a,b)=>b.win-a.win);
  const lineRows=[];
  for(let i=0;i<r.lines.length;i++)for(let j=i+1;j<r.lines.length;j++){
    const sorted=l=>[...r.lines[l]].sort((a,b)=>pack.win.get(b)-pack.win.get(a));
    const aa=sorted(i),bb=sorted(j),unique=pack.win.get(aa[0])>(pack.win.get(aa[1])??-1)+1e-12&&pack.win.get(bb[0])>(pack.win.get(bb[1])??-1)+1e-12;
    const playerPairs=aa.flatMap(a=>bb.map(b=>({pair:[a,b],probability:pairs.get([a,b].sort((x,y)=>x-y).join('-')),
      kind:a===aa[0]&&b===bb[0]?'SS':a===aa[0]||b===bb[0]?'SW':'WW'}))).sort((a,b)=>b.probability-a.probability);
    const ss=playerPairs.find(z=>z.kind==='SS'),sw=playerPairs.filter(z=>z.kind==='SW')[0];
    lineRows.push({lines:[i,j],ranks:[pack.lineRank[i],pack.lineRank[j]],probability:playerPairs.reduce((s,z)=>s+z.probability,0),
      playerPairs,strongPair:ss,bestStrongWeak:sw,playerSwitch:unique&&!!sw&&sw.probability>ss.probability+1e-12,
      playerStrengthUnique:unique});
  }
  lineRows.sort((a,b)=>b.probability-a.probability);
  const unique=orderedLines.length===3&&orderedLines[1].win>orderedLines[2].win+1e-12;
  const ss=unique?lineRows.find(z=>z.lines.every(l=>l!==orderedLines[2].line)):null;
  const sw=unique?lineRows.filter(z=>z.lines.includes(orderedLines[2].line)).sort((a,b)=>b.probability-a.probability)[0]:null;
  const cars=[...pack.win.keys()].sort((a,b)=>pack.win.get(b)-pack.win.get(a));
  const playerUnique=pack.win.get(cars[1])>pack.win.get(cars[2])+1e-12;
  const strongCars=new Set(cars.slice(0,2));
  const globalPairs=pack.records.map(z=>({...z,probability:pairs.get(z.key),
    kind:z.pair.every(n=>strongCars.has(n))?'SS':z.pair.some(n=>strongCars.has(n))?'SW':'WW'}));
  const globalSS=globalPairs.find(z=>z.kind==='SS'),globalSW=globalPairs.filter(z=>z.kind==='SW'&&z.linePair[0]!==z.linePair[1]).sort((a,b)=>b.probability-a.probability)[0];
  const globalPlayers={strongCars:[...strongCars],strongPair:globalSS,bestStrongWeak:globalSW,
    unique:playerUnique,strongPairSameLine:globalSS.linePair[0]===globalSS.linePair[1],
    switch:playerUnique&&globalSS.linePair[0]!==globalSS.linePair[1]&&!!globalSW&&globalSW.probability>globalSS.probability+1e-12};
  return {lineRows,orderedLines,sameLine,strongStrong:ss,bestStrongWeak:sw,globalPlayers,
    lineSwitch:!!ss&&!!sw&&sw.probability>ss.probability+1e-12,lineStrengthUnique:unique,
    definition:'強弱は出走前v3モデルの単独1着周辺確率による相対順位。絶対的な強強弱分類ではない',
    note:'同時成立が低いことだけでは強弱成立を意味しない。筋決着も独立に比較する'};
}

/** Conditional switches are comparisons of separate joint events, not fixed roles. */
export function coexistenceEligibility(input,candidates,forecast){
  const audit=auditCoexistence(input,forecast),active=forecast?.source?.id?.startsWith(COEXISTENCE_VERSION+'_')&&forecast.coexistenceGateEnabled===true;
  const setKey=xs=>[...xs].sort((a,b)=>a-b).join('-');
  const rows=candidates.map(c=>{
    const reasons=[],line= audit?.lineRows.find(z=>setKey(z.lines)===setKey(c.lines));
    if(active&&audit.lineSwitch&&setKey(c.lines)===setKey(audit.strongStrong.lines))reasons.push('LINE_STRONG_WEAK_MORE_JOINT');
    if(active&&line?.playerSwitch&&setKey(c.firstPair)===setKey(line.strongPair.pair))reasons.push('WITHIN_LINE_PAIR_STRONG_WEAK_MORE_JOINT');
    if(active&&audit.globalPlayers.switch&&setKey(c.firstPair)===setKey(audit.globalPlayers.strongPair.pair))reasons.push('GLOBAL_PLAYER_STRONG_WEAK_MORE_JOINT');
    return {signature:c.signature,eligible:reasons.length===0,reasons};
  });
  return {active,audit,rows};
}
