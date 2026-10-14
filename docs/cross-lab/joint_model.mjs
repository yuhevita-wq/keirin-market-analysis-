/** A/S separate joint top-two model; third is conditional on that SAME ordered pair. */
import {normalizeRace} from './cross_engine.mjs';
export const JOINT_VERSION='conditional-pair-v3';
const fields=['score','win_rate','top3_rate','b','nige','makuri','sashi','mark','age'];
const scales=[5,20,25,6,6,6,6,6,12];
const clip=x=>Math.max(-3,Math.min(3,x));
const softmax=xs=>{const m=Math.max(...xs),e=xs.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s);};
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export function featurePack(input){
  const r=normalizeRace(input);
  if(r.entries.length!==7)throw Error('同時入着v3は7車立てが対象');
  for(const e of r.entries){
    if(fields.some(k=>e[k]==null||e[k]===''||!Number.isFinite(Number(e[k]))||Number(e[k])<0)||
       e.score<=0||e.age<=0||e.win_rate>100||e.top3_rate>100||!['逃','両','追'].includes(e.style))
      throw Error(e.no+'番の得点・年齢・勝率・3連対率・脚質・決まり手が不足（0得点は未評価）');
  }
  const cars=r.entries.map(e=>e.no).sort((a,b)=>a-b);
  const means=fields.map(k=>r.entries.reduce((s,e)=>s+Number(e[k]),0)/7);
  const raw=no=>r.riders.get(no),pos=no=>r.position.get(no),line=no=>r.lines[pos(no).line];
  const front=no=>raw(line(no)[0]);
  const stats=no=>[...fields.map((k,j)=>clip((Number(raw(no)[k])-means[j])/scales[j])),+(raw(no).style==='逃'),+(raw(no).style==='両')];
  const role=no=>[+(pos(no).pos===1),+(pos(no).pos===2),+(pos(no).pos>2),+(line(no).length===1),line(no).length/3];
  const load=e=>Math.min(3,e.b/6), esc=e=>e.nige/(e.nige+e.makuri+e.sashi+e.mark+1);
  const pressure=no=>r.lines.filter(l=>l[0]!==line(no)[0]).reduce((s,l)=>s+load(raw(l[0]))*esc(raw(l[0])),0)/3;
  const pairs=[],pairFeatures=[],thirdFeatures=[];
  for(const a of cars)for(const b of cars){
    if(a===b)continue;
    const pa=pos(a),pb=pos(b),same=+(pa.line===pb.line),diff=1-same;
    const af=+(pa.pos===1),bf=+(pb.pos===1),ab=+(pa.pos===2),bb=+(pb.pos===2);
    const fa=front(a),fb=front(b),la=load(fa),lb=load(fb),ea=esc(fa),eb=esc(fb);
    const base=[...stats(a),...stats(b),...role(a),...role(b),same,
      same*+(pb.pos===pa.pos+1),same*+(pa.pos===pb.pos+1)];
    const interactions=[
      diff*af*bf*la*lb,diff*af*bf*ea*eb,diff*ab*bb*la*lb,
      af*pressure(a),bf*pressure(b),ab*la*pressure(a),bb*lb*pressure(b),
      diff*af*raw(b).makuri/6,diff*bf*raw(a).makuri/6,
      diff*ab*la*raw(b).makuri/6,diff*bb*lb*raw(a).makuri/6,
      same*af*bb*ea,same*ab*bf*ea,
      diff*raw(a).sashi*raw(b).sashi/36,
      diff*raw(a).mark*raw(b).mark/36,
      diff*af*bb*la*eb,diff*ab*bf*lb*ea,
      diff*ab*bb*ea*eb
    ].map(clip);
    pairs.push([a,b]);pairFeatures.push([...base,...interactions]);
    thirdFeatures.push(cars.filter(c=>c!==a&&c!==b).map(c=>{
      const pc=pos(c),sc1=+(pc.line===pa.line),sc2=+(pc.line===pb.line),fc=front(c);
      return [...stats(c),...role(c),sc1,sc2,sc1*+(pc.pos===pa.pos+1),sc2*+(pc.pos===pb.pos+1),
        sc1*+(pc.pos<pa.pos),sc2*+(pc.pos<pb.pos),load(fc)*pressure(c),
        +(pc.pos===2)*load(fc),+(pc.pos===1)*pressure(c),same*+(pc.line!==pa.line),
        diff*+(pc.line!==pa.line&&pc.line!==pb.line),ab*sc1,bb*sc2].map(clip);
    }));
  }
  return {cars,pairs,pairFeatures,thirdFeatures,basePairFeatureCount:35};
}
export function predictJoint(input,bundle){
  const race=normalizeRace(input),cls=/[ＳS]級/.test(race.race_type)?'S':/[ＡA]級/.test(race.race_type)?'A':null;
  if(!cls||/[ＡA]級.*(?:チャレンジ|チ予|チ般|チ選|チ準|チ決)/.test(race.race_type)||race.entries.some(e=>/A3/.test(e.class||'')))
    throw Error('同時入着v3の対象級を確認してください（A級1・2班／S級）');
  const model=bundle?.models?.[cls];
  if(bundle?.version!==JOINT_VERSION||!model)throw Error('対象級の同時入着モデルがありません');
  const pack=featurePack(input),{cars,pairs,pairFeatures,thirdFeatures}=pack;
  if(model.pair.length!==pairFeatures[0].length||model.third.length!==thirdFeatures[0][0].length||
     [...model.pair,...model.third].some(x=>!Number.isFinite(x)))throw Error('同時入着モデルの特徴・係数が不正');
  const pairP=softmax(pairFeatures.map(x=>dot(x,model.pair))),probabilities={};
  pairs.forEach(([a,b],i)=>{
    const rest=cars.filter(c=>c!==a&&c!==b),thirdP=softmax(thirdFeatures[i].map(x=>dot(x,model.third)));
    rest.forEach((c,j)=>probabilities[[a,b,c].join('-')]=pairP[i]*thirdP[j]);
  });
  return {raceId:race.race_id,source:{id:JOINT_VERSION+'_'+cls,trainThrough:bundle.trainThrough,
    scope:cls+'級・7車',status:'EXPERIMENTAL',selectionObjective:'HIT_PROBABILITY',
    caveat:'過去データで学習した同時入着モデル。収益性は未実証'},probabilities};
}

/** All quantities share one complete race distribution; never P(A wins)*P(B wins). */
export function jointDiagnostics(candidate,forecast){
  if(!forecast)return null;
  const p=forecast.probabilities,heads=candidate.firstPair,thirds=new Set(candidate.remaining);
  return heads.map((a,index)=>{
    const b=heads[1-index];let winner=0,pair=0,core=0;
    for(const [key,v] of Object.entries(p)){
      const [i,j,k]=key.split('-').map(Number);
      if(i===a){winner+=v;if(j===b){pair+=v;if(thirds.has(k))core+=v;}}
    }
    return {first:a,second:b,winProbability:winner,orderedPairProbability:pair,
      secondGivenFirst:winner>0?pair/winner:null,coreThirdGivenPair:pair>0?core/pair:null,
      coreProbability:core};
  });
}
