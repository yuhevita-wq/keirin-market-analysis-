import {normalizeRace} from './cross_engine.mjs';
import {featurePack} from './joint_model.mjs';
export const EXTENDED_VERSION='conditional-pair-extended-v1';
const softmax=xs=>{const m=Math.max(...xs),e=xs.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s);};
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export function extendedScope(input){
 const r=normalizeRace(input),classes=r.entries.map(e=>String(e.class||'').normalize('NFKC').toUpperCase());
 if(r.entries.length===9&&classes.every(c=>['S1','S2','SS'].includes(c)))return 'S9';
 if(r.entries.length===7&&classes.every(c=>c==='A3'))return 'A3';
 return null;
}
export function predictExtendedJoint(input,bundle){
  const race=normalizeRace(input),scope=extendedScope(race);
  const model=bundle?.models?.[scope];
  if(bundle?.version!==EXTENDED_VERSION||!model||model.carCount!==race.entries.length)
    throw Error('この級・車数の学習済み4点モデルがありません');
  const pack=featurePack(input,{carCount:model.carCount}),{cars,pairs,pairFeatures,thirdFeatures}=pack;
  if(model.pair.length!==pairFeatures[0].length||model.third.length!==thirdFeatures[0][0].length||
     [...model.pair,...model.third].some(x=>!Number.isFinite(x)))throw Error('同時入着モデルの特徴・係数が不正');
  const pairP=softmax(pairFeatures.map(x=>dot(x,model.pair))),probabilities={};
  pairs.forEach(([a,b],i)=>{
    const rest=cars.filter(c=>c!==a&&c!==b),thirdP=softmax(thirdFeatures[i].map(x=>dot(x,model.third)));
    rest.forEach((c,j)=>probabilities[[a,b,c].join('-')]=pairP[i]*thirdP[j]);
  });
  const additiveControl=null;
  return {raceId:race.race_id,source:{id:EXTENDED_VERSION+'_'+scope,trainThrough:bundle.trainThrough,
    scope,status:'EXPERIMENTAL',selectionObjective:'HIT_PROBABILITY',
    decisionUse:'COMPARISON_ONLY',
    caveat:'過去データで学習した同時入着モデル。順位1位を採用の根拠にしない'},probabilities,additiveControl};
}

