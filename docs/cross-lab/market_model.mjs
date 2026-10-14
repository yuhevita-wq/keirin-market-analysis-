/** Port of research/market_probability_v1/train.py. Frozen coefficients, no refit. */
import {normalizeRace} from './cross_engine.mjs';
import {allOutcomes} from './cross_comparison.mjs';
const fields=['score','win_rate','top3_rate','b','nige','makuri','sashi','mark','age'];
const scales=[5,20,25,6,6,6,6,6,12];
const featureNames=['market_logprob_div5','trifecta_vs_trio_logratio',
  ...[1,2,3].flatMap(place=>['score','win_rate','top3_rate','b_count','nige_count','makuri_count','sashi_count','mark_count','age','style_nige','style_ryo'].map(k=>k+'_place'+place)),
  'same_12','same_23','same_13','all_same','leader_1','leader_2','leader_3','line_size_1','line_size_2','line_size_3'];
const clip=(v,a,b)=>Math.max(a,Math.min(b,v));
const normalize=book=>{const z=Object.values(book).reduce((s,o)=>s+1/o,0);return Object.fromEntries(Object.entries(book).map(([k,o])=>[k,1/o/z]));};

export function predictMarket(input, odds, trioOdds, model) {
  const race=normalizeRace(input), n=race.entries.length;
  if(!race.race_id)throw Error("レースIDを含む出走表URLが必要");
  if(n<5||n>7||!/[ＳS]級/.test(race.race_type)||!/[ＦF]1|Ｆ１/.test(race.meeting_grade))
    throw Error('既存市場モデルの対象はS級F1・5～7車。A級等へ自動転用しません');
  if(JSON.stringify(model.features)!==JSON.stringify(featureNames)||model.coefficients.length!==45 ||
     model.coefficients.some(v=>!Number.isFinite(v)))throw Error('学習済みモデルの特徴順・係数が不正');
  const cars=race.entries.map(e=>e.no).sort((a,b)=>a-b), outcomes=allOutcomes(input);
  const orderedKeys=outcomes.map(t=>t.join('-'));
  const trios=[];for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)for(let k=j+1;k<n;k++)trios.push([cars[i],cars[j],cars[k]].join('='));
  function validate(book,keys,name){
    if(!book||Object.keys(book).length!==keys.length||keys.some(k=>typeof book[k]!=='number'||!Number.isFinite(book[k])||book[k]<1))
      throw Error(name+'全組み合わせのオッズが必要');
  }
  validate(odds,orderedKeys,'3連単');validate(trioOdds,trios,'3連複');
  // Runtime refuses missing card fields instead of silently creating zero strength.
  for(const e of race.entries){
    if(fields.some(k=>e[k]===null||e[k]===undefined||e[k]===''||!Number.isFinite(Number(e[k]))||Number(e[k])<0)||
       !['逃','両','追'].includes(e.style)||Number(e.win_rate)>100||Number(e.top3_rate)>100)
      throw Error(e.no+'番の得点・年齢・勝率・3連対率・脚質・決まり手を確認してください');
  }
  const means=fields.map(k=>race.entries.reduce((s,e)=>s+Number(e[k]),0)/n);
  const stats=new Map(race.entries.map(e=>[e.no,[...fields.map((k,j)=>clip((Number(e[k])-means[j])/scales[j],-3,3)),e.style==='逃'?1:0,e.style==='両'?1:0]]));
  const q=normalize(odds), qt=normalize(trioOdds), sets={};
  outcomes.forEach(t=>{const key=[...t].sort((a,b)=>a-b).join('=');sets[key]=(sets[key]||0)+q[t.join('-')];});
  const logits=outcomes.map(t=>{
    const key=t.join('-'), grouping=[...t].sort((a,b)=>a-b).join('='), lp=t.map(c=>race.position.get(c));
    const same=(a,b)=>lp[a].line===lp[b].line?1:0;
    const x=[Math.log(q[key])/5,clip(Math.log(sets[grouping]/qt[grouping]),-2,2),
      ...t.flatMap(c=>stats.get(c)),same(0,1),same(1,2),same(0,2),same(0,1)&&same(1,2)?1:0,
      ...lp.map(p=>p.pos===1?1:0),...lp.map(p=>race.lines[p.line].length/4)];
    return Math.log(q[key])+x.reduce((s,v,j)=>s+v*model.coefficients[j],0);
  });
  const max=Math.max(...logits), unnormalized=logits.map(z=>Math.exp(z-max)), total=unnormalized.reduce((s,v)=>s+v,0);
  return {raceId:race.race_id,source:{id:'market_probability_v1_s_f1_2025',trainThrough:model.train_through,
    scope:'S級F1・5～7車',status:'EXPERIMENTAL_CROSS_SELECTION',
    coefficientSource:'data/models/market_probability_v1_s_f1_2025.json',
    caveat:'既存の確定オッズ研究モデルを使用。クロス4点の選定成績と締切前運用は未検証'},
    probabilities:Object.fromEntries(outcomes.map((t,i)=>[t.join('-'),unnormalized[i]/total]))};
}
