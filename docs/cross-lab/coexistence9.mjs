/** Observed pair coexistence, nine-car S-class G1/G2/G3.
 * This is a pair diagnostic, NOT a four-ticket winner model.
 * Trained on actual 2022-24 finishes; tested untouched 2025/2026H1.
 */
import {normalizeRace} from './cross_engine.mjs';
export const OBSERVED_PAIR9_VERSION='observed-pair-compatibility9-v1';
const fields=['score','win_rate','top3_rate','b','nige','makuri','sashi','mark'];
const scales=[10,20,25,8,6,6,6,6];
const number=(e,k)=>{
  const aliases={b:['b','b_count'],nige:['nige','nige_count'],makuri:['makuri','makuri_count'],
    sashi:['sashi','sashi_count'],mark:['mark','mark_count'],score:['score','race_score'],
    win_rate:['win_rate'],top3_rate:['top3_rate']};
  for(const name of aliases[k]||[k]){
    const v=e[name];if(v==null||v==='')continue;
    const parsed=Number(String(v).replace('%',''));
    if(Number.isFinite(parsed)&&parsed>=0)return parsed;
  }
  throw Error(e.no+'番の'+k+'が不足。実績を0で代用して同時入着を推定しません');
};
const logistic=(features,model)=>{
  if(!model||model.coef?.length!==features.length||model.mean?.length!==features.length||model.scale?.length!==features.length)
    throw Error('学習済み9車ペアモデルの特徴数が不正');
  let z=model.intercept;
  for(let i=0;i<features.length;i++)z+=model.coef[i]*(features[i]-model.mean[i])/model.scale[i];
  return 1/(1+Math.exp(-Math.max(-40,Math.min(40,z))));
};
export function pairFeaturePack9(input){
  const race=normalizeRace(input);
  if(race.entries.length!==9||!/[SsＳｓ]級/.test(race.race_type))
    throw Error('9車のS級専用の実結果学習モデルです');
  const entry=new Map(race.entries.map(e=>[e.no,e]));
  const stats=new Map(race.entries.map(e=>[e.no,Object.fromEntries(fields.map(k=>[k,number(e,k)]))]));
  const means=Object.fromEntries(fields.map(k=>[k,race.entries.reduce((s,e)=>s+stats.get(e.no)[k],0)/9]));
  const line=no=>race.position.get(no).line;
  const pos=no=>race.position.get(no).pos;
  const lineSize=no=>race.lines[line(no)].length;
  const rider=no=>fields.map((k,i)=>(stats.get(no)[k]-means[k])/scales[i])
    .concat([1,2,3].map(p=>+(pos(no)===p)),[+(lineSize(no)===3)]);
  const leader=no=>race.lines[line(no)][0];
  const b=no=>stats.get(no).b,nige=no=>stats.get(no).nige;
  const top3Line=li=>race.lines[li].reduce((s,no)=>s+stats.get(no).top3_rate,0);
  const pairs=[];
  for(let a=1;a<=9;a++)for(let c=a+1;c<=9;c++){
    const [u,v]=[a,c].sort((x,y)=>stats.get(y).top3_rate-stats.get(x).top3_rate||
      stats.get(y).score-stats.get(x).score||x-y);
    const pu=pos(u),pv=pos(v),lu=line(u),lv=line(v),same=lu===lv;
    const fu=leader(u),fv=leader(v),bu=b(fu),bv=b(fv),nu=nige(fu),nv=nige(fv);
    const other=race.lines.filter((l,i)=>i!==lu&&i!==lv).map(l=>b(l[0]));
    const base=[...rider(u),...rider(v)];
    const mechanism=[
      +same,+(pu===1&&pv===1),+((pu===1&&pv===2)||(pu===2&&pv===1)),
      +(pu===2&&pv===2),+((pu===1&&pv===3)||(pu===3&&pv===1)),
      +((pu===2&&pv===3)||(pu===3&&pv===2)),+(pu===3&&pv===3),
      +(same&&Math.abs(pu-pv)===1),+(!same&&pu===1&&pv===1),
      +(!same&&pu===2&&pv===2),+(!same&&((pu===1&&pv===2)||(pu===2&&pv===1))),
      +(bu>=5&&bv>=5),same?0:bu*bv/100,same?0:Math.abs(bu-bv)/10,
      same?0:nu*nv/100,Math.max(0,...other)/10,other.reduce((s,x)=>s+x,0)/20,
      +(!same&&bu>=5&&bv>=5),
      +(stats.get(u).mark>=2&&stats.get(v).mark>=2),
      +(((pu===1&&pv===2)||(pu===2&&pv===1))&&Math.max(bu,bv)>=5),
      +(!same&&((pu===1&&pv===2)||(pu===2&&pv===1))&&bu>=5&&bv>=5),
      +(lineSize(u)===3&&lineSize(v)===3),+(lineSize(u)===3||lineSize(v)===3),
      +(lineSize(u)===1||lineSize(v)===1),
      same?0:Math.abs(top3Line(lu)-top3Line(lv))/100,
      same?0:Math.abs(stats.get(fu).score-stats.get(fv).score)/10
    ];
    if(base.length!==24||mechanism.length!==26)throw Error('9車ペア特徴が不正');
    pairs.push({pair:[a,c],sameLine:same,base,mechanism});
  }
  return {race,pairs};
}
export function predictObservedPair9(input,bundle){
  if(bundle?.version!==OBSERVED_PAIR9_VERSION||bundle?.trainThrough!=='2024-12-31'
    ||bundle?.evaluationStatus!=='HOLDOUT_PASSED'||bundle?.canAutoRankCross4!==false)
    throw Error('未検証の9車共存モデルを採用判定に使用しません');
  const {race,pairs}=pairFeaturePack9(input);
  const date=String(race.race_id).match(/^\\d{2}(\\d{4})(\\d{2})(\\d{2})/);
  if(!date||date.slice(1).join('-')<=bundle.trainThrough)throw Error('出走レースID・学習期間の検証に失敗');
  const scores=pairs.map(z=>{
    const full=[...z.base,...z.mechanism];
    return {pair:z.pair,sameLine:z.sameLine,
      top2Raw:logistic(full,bundle.models.top2.mechanism),
      top2BaseRaw:logistic(z.base,bundle.models.top2.rider_only),
      top3Raw:logistic(full,bundle.models.top3.mechanism),
      top3BaseRaw:logistic(z.base,bundle.models.top3.rider_only)};
  });
  const total2=scores.reduce((s,z)=>s+z.top2Raw,0),total2Base=scores.reduce((s,z)=>s+z.top2BaseRaw,0);
  const rows=scores.map(z=>({
    pair:z.pair,sameLine:z.sameLine,
    top2:z.top2Raw/total2,top2RiderOnly:z.top2BaseRaw/total2Base,
    top3:z.top3Raw,top3RiderOnly:z.top3BaseRaw,
    interactionDeltaTop2:z.top2Raw/total2-z.top2BaseRaw/total2Base,
    interactionDeltaTop3:z.top3Raw-z.top3BaseRaw
  }));
  return {version:bundle.version,trainThrough:bundle.trainThrough,status:'PAIR_ONLY_NOT_4TICKET',
    input:race.race_id,rows,
    top2Ranking:rows.filter(r=>!r.sameLine).sort((a,b)=>b.top2-a.top2),
    top3Ranking:rows.filter(r=>!r.sameLine).sort((a,b)=>b.top3-a.top3),
    message:'過去の実結果で学習した同時1・2着／同時3着以内。個人能力のみとの予測差を併記。3着指定の4点確率・期待値は未算出。'};
}
