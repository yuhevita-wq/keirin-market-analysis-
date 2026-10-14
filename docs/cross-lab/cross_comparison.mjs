/** Exact ticket aggregation and explicit decision records. No fitted coefficients. */
import {normalizeRace} from './cross_engine.mjs';

export function allOutcomes(input) {
  const race=normalizeRace(input), cars=race.entries.map(e=>e.no), out=[];
  for(const a of cars)for(const b of cars)for(const c of cars)
    if(a!==b && a!==c && b!==c)out.push([a,b,c]);
  return out;
}

export function classifyOutcome(input, ticket) {
  const r=normalizeRace(input), t=ticket.map(Number);
  if(t.length!==3 || new Set(t).size!==3 || t.some(n=>!r.position.has(n)))
    throw Error('不正な確定着順');
  const p=t.map(n=>r.position.get(n));
  if(p[0].line===p[1].line)return {kind:'SAME_LINE_12',coreCross:false};
  if(new Set(p.map(x=>x.line)).size===3)return {kind:'THREE_LINES',coreCross:false};
  if(p.some(x=>x.pos>2 || r.lines[x.line].length<2))return {kind:'OUTSIDE_CORE_ROLE',coreCross:false};
  return {kind:'CORE_CROSS',coreCross:true};
}

// Four equal stakes: E(return / investment)=sum(p_i * odds_i)/4.
// min paying odds / 4 is a realized break-even property, not a forecast.
export function ticketMarket(candidate, odds={}) {
  const keys=candidate.tickets.map(t=>t.join('-'));
  const prices=keys.map(k=>odds[k]);
  const valid=prices.every(x=>typeof x==='number' && Number.isFinite(x) && x>=1);
  return {complete:valid,priced:prices.filter(x=>typeof x==='number' && Number.isFinite(x) && x>=1).length,
    tickets:keys.map((key,i)=>({key,odds:prices[i]??null})),
    minOdds:valid?Math.min(...prices):null,maxOdds:valid?Math.max(...prices):null,
    minRealizedReturnRatio:valid?Math.min(...prices)/keys.length:null,
    inverseOddsSum:valid?prices.reduce((s,o)=>s+1/o,0):null};
}

export function compareJointForecast(input, candidates, forecast, odds={}) {
  if(!forecast)return null;
  const race=normalizeRace(input), expected=allOutcomes(input).map(t=>t.join('-'));
  const p=forecast.probabilities, source=forecast.source;
  if(!p || !source?.id || !source?.trainThrough || !forecast.raceId ||
      !race.race_id || String(forecast.raceId)!==String(race.race_id))
    throw Error('予測のレースID・モデルID・学習終了日が必要');
  const match=String(race.race_id).match(/^\d{2}(\d{4})(\d{2})(\d{2})/);
  const date=match?match.slice(1).join('-'):null;
  if(!/^\d{4}-\d{2}-\d{2}$/.test(source.trainThrough) || !date || source.trainThrough>=date)
    throw Error('当日以降を学習した予測は採用できません');
  if(Object.keys(p).length!==expected.length || expected.some(k=>typeof p[k]!=='number' ||
      !Number.isFinite(p[k]) || p[k]<0 || p[k]>1))throw Error('全着順の同時確率が必要');
  const total=expected.reduce((s,k)=>s+p[k],0);
  if(Math.abs(total-1)>1e-8)throw Error('全着順の確率合計は1が必要');
  const mass={SAME_LINE_12:0,THREE_LINES:0,OUTSIDE_CORE_ROLE:0,CORE_CROSS:0};
  for(const key of expected)mass[classifyOutcome(input,key.split('-')).kind]+=p[key];
  const rows=candidates.map(c=>{
    const market=ticketMarket(c,odds), ks=c.tickets.map(t=>t.join('-'));
    return {signature:c.signature,hitProbability:ks.reduce((s,k)=>s+p[k],0),
      returnRatio:market.complete?ks.reduce((s,k)=>s+p[k]*odds[k],0)/ks.length:null};
  });
  const ranked=rows.filter(r=>r.returnRatio!==null).sort((a,b)=>b.returnRatio-a.returnRatio);
  // Partial candidate prices can reverse the choice: never optimize only priced subset.
  const complete=ranked.length===candidates.length && candidates.length>0;
  const tie=complete && ranked.length>1 && Math.abs(ranked[0].returnRatio-ranked[1].returnRatio)<=1e-12;
  return {source,mass,rows,complete,
    selectedSignature:complete&&!tie?ranked[0].signature:null,
    reason:!complete?'全候補の基本4点オッズが不足':tie?'モデル基準で同値':'入力モデル基準の4点等額期待回収を比較',
    note:'入力モデルの較正・収益性をこの計算器は証明しません。市場重みを予測確率で代用しません'};
}

export function resolveDecision(result, decision=null, comparison=null) {
  if(comparison?.selectedSignature) {
    const main=result.candidates.find(c=>c.signature===comparison.selectedSignature);
    if(!main)throw Error('モデル選定候補が比較集合にない');
    return {main,label:'主採用（入力モデル基準）',provisional:true,
      reason:comparison.reason+'。'+comparison.note};
  }
  if(!decision)return {main:null,label:'主採用未決定',provisional:false,
    reason:'全型を列挙しました。車番順・根拠件数を採用理由にせず、比較記録から1型を選定してください'};
  const main=result.candidates.find(c=>c.signature===decision.signature);
  if(!main)throw Error('採用クロスが候補集合に存在しません');
  for(const k of ['interaction','alternativeCross','sameLine','outside'])
    if(typeof decision[k]!=='string'||!decision[k].trim())
      throw Error('採用理由・別クロス比較・筋比較・選外車の反証をすべて記入してください');
  return {main,label:'主採用（分析者の比較判断）',provisional:true,
    reason:[decision.interaction,decision.alternativeCross,decision.sameLine,decision.outside].join('。'),
    record:{...decision}};
}

export function gradeFrozenDecision(input, decision, actual, payoutPer100, result) {
  const selected=result.candidates.find(c=>c.signature===decision.signature);
  if(!selected)throw Error('固定答案が候補集合に存在しません');
  const key=actual.join('-'), classification=classifyOutcome(input,actual);
  const candidateContains=result.candidates.some(c=>c.tickets.some(t=>t.join('-')===key));
  const hit=selected.tickets.some(t=>t.join('-')===key);
  return {classification,hit,candidateContains,
    failure:hit?'HIT':classification.coreCross?(candidateContains?'SELECTION_MISS':'GENERATION_MISS'):classification.kind,
    stake:400,payout:hit?payoutPer100:0};
}
