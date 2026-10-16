/** Exact ticket aggregation and explicit decision records. No fitted coefficients. */
import {normalizeRace} from './cross_engine.mjs';
import {jointDiagnostics,pairCompetition} from './joint_model.mjs';

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
  if(p[0].line===p[1].line)return {kind:'SAME_LINE_12',coreCross:false,generalCross:false};
  if(new Set(p.map(x=>x.line)).size===3)return {kind:'THREE_LINES',coreCross:false,generalCross:false};
  if(p.some(x=>x.pos>2 || r.lines[x.line].length<2))return {kind:'OUTSIDE_CORE_ROLE',coreCross:false,generalCross:p.every(x=>r.lines[x.line].length>=2)};
  return {kind:'CORE_CROSS',coreCross:true,generalCross:true};
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

export function compareJointForecast(input, candidates, forecast, odds={}, {objective='EXPECTED_RETURN'}={}) {
  if(!forecast)return null;
  if(!['EXPECTED_RETURN','HIT_PROBABILITY'].includes(objective))throw Error('比較目的が不正');
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
      directions:jointDiagnostics(c,forecast),competition:pairCompetition(c,forecast,input),
      returnRatio:market.complete?ks.reduce((s,k)=>s+p[k]*odds[k],0)/ks.length:null};
  });
  // Incomplete market prices must never rank only the priced subset.
  // Fall back to the full-distribution hit probability; do not call it market value.
  const priceComplete=rows.every(r=>r.returnRatio!==null);
  const actualObjective=objective==='EXPECTED_RETURN' && !priceComplete?'HIT_PROBABILITY':objective;
  const metric=actualObjective==='EXPECTED_RETURN'?'returnRatio':'hitProbability';
  const ranked=rows.filter(r=>r[metric]!==null).sort((a,b)=>b[metric]-a[metric]);
  const complete=ranked.length===candidates.length && candidates.length>0;
  const tie=complete && ranked.length>1 && Math.abs(ranked[0][metric]-ranked[1][metric])<=1e-12;
  return {source,mass,rows,complete,objective:actualObjective,requestedObjective:objective,
    priceComplete,ranking:ranked.map(r=>({signature:r.signature,hitProbability:r.hitProbability,
      returnRatio:r.returnRatio,metricValue:r[metric]})),
    rankedSignature:complete&&!tie?ranked[0].signature:null,
    selectedSignature:null,
    selectionStatus:complete&&!tie?'PROVISIONAL_RANKING':'NO_UNIQUE_RANKING',
    reason:!complete?'全候補の比較データ不足':tie?'モデル基準で同値（主採用の優劣を識別不能）':
      objective==='EXPECTED_RETURN'&&!priceComplete?'全候補オッズ不足のため4点合計的中確率で暫定選定。高配当・期待値は未評価':
      actualObjective==='EXPECTED_RETURN'?'同時入着確率×全候補オッズで4点等額期待回収を比較':
      '同時入着確率から基本4点の合計的中確率を比較（利益順位ではない）',
    note:'入力モデルの較正・収益性をこの計算器は証明しません。どちらかが勝つ確率ではなく、二人が1・2着を分ける確率を集計しています'};
}

export function resolveDecision(result, decision=null, comparison=null) {
  if(!decision){
    const main=comparison?.rankedSignature?
      result.candidates.find(c=>c.signature===comparison.rankedSignature):null;
    if(main){
      const row=comparison.rows.find(r=>r.signature===main.signature);
      const metric=comparison.objective==='EXPECTED_RETURN'?'期待回収':'4点的中確率';
      return {main,label:'主採用（モデル暫定1位）',provisional:true,
        reason:metric+'で全候補を比較し1位。'+comparison.reason+
          '。表裏の同時残存を学習したモデルの順位であり、実際の展開・高配当・利益優位を保証しません',
        record:{signature:main.signature,automatic:true,objective:comparison.objective,
          source:comparison.source,hitProbability:row.hitProbability,returnRatio:row.returnRatio},
        basis:{status:'MODEL_PROVISIONAL',observedFacts:null,sourceContext:comparison.source.id,
          counterEvidence:'同線決着・別クロス・選外ラインの侵入',uncertainty:'個別展開未検証、モデル未較正'}};}
    return {main:null,label:'主採用判定不能',provisional:false,
      reason:comparison?'モデル値が同順位か比較データが不足。車番・単独勝率で無理に決めません':
        'モデル未評価。全選手の出走前入力・レースID・対象級を確認してください'};}

  const main=result.candidates.find(c=>c.signature===decision.signature);
  if(!main)throw Error('採用クロスが候補集合に存在しません');
  for(const k of ['interaction','alternativeCross','sameLine','outside'])
    if(typeof decision[k]!=='string'||!decision[k].trim())
      throw Error('採用理由・別クロス比較・筋比較・選外車の反証をすべて記入してください');
  const directions=decision.directions;
  if(!Array.isArray(directions)||directions.length!==2||
      !main.firstPair.every((first,i)=>directions.some(d=>d.first===first&&
        d.second===main.firstPair[1-i]&&['coexistence','secondCompetition'].every(k=>
          typeof d[k]==='string'&&d[k].trim()))))
    throw Error('表裏それぞれに、勝つ走りで相手が2着に残る理由と、同線番手を含む2着競合の比較が必要です');
  return {main,label:'主採用（分析者の比較判断）',provisional:true,
    reason:[decision.interaction,decision.alternativeCross,decision.sameLine,decision.outside].join('。'),
    record:{...decision},
    basis:{
      observedFacts:decision.observedFacts?.trim()||null,
      sourceContext:decision.sourceContext?.trim()||null,
      inference:directions.map(d=>({first:d.first,second:d.second,
        coexistence:d.coexistence,secondCompetition:d.secondCompetition})),
      counterEvidence:decision.counterEvidence?.trim()||null,
      uncertainty:decision.uncertainty?.trim()||null,
      status:'ANALYST_JUDGMENT',
      unrecorded:['observedFacts','sourceContext','counterEvidence','uncertainty']
        .filter(k=>!decision[k]?.trim()),
      note:'記録の存在は正しさの証明ではない。未記録は不成立の証拠ではない。数値順位による補完はしない'
    }};
}

export function gradeFrozenDecision(input, decision, actual, payoutPer100, result) {
  const selected=result.candidates.find(c=>c.signature===decision.signature);
  if(!selected)throw Error('固定答案が候補集合に存在しません');
  const key=actual.join('-'), classification=classifyOutcome(input,actual);
  const candidateContains=result.candidates.some(c=>c.tickets.some(t=>t.join('-')===key));
  const hit=selected.tickets.some(t=>t.join('-')===key);
  return {classification,hit,candidateContains,
    failure:hit?'HIT':classification.generalCross?(candidateContains?'SELECTION_MISS':'GENERATION_MISS'):classification.kind,
    stake:400,payout:hit?payoutPer100:0};
}

