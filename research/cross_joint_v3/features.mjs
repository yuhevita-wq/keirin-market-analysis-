import fs from 'node:fs';
import {featurePack} from '../../docs/cross-lab/joint_model.mjs';
const [input,prefix]=process.argv.slice(2),rows=JSON.parse(fs.readFileSync(input));
const packs=rows.map(r=>featurePack(r.input)),pairs=[],thirds=[],yPair=[],yThird=[];
for(let i=0;i<rows.length;i++){
  const p=packs[i],actual=rows[i].target;
  const pi=p.pairs.findIndex(([a,b])=>a===actual[0]&&b===actual[1]);
  pairs.push(...p.pairFeatures.flat());
  thirds.push(...p.thirdFeatures[pi].flat());
  yPair.push(pi);yThird.push(p.cars.filter(c=>!actual.slice(0,2).includes(c)).indexOf(actual[2]));
}
const write=(name,arr,Type)=>fs.writeFileSync(prefix+name,Buffer.from(new Type(arr).buffer));
write('.pair.f32',pairs,Float32Array);write('.third.f32',thirds,Float32Array);
write('.yp.i32',yPair,Int32Array);write('.yt.i32',yThird,Int32Array);
fs.writeFileSync(prefix+'.shape.json',JSON.stringify({races:rows.length,pairFeatures:packs[0].pairFeatures[0].length,thirdFeatures:packs[0].thirdFeatures[0][0].length,baseFeatures:packs[0].basePairFeatureCount}));
