import fs from 'node:fs';
import {featurePack} from '../../docs/cross-lab/joint_model.mjs';
const [input,prefix]=process.argv.slice(2),rows=JSON.parse(fs.readFileSync(input));
const pairs=[],thirds=[],yPair=[],yThird=[];let pack;
for(const row of rows){
 pack=featurePack(row.input,{carCount:row.input.entries.length});
 const pi=pack.pairs.findIndex(([a,b])=>a===row.target[0]&&b===row.target[1]);
 pairs.push(...pack.pairFeatures.flat());thirds.push(...pack.thirdFeatures[pi].flat());
 yPair.push(pi);yThird.push(pack.cars.filter(c=>!row.target.slice(0,2).includes(c)).indexOf(row.target[2]));
}
for(const [suffix,arr,T] of [['pair',pairs,Float32Array],['third',thirds,Float32Array],['yp',yPair,Int32Array],['yt',yThird,Int32Array]])fs.writeFileSync(prefix+'.'+suffix,Buffer.from(new T(arr).buffer));
fs.writeFileSync(prefix+'.json',JSON.stringify({n:rows.length,pairs:pack.pairs.length,rest:pack.cars.length-2,pf:pack.pairFeatures[0].length,tf:pack.thirdFeatures[0][0].length}));
