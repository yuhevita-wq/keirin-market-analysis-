import zipfile,lzma,struct,hashlib,pathlib,time
class StrongLZMA(zipfile.LZMACompressor):
 def _init(self):
  filt={'id':lzma.FILTER_LZMA1,'preset':9|lzma.PRESET_EXTREME}
  props=lzma._encode_filter_properties(filt);self._comp=lzma.LZMACompressor(lzma.FORMAT_RAW,filters=[filt]);return struct.pack('<BBH',9,4,len(props))+props
name='finals_ai/data/finals.sqlite';t=time.time()
with zipfile.ZipFile('restored_1417/finals_checkpoint_updated.zip') as z:raw=z.read(name);before=z.getinfo(name).compress_size
assert len(raw)==281206784 and hashlib.sha256(raw).hexdigest()=='04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91'
original=zipfile.LZMACompressor;zipfile.LZMACompressor=StrongLZMA
try:
 with zipfile.ZipFile('compression_candidate_1417.zip','w',compression=zipfile.ZIP_LZMA) as z:
  with z.open(name,'w') as f:
   for pos in range(0,len(raw),1<<20):
    f.write(raw[pos:pos+(1<<20)])
    if pos%(32<<20)==0:print('compressed input MB',pos//(1<<20),'elapsed',round(time.time()-t),flush=True)
finally:zipfile.LZMACompressor=original
with zipfile.ZipFile('compression_candidate_1417.zip') as z:
 assert z.read(name)==raw;after=z.getinfo(name).compress_size
print('candidate full decoded SHA equal',before,after,'saving',before-after,'elapsed',round(time.time()-t),flush=True)
