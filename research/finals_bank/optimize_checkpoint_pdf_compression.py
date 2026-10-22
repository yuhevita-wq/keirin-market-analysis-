"""Losslessly choose the smaller PDF compression, preserving other ZIP member bytes.

Copy complete local ZIP records for unchanged members. Recompressed members and
all copied members are read back and SHA-checked before atomic replacement.
"""
import argparse,copy,hashlib,pathlib,zipfile

def optimize(path):
    temporary=path.with_suffix('.optimized.zip');expected={};savings=0;changed=[]
    with zipfile.ZipFile(path) as src,zipfile.ZipFile(temporary,'w') as dst:
        infos=sorted(src.infolist(),key=lambda i:i.header_offset)
        for index,info in enumerate(infos):
            raw=src.read(info.filename);expected[info.filename]=hashlib.sha256(raw).digest()
            use_lzma=False
            if info.filename.lower().endswith('.pdf') and info.compress_type==zipfile.ZIP_DEFLATED:
                compressor=zipfile.LZMACompressor();candidate=compressor.compress(raw)+compressor.flush()
                use_lzma=len(candidate)<info.compress_size
            if use_lzma:
                dst.writestr(info.filename,raw,compress_type=zipfile.ZIP_LZMA)
                savings+=info.compress_size-dst.getinfo(info.filename).compress_size;changed.append(info.filename)
            else:
                end=infos[index+1].header_offset if index+1<len(infos) else src.start_dir
                src.fp.seek(info.header_offset);record=src.fp.read(end-info.header_offset);assert len(record)==end-info.header_offset
                replacement=copy.copy(info);replacement.header_offset=dst.fp.tell();dst.fp.write(record)
                dst.filelist.append(replacement);dst.NameToInfo[replacement.filename]=replacement;dst.start_dir=dst.fp.tell();dst._didModify=True
    with zipfile.ZipFile(temporary) as check:
        assert len(check.namelist())==len(expected) and set(check.namelist())==set(expected)
        for name,digest in expected.items():assert hashlib.sha256(check.read(name)).digest()==digest,name
    temporary.replace(path);return dict(bytes=path.stat().st_size,saved_compressed_bytes=savings,recompressed=changed,all_member_hashes_equal=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('zip',type=pathlib.Path);print(optimize(p.parse_args().zip))
