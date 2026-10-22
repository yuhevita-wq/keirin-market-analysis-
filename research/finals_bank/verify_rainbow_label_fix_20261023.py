"""Regression and adversarial checks for the collection's male-final predicate."""
import ast,pathlib,re,unicodedata,sqlite3,json,os
P=pathlib.Path(__file__).resolve().parent;root=pathlib.Path(os.environ.get('FINALS_ROOT',P.parent/'current_data/unpacked/finals_ai')).resolve()
def load_function(s):
 tree=ast.parse(s);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['norm','final_label','category']]
 scope={'re':re,'unicodedata':unicodedata};exec(compile(ast.Module(body=nodes,type_ignores=[]),'predicate','exec'),scope);return scope
new=load_function((root/'pipeline.py').read_text())
positive=['Ａ級ファイナル','A級ファイナル','レインボーカップＡ級ファイナル','チャレンジファイナル','レインボーカップ チャレンジファイナル','Ｓ級決勝','Ａ級決勝','チャレンジ決勝','ヤンググランプリ','ＫＥＩＲＩＮグランプリ']
negative=['Ａ級準決勝','Ｓ級準決勝','ガールズ決勝','女子ファイナル','L級ファイナル','ガールズグランプリ','S級特秀','A級準決','ルーキーチャンピオン','チャレンジ予選','ファイナル予想','Ａ級一般']
for x in positive:assert new['final_label'](x),x
for x in negative:assert not new['final_label'](x),x
assert new['category']('Ａ級ファイナル')=='A12';assert new['category']('チャレンジファイナル')=='A3'
# Exact historical baseline predicate extracted from the materialized canonical34 ZIP.
import zipfile
old=load_function((P/'pipeline_before_label_fix.py').read_text())
c=sqlite3.connect(f'file:{(root/"data/finals.sqlite").resolve()}?mode=ro',uri=True)
labels=c.execute('select race_id,label from races').fetchall();assert len(labels)==5354
assert all(bool(old['final_label'](label))==bool(new['final_label'](label)) for _,label in labels)
assert all(not old['final_label'](x) and new['final_label'](x) for x in positive[:5])
# Mutant drops ファイナル: actual source label positives then reject the mutation.
mutant=lambda x:old['final_label'](x)
assert any(not mutant(x) for x in positive[:5])
report={'existing5354label_results_unchanged':True,'new_final_positive_cases':positive[:5],'negative_cases':negative,'category_A12_A3_verified':True,'all_finals_complete':False,'no_collector_restart':True,'scope':'Predicate only; calendar must be re-audited before full scope certification.'}
(P/'label_fix_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('label fix: historical5354 unchanged;5 newly recognized final labels;12 negative cases rejected')
