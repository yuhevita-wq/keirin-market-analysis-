# G3専用｜ベンター市場統合・期待値検証基盤 v1

**目的**：このフォルダーと凍結済みG3独自モデルだけを読み、**新しい会話から同じ条件で市場誤差研究を実行する**。個別開催の分析や購入判断はこのリポジトリ設定の段階では行わない。

## 現在の完了・未完了

| 項目 | 状態 |
|---|---|
| G3専用1着・2着・3着の数式 | **凍結済み**（学習済み係数は変更しない） |
| 全順序付き3連単確率 | **計算可能**（9車504通り） |
| 全7券種の買い目への確率集約 | **実装済み**（ただし各券種のオッズ取得可否と統合較正は別） |
| 排他的券種の市場確率近似 | **実装済み**（完全なオッズブックに対し逆オッズを正規化） |
| ワイド | **3組が同時に的中する非排他的事象として別処理**。逆オッズの総和を1へ正規化しない |
| G3独自確率＋市場確率の重み α,β | **まだ実結果で推定していない**。任意に1/1などを「採用値」としない |
| 保存済み過去オッズ | **3連単・3連複、確定時点のみ**。G3で条件一致する件数は未監査 |
| 締切15分前の全7券種オッズ | **収集済みとは確認していない** |
| EV＋の実在する買い目 | **まだ認定できない** |
| ケリー配分・投票によるオッズ変化 | **未実装**（板厚・プールが不明） |

参照元とGit blob SHAのロックは [`source_lock.json`](source_lock.json)。

## ベンター1994原論文の工程（原典に基づく）

1. 出走条件・選手の過去情報などから得た**独立予測確率**と、投票市場の**公衆確率**を別々に記録。
2. **過去の実結果**と比較して、公衆予測が独自予測から外れたときの誤差を調べる。
3. p.187–188の式（独立予測値と公衆確率の対数を共変量とした第2段階ロジット）：

\[
c_i = \frac{\exp(\alpha\log f_i + \beta\log p_i)}{\sum_j\exp(\alpha\log f_j+\beta\log p_j)}
\]

ここで `f` は学習で再利用されていない独立モデル予測、`p` は市場確率、`c` は統合確率。係数 α,β は**過去結果の最尤推定**で求める。α+β=1とは限らない。

4. 統合後の確率の較正と市場単独に対する追加予測力を、**別期間**で確認する。
5. pp.190–192の考え方で `期待回収倍率 = 統合確率 × 実際に得られる見込払戻倍率` を求め、プラスの価格だけを候補にする。複数買い目の依存、ケリー、投票による払戻倍率低下は**別の最適化**。

### 原論文からの拡張（ベンターが競輪で使った手法ではない）

競輪には通常の単勝市場がない。3連単の完全オッズブックから逆オッズを正規化し、その1着周辺確率を**公衆の1着確率の代理**とする処理は競輪への独自移植。更に3連単・3連複などの**相互排他的な車券の全候補**に同じ対数線形結合を適用するのも、原論文の思想に基づく**研究用の拡張**。各プールのオッズが独立に決まるため、確率の整合と取引コスト・自分の投票の影響を別途確認する。

## 実装済みコード

- [`engine.py`](engine.py)：G3の1・2・3着順序確率を全券種事象へ集約し、完全なオッズブックを検査。係数未学習時は**EV欄をnull**。
- [`stage_matched.py`](stage_matched.py)：**同一レースIDの事前G3独立確率・全券種オッズ・確定1〜3着を検査して券種別の学習JSONLを生成**。
- [`train_fusion.py`](train_fusion.py)：G3のレースID・確定オッズ・実着順・事前独立確率を照合済みのJSONLから、券種別または**1着市場代理**の α,β を最尤推定、日付で分離して評価。
- [`run.py`](run.py)：**新しい会話から呼ぶ入口**。締切15分前の時刻照合・市場欠損の強制検出・保存用JSON出力。
- [`test_market_bridge.py`](test_market_bridge.py)：人工データによる排他性・確率正規化・オッズ欠損・時間窓・学習・漏洩防止のテスト。
- [`fixtures/synthetic_g3_card.json`](fixtures/synthetic_g3_card.json)・[`fixtures/synthetic_final_odds.json`](fixtures/synthetic_final_odds.json)：**架空**の5車・完全ブック入力。実際のオッズ・競走情報ではない。

## 新しい会話からの操作

GitHubのブランチは `research/g3-independent-first-20261018`。まず [`NEW_CHAT_HANDOFF.md`](NEW_CHAT_HANDOFF.md) を読む。

### 1. 診断・未較正状態（人工データの動作確認）

```bash
python research/g3_independent_v1/market_benter/run.py \
  --card research/g3_independent_v1/market_benter/fixtures/synthetic_g3_card.json \
  --odds research/g3_independent_v1/market_benter/fixtures/synthetic_final_odds.json \
  --historical-final \
  --output /tmp/g3_market_demo.json
```

これは**市場確率の近似と価格の照合だけ**。モデル較正なしなので `expected_return_multiple=null`。人工オッズから利益は主張できない。

### 2. 実際の歴史的なオッズとの統合係数研究

必要入力：G3の同一レースIDで、(a)各候補への**当時以前の情報だけから算出した**独立勝率由来の券種別確率、(b)同時点の全選択肢の確定オッズ、(c)実際の当たり買い目、(d)レース日付、(e)出所・抽出時刻。

結合済みの元入力は1行1レースで、`scope, race_id, race_date, fundamental_provenance, price_phase, card, finish, markets` を持つ。`card` は独立G3モデルから得た全車の事前1着確率と予想ラインのみ。`finish` は確定1〜3着、`markets` は完全な市場オッズブックであり、G3係数モデルは`stage_matched.py`がGitHubの凍結ファイルから読み込む。

```bash
python research/g3_independent_v1/market_benter/stage_matched.py \\
  --input /path/to/g3_joined_history.jsonl \\
  --output-dir /path/to/g3_staged_market_books
```

このステージングはレースIDや欠損オッズを自動で補完せず、排他的市場ブックに穴があれば失敗する。出力は `3連単.jsonl`、`3連複.jsonl`、`1着市場代理_3連単.jsonl` 等の**市場別ファイル**となる。原始オッズ・結果・独自予測のレースID結合は、この前の工程で監査すること。

その市場別の**学習入力JSONL**は1行1レース1券種、例（**書式だけの概念例**。実レースで全候補・全オッズに置換必須）：

```json
{
  "scope": "G3 S級", "market": "3連単",
  "race_id": "SYNTHETIC-ID", "race_date": "2024-01-01",
  "fundamental_provenance": "G3_only_prior_period_independent",
  "price_phase": "final",
  "fundamental_probabilities": {"1-2-3": 0.002},
  "odds": {"1-2-3": 400.0},
  "actual_winning_selection": "1-2-3",
  "original_asof_card_certified": false
}
```

**注意**：上の例は1選択肢だけで、合計確率1にも全買い目にもなっていないため、そのまま入力すると**正しく拒否される**。一部の買い目だけを抜いたモデルにはしない。

```bash
python research/g3_independent_v1/market_benter/train_fusion.py \
  --input /path/to/g3_staged_market_books/3連単.jsonl \
  --market 3連単 \
  --training-before 2026-01-01 \
  --evaluation-from 2026-01-01 \
  --output /path/to/g3_trifecta_calibration.json
```

3連単から市場の**1着周辺確率**を再構成するモデルも `--market 1着市場代理_3連単` で別個に学習できる。この場合は同一レース全3連単オッズと5～9車の車番、独立モデルの1着確率、実1着が必要。**3連単そのものの車券期待値の係数と同一視しない。**

### 3. 締切15分前の実戦段階

オッズ入力は最低限次の形：

```json
{
  "snapshot": {
    "price_phase": "predeadline",
    "captured_at": "YYYY-MM-DDTHH:MM:SS+09:00",
    "bet_close_at": "YYYY-MM-DDTHH:MM:SS+09:00",
    "target_minutes_before_close": 15,
    "tolerance_minutes": 2
  },
  "markets": { "3連単": {"1-2-3": 400.0} }
}
```

上も書式例のみ。実際は**対象券種の全売出し選択肢**が必要。異なる券種の一部買い目、取得時刻不明、締切後や確定オッズとの混在は拒否する。

**重要**：現在存在する市場統合係数はゼロ。**確定オッズで学習した係数をそのまま15分前に流用することもコードが拒否**する。過去15分前のオッズで較正された統合モデルがない場合、ライブのEV＋は出さず、利用可能な独立確率・市場近似・足りない条件を報告する。

## 次の会話で未完了の作業

1. 保存済みの3連単・3連複の**G3レースIDとの実照合**を監査し、対象母数を確定。
2. G3各年の**事前独立確率をレースIDで再現**してオッズ・着順と結合する。この結合が実証されるまで適当なレース数を主張しない。
3. 公衆確率と独立確率の価格ズレを**先に観測**し、α,βを当該G3の結果で学習・期間別検証。
4. 3連単・3連複以外の5券種が欠損なら、その旨を明記。全券種を見た、全買い目のEVを出した、とは主張しない。
5. 払戻見込み、投票による価格下落、各買い目の同時購入相関が測れない段階では**購入金額を捏造しない**。

**ゴールは市場が繰り返し間違える価格だけを、再現可能な数値で抽出すること。** 全外・人気順・固定買い目の主観置換は禁止。
