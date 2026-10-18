# G3専用 × ビル・ベンター（1994）市場統合：新しい会話で再現するための基盤

**現在地：数式・7賭式の確率変換・時刻監査・統合係数推定コード・テストを保存済み。ただし実G3の完全な過去2車単オッズとの結合、α・β推定、T−15分統合モデルの検証は未完了。未完了を完成と呼ばない。**

## 0. 研究範囲の固定

このディレクトリは **G3開催のS級だけ**。G1/G2、F1の係数、A級、ガールズを混ぜない。既存G3の1着・2着・3着モデルは**凍結・改変禁止**。個別開催の予想はこの基盤作成の対象外。

- [研究モデル3本と既存オッズの固定台帳](MANIFEST.json)
- [G3の凍結1着モデル](../output/g3_first_win_report.json)
- [G3の凍結2着条件付きモデル](../models/g3_second_conditional_v1.json)
- [G3の凍結3着条件付きモデル](../models/g3_third_conditional_v1.json)

## 1. ベンター原論文のオッズ利用法（元論文に実際に書かれていること）

出典：William Benter, *Computer Based Horse Race Handicapping and Wagering Systems: A Report*（1994年）。原論文印刷頁 p.187–188『Creating Unbiased Probability Estimates』、p.190–191『Wagering Strategy』、p.192–193『Exotic Bets』。

1. 選手・競技情報に相当する**独立（fundamental）モデル**を構築し、観測レースに対する確率 `f_i` を得る。
2. 公衆が価格として表した確率 `p_i` を別に求める。ベンターの原研究は**競馬の単勝市場**を使う。
3. 独立モデルの予測を、当該レースの学習済み結果にそのまま再使用せず、**別標本で求めた確率**と過去の公衆確率から、第2段階多項ロジットの係数 α・βを実際の着順で最尤推定：

   `c_i = exp(α log(f_i) + β log(p_i)) / Σ_h exp(α log(f_h) + β log(p_h))`

4. 真の的中確率の推定 `c` と実際に提示された払戻倍率 `d` で `ER=c×d` を計算し、`ER>1` の価格誤差だけを検討。
5. 賭け金は投票による配当の下落と他の同時賭けを考慮し、フルケリーを無条件に使わない。プールの流動性が分からなければ**配分額を捏造しない**。
6. 2・3着など特定の着順を指定する組み合わせは、順序付き確率を使って評価する。市場の中にある複数車の相乗的な優位が単一選手の非優位を補う場合がある。

## 2. 競輪へ移植するために追加した処理（ベンター本人の手法とは区別）

競輪には通常の**単勝・複勝がない**。売られているのは7種類：`2枠複 / 2枠単 / 2車複 / 2車単 / 3連複 / 3連単 / ワイド`。よって公衆の1着確率 `p_i` を原論文と同じ方法では直接取得できない。

本基盤では**完全に取得された2車単オッズ**の逆数を同一プール内で正規化して、順序付き市場確率の代理値とする：

`u_ij = (1 / O_ij) / Σ_{a≠b}(1 / O_ab)`、`p_i^(proxy)=Σ_{j≠i}u_ij`。

これは**研究上の仮説・価格復元方式**であり、ベンター原論文の記述ではない。欠損する2車単を推測で補填しない。少なくとも過去の実1着頻度との一致、後期間対数損失、独立モデルとの相補性を確認し、α・βをG3の過去結果だけから推定する。

## 3. なぜ7賭式をすべて計算できるか

G3モデルの3着までの順序確率は

`P(i,j,k)=q_i × P_G3(j|i;q) × P_G3(k|i,j;q)`。

7車立て210通り、9車立て504通りの全順序確率を列挙して各賭式へ集約する。

- `2車単(i,j)`：一・二着の順序を指定。3着kをすべて合計。
- `2車複{i,j}`：先着2車の順不同。
- `3連単(i,j,k)`：3着までの指定順序。
- `3連複{i,j,k}`：上位3車の順不同。
- `ワイド{i,j}`：選んだ2車が共に3着以内。1回の結果で3本のワイドが当たるので**賭式内の確率総和は3**で、1ではない。
- `2枠単(a,b)`・`2枠複{a,b}`：指定枠へ1着・2着の車番から集約。9車の場合、4/5→4枠、6/7→5枠、8/9→6枠。

プログラム：[`ticket_space.py`](ticket_space.py)。**全賭式の全価格を実際の市場データから確認するまで、全買い目を評価済みとは言わない**。未発売賭式はオッズ0ではなく「発売なし」。欠損価格は推測しない。

## 4. 統合確率を全順序確率へ接続する方法と限界

補正した市場統合の1着確率を `c_i`、元のG3独立1着確率を `q_i` としたとき、本基盤ではG3の既存2着・3着の条件付きモデルを**そのまま固定**して次式で1着の周辺確率だけを市場で補正する：

`P_market(i,j,k) = c_i × P_G3(j|i;q) × P_G3(k|i,j;q)`。

これは数学的に確率整合しており、1着周辺確率が `c_i` になる。一方、G3の2着・3着を `c` で再学習したわけではない。**ベンターの本文にこの厳密な競輪用接続が書かれているわけでもない**。したがって、3連単・3連複などの全順序確率について、別期間の実結果によるモデル妥当性を検証するまでは**探索版**。

## 5. 保存済みデータをどこまで使用できるか

- G3・S級の独自1着データ：2022〜2026上期 **8,995R**。うち2023〜2026上期の年次学習から作った `q` は**7,034R**。`q` の生成・保存：[export_prior_g3_q.py](export_prior_g3_q.py)。生成物は [`../data/g3_prior_year_first_q_2023_2026h1.jsonl.gz`](../data/g3_prior_year_first_q_2023_2026h1.jsonl.gz)。着順は較正時の**教師ラベル**で、実レース予測入力には入れない。
- 旧データの通常オッズ専用書き出しは**3連単・3連複の確定オッズのみ**（36,383R、全グレード合計）。G3の完全2車単市場の根拠にはならない。
- 7賭式の暗号化保存アーカイブには**全グレード合計で少なくとも134,336の重複除去済みレース数下界**という監査情報がある。**G3の2車単が何R・どのタイミングで復号可能かは未監査**。
- 昔の確定オッズを「締切15分前オッズ」と偽ることは厳禁。**最終オッズのα・βをそのままT−15分へ転用することも不可**。
- **実G3市場＋過去の年次独立q＋確定1着の完全マッチ件数が分かるまではα・βを仮定しない**。

## 6. 再現プログラム

前提：`Python 3.11+`、`numpy`、`scipy`。GitHub ActionsによるCIあり。

```bash
# 1. 原始G3 ZIPから「前年までで学習した1着独立確率」を復元（保存済み）
python research/g3_independent_v1/benter_market/export_prior_g3_q.py

# 2. 実際の過去2車単価格（全組番）を取得し、別ファイルで監査・保存した後に突合
python research/g3_independent_v1/benter_market/join_history.py --market ACTUAL_G3_2車単_HISTORY.jsonl --output stage/g3_benter_market_history.jsonl

# 3. 同じ価格時点のサンプルだけでベンター式αβを実結果から推定・期間外検証
python research/g3_independent_v1/benter_market/calibrate.py --history stage/g3_benter_market_history.jsonl --phase final --output stage/g3_benter_final_calibration.json

# 4. 将来のレースは実際のT−15分の全発売賭式を取り込んで監査
#    本番のオッズを使ったEV評価は、同じT−15分帯でαβを校正できてからのみ可能
python research/g3_independent_v1/benter_market/evaluate.py --race VALID_ACTUAL_MARKET_INPUT.json --mode live --calibration VERIFIED_T15_CALIBRATION.json --output stage/g3_market_result.json

# 係数未推定の現在は --calibration を付けずに実行
# → CALIBRATION_REQUIRED_NOT_EV_READY（全券種監査と市場proxyまで。EV+の断定なし）
```

### 新しい会話での推奨：1コマンドで全部つなぐ

過去データの較正は上の手順による。未来の指定レースでは、**G3独自1着確率の計算もこのコマンドで内部実行**するため、qを手作業で書く必要がない。

```bash
python research/g3_independent_v1/benter_market/run_pipeline.py --input ACTUAL_RAW_G3_CARD_AND_MARKET.json --mode live --output stage/g3_precheck.json
```

`--calibration` を省略したときは全7賭式の確率と価格の整合監査まで行い、**EVプラス断定を拒否**する。G3のT−15分に実証済みのαβが手に入った場合だけ、同じコマンドに `--calibration ACTUAL_G3_T15_CALIBRATION.json` を追加できる。

この入力は `racecard` に**実際の出走選手情報**（選手ごとの `car_no`, `score`, `age`, `term`, `b_count`, `s_count`, `nige_count`, `makuri_count`, `sashi_count`, `mark_count`, `first_count`, `second_count`, `third_count`, `outside_count`, `win_rate`, `top2_rate`, `top3_rate`, `gear` 等）を渡し、`market_snapshot` に実際の発売オッズを渡す。凍結済み37特徴量モデルがqを計算してから7券種の価格判定へ進む。

[`predict_g3_first.py`](predict_g3_first.py) は凍結したG3だけの1着係数を読み込む。出走表の `source_url` と `snapshot_captured_at_utc` が必須で、確定着順やオッズは特徴量へ入れない。

### 入力契約

`VALID_ACTUAL_MARKET_INPUT.json` は `racecard` と `market_snapshot` の2つのルート項目からなる。

`racecard`：`race_id`、`meeting_grade=G3`、`race_type`（S級）、`probability_model='G3 independent 1st v1'`、`predicted_line_formation`、`first_win=[{car_no,probability},...]`。**既存のG3独自モデルが実際の出走表を用いて計算した値を読み込み、想像で作らない**。

`market_snapshot`：同じ `race_id`、取得元 `source_url`、監査可能な `capture_provenance`、タイムゾーン付き `captured_at_utc` と `scheduled_start_at_utc`、`odds_phase='T-15'`、実際に販売している `sold_bet_types`、全価格 `quotes=[{bet_type,selection,decimal_odds},...]`、非発売組番 `not_offered_tickets=[{bet_type,selection,source_reason}]`。

**T−15分は発走15±2分の記録のみ受理**。市場価格がレンジ表示の場合、その上限下限を都合よく実配当にせず確定倍率が得られるまで停止。発売中の賭式で価格が欠ける買い目が1つでもあれば、**全買い目評価完了の表示を拒否**。

### リスク・配分

全価格から `c_ticket × odds > 1` の買い目だけを抽出する**数式インターフェース**は用意。ただし期待値計算前に較正検証ゲートがある。的中確率が不確実で、プール規模・自己投票による配当変化が不明な場合、**購入額は `null`**。ケリー基準で独断の配分や購入点数上限を作らない。

## 7. 完成と未完成

| 対象 | 状態 |
|---|---|
| G3の1着・2着・3着研究モデル | 凍結済み |
| 全7賭式の的中確率に変換するコード | 実装・CIテスト済み |
| 過去年次G3独立1着確率の保存 | 再生成して監査（別GitHub Actions） |
| 2車単から公衆1着proxyの計算 | コード完成、**実G3値での較正未検証** |
| 実過去G3オッズと独立確率の突合 | **未完了** |
| ベンター第2段階のαβ | **未推定** |
| G3全7券種の市場価格誤差の実証 | **未実施** |
| 15分前の予測・オッズと市場誤差の検証 | **未実施** |
| 購入点数・投票金額 | **未確定** |

以上を次会話にそのまま引き継ぐ。新しい独自モデルを勝手に作り直す必要はなく、**欠けた価格データと市場統合の統計的推定**から実行する。

## 8. 新しい会話への移動

次の会話では、[NEW_CHAT_HANDOFF.md](NEW_CHAT_HANDOFF.md) の指示文を貼り付け、このディレクトリと本マニフェストをGitHubで読み込む。**この会話では今日の個別レースの判定は行わない。**
