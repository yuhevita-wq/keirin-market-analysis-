# LIVE ODDS ISOLATION — 締切直前の全券種監視

## 目的

締切直前の全発売券種・全組番のオッズを保存し、他市場からの支持と乖離する車券を探す。ワイドを除外しない。結果・ライン・選手能力を推定器には入力しない。

## 現在の実装と限界

- 許諾を得たKドリームスのHTMLを1レースにつき45秒ごとに確認する。別配信元の自動フォールバックは未実装。
- 7車なら5券種329点。9車は枠系を含む7券種783点。全組番・車番・同一更新時刻を厳密検査。
- 元データの時刻が取得時点の4分より古ければ STALE として失敗。レース日、券種、時刻、車番に不整合があればデータ不足のまま中止。
- 生オッズは暗号化せずに公開GitHubに保存しない。GitHub Actionsは完全スナップショットだけを公開証明書でAES-256-GCM暗号化して保存。
- 確定オッズをライブ速報として扱わない。GitHub Actionsの起動遅延やサイト側キャッシュで取得できないときは取得失敗と表示。
- 実サイトでの締切前安定取得は未検証。GitHub Actionsだけでは秒単位の取得保証を提供できない。

## スコアの意味

各券種の逆数オッズを券種内部で正規化する。ワイドは1レースで3組が同時的中するため、支持総量3に正規化する。

3連単のスコアは、当該3連単のオッズを見ないで2車単と3連複の周辺支持をIPFで整合させ、各順序の予測支持と比較する。他の券種についても包含事象の周辺確率を別券種から合算して比較する。

これは市場支持の矛盾を示す暫定指標であり、期待値・的中確率・市場の間違いを証明しない。取得失敗は捏造して補完しない。検証はレース前のランキングを固定して、結果を後から別ラベルとして照合する。

## GitHub上で使う

[Live odds isolation Actions](https://github.com/yuhevita-wq/keirin-market-analysis-/actions/workflows/live-odds-isolation.yml) の Run workflow で：

1. race_url に Kドリームスの当該 racedetail URL。
2. close_jst に投票締切を 2026-10-16T20:25+09:00 の書式で指定（これは日時の書式例）。
3. 自動取得・保存の許諾を確認できる場合に限り confirm_authorized にチェック。
4. 締切20分前程度から起動。ログに FRESH、UNCHANGED、FAILED、CLOSED、上位車券、時刻が出る。古いデータなら FAILED。
5. 終了後は検証済み完全スナップショットの暗号文のみコミットする。復号鍵はGitHubに置かない。

## 常駐端末での実行例

    python -m pip install requests beautifulsoup4
    python -m odds_market.test_collect_all_odds
    python -m odds_market.test_live_watch
    python -m odds_market.live_watch \
      --url 'https://keirin.kdreams.jp/shizuoka/racedetail/3820261015010003/?pageType=odds' \
      --close-jst '2026-10-15T21:15+09:00' \
      --interval-seconds 45 --out /tmp/keirin_live_odds_private \
      --accept-terms

これは終了済みレースのコマンド書式例であり、今から実行してもライブ取得はできない。

## 次に必要なもの

1. 許諾を得た複数配信元で、ソースの真正時刻と券種網羅率を実証。
2. レース開催・公式締切の予約ディスカバリ。ワークフローの遅延を回避するには常駐サービスが必要。
3. 薄い投票プールとワイド変動による見かけの乖離を区別。
4. 記録したレース前の異常ランキングを結果・払戻と時間外検証し、軽微な値付け差ではない誤差を探索する。
