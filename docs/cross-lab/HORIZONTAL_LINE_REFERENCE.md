# CROSS LAB 横棒（－）ライン筋・参考パネル

対象：CROSS LABで選択した二線の横棒（同ライン）決着を、クロスの縦・斜め4経路に**追加で表示**する。

## 表示例

選択ライン A = 1－2、B = 3－4 の場合：

| 区分 | 1着→2着 | 扱い |
|---|---|---|
| A順筋 | 1→2 | 横棒のライン筋、参考 |
| A逆筋 | 2→1 | 同ライン逆転、参考 |
| B順筋 | 3→4 | 横棒のライン筋、参考 |
| B逆筋 | 4→3 | 同ライン逆転、参考 |

- 各1→2着について、3着には出走するその他の全選手を列挙し、取得済み3連単オッズを出す。
- 全件オッズのない場合は該当オッズを「—」にして市場内重みを表示しない。
- 全件オッズのある場合のみ、逆オッズ総和で正規化した参考**市場内重み**を表示する。これはレース結果に基づく的中率・実確率ではない。
- **買い目の自動追加なし、資金配分の変更なし、クロス経路の採点・選択を変更しない。**
- 3-2-2の2車×2車クロスの3着候補+1の処理（`322_22_third_expansion.mjs`）とは独立。

## 既存サイトへの組み込み例

既存CROSS LABのデータオブジェクト名は本リポジトリにはないため、
以下の `race.lines`、`chosenTwoLineIndices`、`race.trifectaOdds` は**説明用プレースホルダ**。
実際のサイトのソース内で使っているデータに合わせて接続すること。

```js
import {
  buildHorizontalLineReference,
  renderHorizontalLineReference,
} from "./horizontal_line_reference.mjs";

// クロス結果カードの後ろに、独立した参考表示領域を追加する。
// 例：<div id="line-reference"></div>
const reference = buildHorizontalLineReference({
  lines: race.lines,                // [[5,6,7],[1,2],[3,4]]
  selectedLineIndexes: chosenTwoLineIndices, // [1,2]
  odds: race.trifectaOdds,          // {"1-2-3": 23.4, ...}
});
renderHorizontalLineReference(
  document.getElementById("line-reference"), reference
);
```

テスト：`node --test docs/cross-lab/horizontal_line_reference.test.mjs`

**公開状態**：本コードはGitHubの独立モジュール。ChatGPT Site
`https://keirin-cross-lab.kuwawa01.chatgpt.site` への直接の編集・公開は未実施。
サイトの編集権限ある環境で統合して再公開が必要。
