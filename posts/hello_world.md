---
title: はじめに — ノート運用方針
date: 2025-08-17
description: このサイトの意図と、Markdown での執筆約束事のメモ。
---

# はじめに

ここは作業メモ置き場。FrontMatter の `title` / `date` / `description` があればトップに反映されます。未設定なら先頭の `# 見出し` と最初の段落から抽出。

- 記事ファイルは `/posts/<slug>.md`
- 一覧は `/posts/index.json` にスラッグを追記（最小限でOK）。

```bash
# 例: 新規記事 foo-bar.md を作ったとき
# posts/index.json に {"slug":"foo-bar","path":"/posts/foo-bar.md"} を追加
```

> 投稿は日付降順で表示。日付がない場合は順不同。

---

（ここから本文……）
