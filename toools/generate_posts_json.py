#!/usr/bin/env python3
# tools/generate_posts_json.py
import json, re, glob, os, sys
from datetime import datetime
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

NOTE_DIR = os.environ.get("NOTE_DIR", "notes")
OUT = os.environ.get("OUT_FILE", "posts.json")

def split_front_matter(md_text: str):
    if md_text.startswith("---"):
        m = re.search(r"^---\n(.*?)\n---\n", md_text, re.S | re.M)
        if m:
            fm = m.group(1)
            body = md_text[m.end():]
            return fm, body
    return None, md_text

def parse_meta(fm_text):
    if not fm_text: return {}
    if yaml:
        try:
            data = yaml.safe_load(fm_text) or {}
            return dict(data)
        except Exception:
            pass
    # フォールバック（key: value の素朴パース）
    meta = {}
    for line in fm_text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta

def first_h1(md_body):
    m = re.search(r"^#\s+(.+)$", md_body, re.M)
    return m.group(1).strip() if m else None

def strip_md(text):
    # 粗削りでOK：コードブロック/行内コード/リンク記法などを潰す
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)  # 画像
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # リンク
    text = re.sub(r"^>+\s?", "", text, flags=re.M)  # 引用記号
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)  # 見出し #
    text = re.sub(r"\*\*([^*]+)\*\*|\*([^*]+)\*", r"\1\2", text)  # 強調
    text = re.sub(r"\s+", " ", text).strip()
    return text

def pick_excerpt(body, n=140):
    body = re.sub(r"^---.*?---\n", "", body, flags=re.S)  # 念のため
    # 最初の非空行〜2,3文くらい
    txt = strip_md(body).strip()
    return (txt[:n] + "…") if len(txt) > n else txt

posts = []
for path in sorted(glob.glob(os.path.join(NOTE_DIR, "*.md"))):
    slug = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8") as f:
        md = f.read()

    fm_text, body = split_front_matter(md)
    meta = parse_meta(fm_text)

    title = meta.get("title") or first_h1(body) or slug
    # date は front matter > git mtime > fs mtime の順に
    date = meta.get("date")
    if not date:
        try:
            # git 最終コミット時刻
            import subprocess
            ts = subprocess.check_output(["git","log","-1","--format=%ct","--", path], text=True).strip()
            date = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
        except Exception:
            date = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")

    tags = meta.get("tags", [])
    if isinstance(tags, str):
        # "a, b, c" 形式のフォールバック
        tags = [t.strip() for t in re.split(r"[,\s]+", tags) if t.strip()]

    excerpt = meta.get("excerpt") or pick_excerpt(body)

    posts.append({
        "slug": slug,
        "title": title,
        "date": date,
        "tags": tags,
        "excerpt": excerpt
    })

# 日付降順 → slug 昇順で安定化
posts.sort(key=lambda x: (x.get("date",""), x["slug"]), reverse=True)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(posts, f, ensure_ascii=False, indent=2)

print(f"[ok] wrote {OUT} ({len(posts)} posts)")
