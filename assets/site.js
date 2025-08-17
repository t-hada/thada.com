/** ユーティリティ **/
const fmtDate = (iso) => {
  if(!iso) return "";
  const d = new Date(iso);
  const y = d.getFullYear(), m = String(d.getMonth()+1).padStart(2,'0'), day = String(d.getDate()).padStart(2,'0');
  return `${y}-${m}-${day}`;
};
const q = (sel, root=document)=>root.querySelector(sel);

/** index.html 用：posts.json を読み込み一覧を描画 */
async function renderList() {
  const el = q('#notes-list'); if(!el) return;
  try {
    const res = await fetch('posts.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const posts = await res.json();
    posts.sort((a,b)=> new Date(b.date||0)-new Date(a.date||0));
    el.innerHTML = posts.map(p=>`
      <a class="item" href="post.html?p=${encodeURIComponent(p.slug)}">
        <div>
          <h3>${p.title}</h3>
          <div class="meta">${fmtDate(p.date)}
            ${Array.isArray(p.tags)? p.tags.map(t=>`<span class="tag">${t}</span>`).join(''):''}
          </div>
          ${p.excerpt? `<div class="subtle" style="margin-top:6px">${p.excerpt}</div>`:''}
        </div>
      </a>`).join('');
  } catch (e) {
    console.error(e);
    el.innerHTML = `<div class="subtle">posts.json を読み込めませんでした：${String(e).replace(/</g,'&lt;')}</div>`;
  }
}

/** post.html 用：?p=slug の .md を取得 → Markdown を HTML に */
async function renderPost() {
  const outlet = q('#post'); if(!outlet) return;
  const slug = new URLSearchParams(location.search).get('p');
  if(!slug){ outlet.innerHTML = '<p class="subtle">記事が指定されていません。</p>'; return; }
  const mdUrl = `notes/${slug}.md`;
  try {
    const res = await fetch(mdUrl, { cache: 'no-store' });
    if(!res.ok){ outlet.innerHTML = `<p class="subtle">記事が見つかりません: ${mdUrl}</p>`; return; }
    const md = await res.text();

    // ―― YAML フロントマター（任意）を抜く
    let meta = {}; let body = md;
    if(md.startsWith('---')){
      const end = md.indexOf('\n---', 3);
      if(end>0){
        const fm = md.slice(3,end).trim();
        body = md.slice(end+4);
        fm.split('\n').forEach(line=>{
          const i=line.indexOf(':'); if(i>0){ meta[line.slice(0,i).trim()] = line.slice(i+1).trim(); }
        });
      }
    }
    // タイトル決定（front matter の title > 先頭 # 見出し > slug）
    let title = meta.title || (body.match(/^#\s+(.+)$/m)?.[1]) || slug;
    document.title = `${title} — thada's note`;

    // Markdown → HTML（marked があれば使う／なければ超簡易）
    let html = '';
    if(window.marked){
      html = marked.parse(body);
    }else{
      html = body
        .replace(/^###\s(.+)$/mg,'<h3>$1</h3>')
        .replace(/^##\s(.+)$/mg,'<h2>$1</h2>')
        .replace(/^#\s(.+)$/mg,'<h1>$1</h1>')
        .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
        .replace(/\*(.+?)\*/g,'<em>$1</em>')
        .replace(/`([^`]+)`/g,'<code>$1</code>')
        .replace(/\n{2,}/g,'</p><p>')
        .replace(/^(?!<h\d|<p|<ul|<pre|<code)/mg,'<p>$&');
      html = `<p>${html}</p>`;
    }

    q('#post-title').textContent = title;
    outlet.innerHTML = html;
  } catch (e) {
    console.error(e);
    outlet.innerHTML = `<p class="subtle">本文を読み込めませんでした：${String(e).replace(/</g,'&lt;')}</p>`;
  }
}

window.addEventListener('DOMContentLoaded', ()=>{
  renderList();
  renderPost();
});
