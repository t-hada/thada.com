// index: posts/index.json を読み、各 Markdown からタイトル/要旨/日付を抽出
(async function(){
  const listEl = document.getElementById('posts');
  if(!listEl) return;
  try{
    const res = await fetch('/posts/index.json', {cache:'no-store'});
    if(!res.ok) throw new Error('index.json not found');
    const items = await res.json(); // [{slug, path}] or [{slug, path, date, title, description}]

    const enriched = await Promise.all(items.map(async it => {
      const {slug, path} = it;
      let md = '';
      try{
        const r = await fetch(path, {cache:'no-store'});
        md = await r.text();
      }catch(e){}

      const {data, content} = parseFrontMatter(md);
      const title = (data.title || firstH1(content) || it.title || slugifyToTitle(slug)).trim();
      const date = (data.date || it.date || '').trim();
      const desc = (data.description || firstParagraph(content) || it.description || '').trim();
      return {slug, path, title, date, desc};
    }));

    // 日付があれば降順でソート
    enriched.sort((a,b)=> (Date.parse(b.date||'0')||0) - (Date.parse(a.date||'0')||0));

    for(const post of enriched){
      const a = document.createElement('a');
      a.className = 'post-card';
      a.href = `/post.html?p=${encodeURIComponent(post.slug)}`;
      a.innerHTML = `
        <h2>${escapeHtml(post.title)}</h2>
        <div class="meta">${post.date? new Date(post.date).toLocaleDateString(): ''}</div>
        ${post.desc? `<p>${escapeHtml(post.desc)}</p>`: ''}
      `;
      listEl.appendChild(a);
    }
  }catch(e){
    listEl.innerHTML = `<p class="muted">投稿が見つかりませんでした。/posts/index.json と Markdown を配置してください。</p>`;
  }

  function parseFrontMatter(md){
    const m = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/m.exec(md);
    if(!m) return {data:{}, content:md||''};
    const data = {};
    m[1].split(/\n/).forEach(line=>{
      const mm = /^(\w[\w-]*):\s*(.*)$/.exec(line);
      if(mm) data[mm[1]] = mm[2];
    });
    return {data, content:m[2]};
  }
  function firstH1(md){
    const m = /^#\s+(.+)$/m.exec(md||'');
    return m? m[1]: '';
    }
  function firstParagraph(md){
    const cleaned = (md||'').replace(/```[\s\S]*?```/g,'').trim();
    const m = /\n{0,2}([^\n][\s\S]*?)\n{2,}/.exec('\n'+cleaned+'\n\n');
    return m? m[1].replace(/\n/g,' ').slice(0,160): '';
  }
  function slugifyToTitle(slug){
    return (slug||'').replace(/[-_]/g,' ').replace(/\b\w/g, s=>s.toUpperCase());
  }
  function escapeHtml(s){
    return (s||'').replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
  }
})();

/* end main.js */
