// post: クエリ ?p=slug を取得し、対応する Markdown を描画
(async function(){
  const params = new URLSearchParams(location.search);
  const slug = params.get('p');
  if(!slug) return;
  const mdPath = `/posts/${slug}.md`;

  try{
    const res = await fetch(mdPath, {cache:'no-store'});
    if(!res.ok) throw new Error('404');
    const raw = await res.text();

    const {data, content} = parseFrontMatter(raw);
    const title = (data.title || firstH1(content) || slug).trim();
    const date = data.date? new Date(data.date).toLocaleDateString(): '';

    document.getElementById('post-title').textContent = title;
    document.getElementById('doc-title').textContent = `${title} — thada`;
    document.getElementById('post-meta').textContent = date;

    // marked で Markdown -> HTML
    if(window.marked){
      marked.setOptions({ breaks:true, gfm:true });
      document.getElementById('post-body').innerHTML = marked.parse(content);
    }else{
      document.getElementById('post-body').textContent = content; // フォールバック
    }

    // シェアボタン
    const url = location.href;
    document.getElementById('share-twitter').onclick = ()=>{
      const u = new URL('https://twitter.com/intent/tweet');
      u.searchParams.set('text', title);
      u.searchParams.set('url', url);
      window.open(u.toString(), '_blank');
    };
    document.getElementById('share-facebook').onclick = ()=>{
      const u = new URL('https://www.facebook.com/sharer/sharer.php');
      u.searchParams.set('u', url);
      window.open(u.toString(), '_blank');
    };
    document.getElementById('share-copy').onclick = async ()=>{
      try{ await navigator.clipboard.writeText(url); alert('リンクをコピーしました'); }catch(e){ prompt('コピーしてください:', url); }
    };

  }catch(e){
    document.getElementById('post-title').textContent = 'Not Found';
    document.getElementById('post-body').innerHTML = '<p class="muted">記事が見つかりませんでした。</p>';
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
})();

/* end post.js */
