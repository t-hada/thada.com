document.addEventListener('DOMContentLoaded', () => {

    // --- Page-specific logic --- //
    // Run logic based on the presence of specific IDs on the page.
    if (document.getElementById('lang-switch')) {
        initializeLanguageSwitch();
        renderPublications();
    }
    if (document.getElementById('post')) {
        renderPost();
    }
});

/**
 * Initializes the language switcher functionality.
 */
function initializeLanguageSwitch() {
    const langSwitchButton = document.getElementById('lang-switch');

    async function loadLanguage(lang) {
        try {
            const response = await fetch(`lang/${lang}.json`);
            if (!response.ok) throw new Error(`Could not load ${lang}.json`);
            const langData = await response.json();

            if (langData.page_title) {
                document.title = langData.page_title;
            }

            document.querySelectorAll('[data-lang-key]').forEach(element => {
                const key = element.getAttribute('data-lang-key');
                if (langData[key]) {
                    element.innerHTML = langData[key];
                }
            });

            langSwitchButton.textContent = langData.language_switcher;
            langSwitchButton.setAttribute('data-lang-toggle', lang === 'ja' ? 'en' : 'ja');
            document.documentElement.lang = lang;

            // Re-render publications with the new language
            renderPublications();

        } catch (error) {
            console.error('Language loading failed:', error);
        }
    }

    langSwitchButton.addEventListener('click', () => {
        const newLang = langSwitchButton.getAttribute('data-lang-toggle');
        localStorage.setItem('preferred_language', newLang);
        loadLanguage(newLang);
    });

    const preferredLang = localStorage.getItem('preferred_language') || 'ja';
    loadLanguage(preferredLang);
}

/**
 * Utility to format date strings.
 */
const fmtDate = (iso) => {
  if(!iso) return "";
  const d = new Date(iso);
  const y = d.getFullYear(), m = String(d.getMonth()+1).padStart(2,'0'), day = String(d.getDate()).padStart(2,'0');
  return `${y}-${m}-${day}`;
};

/**
 * Fetches posts.json and renders the list of publications on the main page.
 */
async function renderPublications() {
  const el = document.querySelector('#notes-list');
  if(!el) return;

  const lang = document.documentElement.lang || 'ja';

  try {
    const res = await fetch('posts.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const posts = await res.json();
    
    posts.sort((a,b)=> new Date(b.date||0)-new Date(a.date||0));

    el.innerHTML = posts.map(p=>{
      const title = (lang === 'ja' ? p.title_ja : p.title_en) || p.title_ja;
      const excerpt = (lang === 'ja' ? p.excerpt_ja : p.excerpt_en) || p.excerpt_ja;

      return `
      <a class="item" href="post.html?p=${encodeURIComponent(p.slug)}">
        <div>
          <h3>${title}</h3>
          <div class="meta">${fmtDate(p.date)}
            ${Array.isArray(p.tags)? p.tags.map(t=>`<span class="tag">${t}</span>`).join(''):''}
          </div>
          ${excerpt? `<div class="subtle">${excerpt}</div>`:''}
        </div>
      </a>`
    }).join('');
  } catch (e) {
    console.error(e);
    el.innerHTML = `<div class="subtle" style="color: var(--text-color-light);">Failed to load publications: ${String(e)}</div>`;
  }
}

/**
 * Fetches a markdown file based on URL param and renders it as a post.
 * It tries to fetch the language-specific version first, then falls back to Japanese.
 */
async function renderPost() {
  const outlet = document.querySelector('#post');
  if (!outlet) return;

  const slug = new URLSearchParams(location.search).get('p');
  if (!slug) {
    outlet.innerHTML = '<p class="subtle">Article not specified.</p>';
    return;
  }

  const lang = document.documentElement.lang || 'ja';
  const primaryUrl = `notes/${slug}.${lang}.md`;
  const fallbackUrl = `notes/${slug}.ja.md`;

  try {
    let res = await fetch(primaryUrl, { cache: 'no-store' });

    // If the primary language file doesn't exist (e.g., .en.md is missing), try the fallback
    if (!res.ok) {
      res = await fetch(fallbackUrl, { cache: 'no-store' });
    }

    // If the fallback also fails, show an error
    if (!res.ok) {
      outlet.innerHTML = `<p class="subtle">Could not find article: ${slug}</p>`;
      return;
    }

    const md = await res.text();

    // --- Parse optional YAML front matter --- 
    let meta = {};
    let body = md;
    if (md.startsWith('---')) {
      const end = md.indexOf('\n---', 3);
      if (end > 0) {
        const fm = md.slice(3, end).trim();
        body = md.slice(end + 4);
        fm.split('\n').forEach(line => {
          const i = line.indexOf(':');
          if (i > 0) {
            meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
          }
        });
      }
    }

    // Determine title (front matter > h1 > slug)
    let title = meta.title || (body.match(/^#\s+(.+)$/m)?.[1]) || slug;
    document.title = `${title} — Tsubasa Hada`;

    // Always remove the H1 from the body to prevent duplication, as #post-title is used.
    body = body.replace(/^#\s+(.+)/m, '').trim();

    // --- Convert Markdown to HTML --- 
    let html = '';
    if (window.marked) {
      html = window.marked.parse(body);
    } else {
      // Basic fallback parser if marked.js is not available
      html = body
        .replace(/^###\s(.+)$/mg, '<h3>$1</h3>')
        .replace(/^##\s(.+)$/mg, '<h2>$1</h2>')
        .replace(/^#\s(.+)$/mg, '<h1>$1</h1>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\n{2,}/g, '</p><p>')
        .replace(/^(?!<h\d|<p|<ul|<pre|<code)/mg, '<p>$&</p>');
      html = `<p>${html}</p>`;
    }

    document.querySelector('#post-title').textContent = title;
    outlet.innerHTML = html;

  } catch (e) {
    console.error(e);
    outlet.innerHTML = `<p class="subtle">Failed to load content: ${String(e)}</p>`;
  }
}