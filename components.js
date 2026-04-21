class SiteNav extends HTMLElement {
  connectedCallback() {
    const base  = this.getAttribute('base') ?? '';
    const active = this.getAttribute('active') ?? '';

    this.innerHTML = `
      <header>
        <nav>
          <a href="${base}index.html" class="logo">Sungmo Ku</a>
          <ul>
            <li><a href="${base}index.html#about" ${active === 'about' ? 'class="active"' : ''}>About</a></li>
            <li><a href="${base}index.html#research" ${active === 'research' ? 'class="active"' : ''}>Research</a></li>
            <li><a href="${base}notes/" ${active === 'notes' ? 'class="active"' : ''}>Notes</a></li>
            <li><a href="${base}index.html#contact" ${active === 'contact' ? 'class="active"' : ''}>Contact</a></li>
          </ul>
        </nav>
      </header>
    `;
  }
}

class SiteFooter extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
      <footer>
        <p>&copy; 2026 Sungmo Ku. Built with GitHub Pages &mdash;
          assisted by <a href="https://claude.ai/code" target="_blank" rel="noopener">Claude Code</a>.
        </p>
      </footer>
    `;
  }
}

customElements.define('site-nav', SiteNav);
customElements.define('site-footer', SiteFooter);
