(() => {
  const els = Array.from(document.querySelectorAll('.reveal'));
  if (!('IntersectionObserver' in window) || els.length === 0) {
    els.forEach(el => el.classList.add('is-visible'));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) e.target.classList.add('is-visible');
    });
  }, { root: null, rootMargin: '0px 0px -10% 0px', threshold: 0.1 });
  els.forEach(el => io.observe(el));
  
  // Scroll progress
  const bar = document.querySelector('.progress__bar');
  const updateProgress = () => {
    const h = document.documentElement;
    const max = (h.scrollHeight - h.clientHeight) || 1;
    const t = Math.max(0, Math.min(1, h.scrollTop / max));
    if (bar) bar.style.width = (t * 100).toFixed(1) + '%';
  };
  document.addEventListener('scroll', updateProgress, { passive: true });
  updateProgress();

  // TOC active link
  const tocAnchors = Array.from(document.querySelectorAll('.toc a'));
  const tocLinks = new Map(tocAnchors.map(a => [a.getAttribute('href').slice(1), a]));
  const sections = tocAnchors
    .map(a => a.getAttribute('href').slice(1))
    .map(id => document.getElementById(id))
    .filter(Boolean);
  const activate = (id) => {
    document.querySelectorAll('.toc a.active').forEach(el => el.classList.remove('active'));
    const link = tocLinks.get(id);
    if (link) link.classList.add('active');
  };
  const getTop = (el) => el.getBoundingClientRect().top + window.scrollY;
  const updateActive = () => {
    if (!sections.length) return;
    const doc = document.documentElement;
    const scBottom = window.scrollY + window.innerHeight;
    const docHeight = doc.scrollHeight;
    // If we are at the very bottom, force last section
    if (scBottom >= docHeight - 2) {
      activate(sections[sections.length - 1].id);
      return;
    }
    const center = window.scrollY + window.innerHeight * 0.4; // 40% from top
    let current = sections[0].id;
    for (let i = 0; i < sections.length; i++) {
      const top = getTop(sections[i]);
      if (top <= center) current = sections[i].id; else break;
    }
    activate(current);
  };
  document.addEventListener('scroll', updateActive, { passive: true });
  window.addEventListener('load', updateActive);
  window.addEventListener('resize', updateActive);
  updateActive();

  // Copy BibTeX
  document.querySelectorAll('.btn-copy').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = document.querySelector(btn.dataset.copy);
      if (!target) return;
      const text = target.innerText;
      navigator.clipboard?.writeText(text).then(() => {
        const old = btn.textContent;
        btn.textContent = 'Copied';
        setTimeout(() => { btn.textContent = old; }, 1200);
      });
    });
  });
})();
