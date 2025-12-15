(() => {
  const btn = document.getElementById('toggleUnbrand');
  const root = document.documentElement;

  function setPressed(state) {
    btn.setAttribute('aria-pressed', String(state));
    btn.textContent = state ? 'Unbrand → Brand' : 'Brand → Unbrand';
  }

  btn?.addEventListener('click', () => {
    const isOn = document.body.classList.toggle('unbrand');
    setPressed(isOn);
  });

  // Subtelny wskaźnik przewinięcia (for effects via CSS vars)
  const onScroll = () => {
    const h = document.documentElement;
    const sc = h.scrollTop;
    const max = (h.scrollHeight - h.clientHeight) || 1;
    const t = Math.max(0, Math.min(1, sc / max));
    root.style.setProperty('--scroll', String(t));
  };
  document.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();

