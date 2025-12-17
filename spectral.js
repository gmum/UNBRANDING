// Interactions adapted from horwitz.ai/spectral_detuning index.js
window.HELP_IMPROVE_VIDEOJS = false;

function toggleMoreWorks() {
  const dropdown = document.getElementById('moreWorksDropdown');
  const button = document.querySelector('.more-works-btn');
  if (!dropdown || !button) return;
  if (dropdown.classList.contains('show')) {
    dropdown.classList.remove('show');
    button.classList.remove('active');
  } else {
    dropdown.classList.add('show');
    button.classList.add('active');
  }
}

document.addEventListener('click', function (event) {
  const container = document.querySelector('.more-works-container');
  const dropdown = document.getElementById('moreWorksDropdown');
  const button = document.querySelector('.more-works-btn');
  if (!container || !dropdown || !button) return;
  if (!container.contains(event.target)) {
    dropdown.classList.remove('show');
    button.classList.remove('active');
  }
});

document.addEventListener('keydown', function (event) {
  if (event.key === 'Escape') {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    dropdown?.classList.remove('show');
    button?.classList.remove('active');
  }
});

function copyBibTeX() {
  const bibtexElement = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  const copyText = button?.querySelector('.copy-text');
  if (!bibtexElement || !button || !copyText) return;
  navigator.clipboard.writeText(bibtexElement.textContent || '').then(function () {
    button.classList.add('copied');
    copyText.textContent = 'Cop';
    setTimeout(function () {
      button.classList.remove('copied');
      copyText.textContent = 'Copy';
    }, 2000);
  }).catch(function () {
    const textArea = document.createElement('textarea');
    textArea.value = bibtexElement.textContent || '';
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
    button.classList.add('copied');
    copyText.textContent = 'Cop';
    setTimeout(function () {
      button.classList.remove('copied');
      copyText.textContent = 'Copy';
    }, 2000);
  });
}

function scrollToTop() { window.scrollTo({ top: 0, behavior: 'smooth' }); }

window.addEventListener('scroll', function () {
  const scrollButton = document.querySelector('.scroll-to-top');
  if (!scrollButton) return;
  if (window.pageYOffset > 300) scrollButton.classList.add('visible'); else scrollButton.classList.remove('visible');
});

