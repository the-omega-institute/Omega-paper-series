/**
 * Omega Showcase — i18n language toggle
 *
 * Toggles between zh (Chinese) and en (English) for:
 * 1. Content blocks marked with .lang-zh / .lang-en classes
 * 2. Toggle button label
 * Note: Videos stay in their original language (no source swap)
 * since English video versions are not yet generated for all content.
 *
 * Preference saved to localStorage.
 */

(function() {
  'use strict';

  var STORAGE_KEY = 'omega-lang';
  var DEFAULT_LANG = 'zh';

  function getLang() {
    try {
      return localStorage.getItem(STORAGE_KEY) || DEFAULT_LANG;
    } catch(e) {
      return DEFAULT_LANG;
    }
  }

  function setLang(lang) {
    try { localStorage.setItem(STORAGE_KEY, lang); } catch(e) {}
    document.documentElement.setAttribute('data-lang', lang);
    applyLang(lang);
  }

  function applyLang(lang) {
    // Toggle .lang-zh / .lang-en content blocks
    document.querySelectorAll('.lang-zh').forEach(function(el) {
      el.style.display = (lang === 'zh') ? '' : 'none';
    });
    document.querySelectorAll('.lang-en').forEach(function(el) {
      el.style.display = (lang === 'en') ? '' : 'none';
    });

    // Videos stay in original language — no source swap.
    // When English versions are generated for all content, re-enable swap here.

    // Update toggle button text
    var btn = document.getElementById('lang-toggle-btn');
    if (btn) {
      btn.textContent = (lang === 'zh') ? 'EN' : '中文';
      btn.title = (lang === 'zh') ? 'Switch to English' : '切换到中文';
    }
  }

  // Create toggle button and inject into navbar
  function createToggle() {
    var btn = document.createElement('button');
    btn.id = 'lang-toggle-btn';
    btn.className = 'btn btn-sm';
    btn.style.cssText = 'margin-left: 1rem; padding: 4px 12px; border: 1px solid rgba(255,255,255,0.3); border-radius: 4px; color: #e6e6e6; background: transparent; cursor: pointer; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.05em;';

    btn.addEventListener('click', function() {
      var current = getLang();
      setLang(current === 'zh' ? 'en' : 'zh');
    });

    // Find navbar right section and append
    var navRight = document.querySelector('.navbar-nav.navbar-nav-scroll.ms-auto');
    if (!navRight) navRight = document.querySelector('.navbar .navbar-collapse');
    if (navRight) {
      var li = document.createElement('li');
      li.className = 'nav-item';
      li.style.cssText = 'display: flex; align-items: center;';
      li.appendChild(btn);
      navRight.appendChild(li);
    }
  }

  // Initialize
  document.addEventListener('DOMContentLoaded', function() {
    createToggle();
    var lang = getLang();
    document.documentElement.setAttribute('data-lang', lang);
    applyLang(lang);
  });
})();