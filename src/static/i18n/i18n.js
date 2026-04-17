/**
 * Lightweight client-side i18n module.
 *
 * Usage:
 *   - Tag elements:  <span data-i18n="topbar.brand">邮件摘要助手</span>
 *   - For attributes: data-i18n-placeholder="command.keyword_placeholder"
 *                     data-i18n-title="hero.edit_btn_title"
 *                     data-i18n-aria-label="hero.aria_label"
 *   - For innerHTML (trusted content like the tutorial):
 *                     data-i18n-html="tutorial.s1_body"
 *   - In JS:  i18n.t("js.cookie_saved")
 *             i18n.t("js.deep_selected_count", { n: 5, max: 100 })
 */
(function () {
  "use strict";

  const STORAGE_KEY = "app_language";
  const SUPPORTED = ["zh", "en"];
  const DEFAULT_LANG = "zh";

  let _currentLang = DEFAULT_LANG;
  let _translations = {};  // { zh: {...}, en: {...} }
  let _ready = false;
  const _readyCallbacks = [];

  /* ── public API ──────────────────────────────────── */

  const i18n = {
    /** Current language code */
    get lang() { return _currentLang; },

    /**
     * Translate a key, with optional {placeholder} interpolation.
     * Falls back to zh → key itself.
     */
    t: function (key, params) {
      const dict = _translations[_currentLang] || _translations[DEFAULT_LANG] || {};
      let val = dict[key];
      if (val === undefined) {
        const fallback = _translations[DEFAULT_LANG] || {};
        val = fallback[key];
      }
      if (val === undefined) return key;
      if (params) {
        Object.keys(params).forEach(function (k) {
          val = val.replace(new RegExp("\\{" + k + "\\}", "g"), params[k]);
        });
      }
      return val;
    },

    /** Switch language and re-render all tagged elements. */
    setLang: function (lang) {
      if (!SUPPORTED.includes(lang)) return;
      _currentLang = lang;
      localStorage.setItem(STORAGE_KEY, lang);
      document.documentElement.lang = _translations[lang]?.meta?.lang || lang;
      i18n.applyAll();
      // Fire custom event so other JS can react
      window.dispatchEvent(new CustomEvent("langchange", { detail: { lang: lang } }));
    },

    /** Toggle between zh and en. */
    toggle: function () {
      i18n.setLang(_currentLang === "zh" ? "en" : "zh");
    },

    /** Apply translations to all data-i18n tagged elements in the DOM. */
    applyAll: function () {
      // Text content
      document.querySelectorAll("[data-i18n]").forEach(function (el) {
        var key = el.getAttribute("data-i18n");
        if (key) el.textContent = i18n.t(key);
      });
      // innerHTML (for trusted content with markup)
      document.querySelectorAll("[data-i18n-html]").forEach(function (el) {
        var key = el.getAttribute("data-i18n-html");
        if (key) el.innerHTML = i18n.t(key);
      });
      // Attributes
      var attrPrefixes = ["placeholder", "title", "aria-label"];
      attrPrefixes.forEach(function (attr) {
        document.querySelectorAll("[data-i18n-" + attr + "]").forEach(function (el) {
          var key = el.getAttribute("data-i18n-" + attr);
          if (key) el.setAttribute(attr, i18n.t(key));
        });
      });
      // Update page title
      var titleKey = document.documentElement.getAttribute("data-i18n-title");
      if (titleKey) document.title = i18n.t(titleKey);
      // Update lang toggle button label
      var toggleLabel = document.getElementById("langLabel");
      if (toggleLabel) {
        toggleLabel.textContent = _currentLang === "zh" ? "EN" : "中文";
      }
      var mobileMoreLangLabel = document.getElementById("mobileMoreLangLabel");
      if (mobileMoreLangLabel) {
        mobileMoreLangLabel.textContent = _currentLang === "zh" ? "EN" : "中文";
      }
    },

    /** Register a callback for when translations are loaded. */
    onReady: function (fn) {
      if (_ready) fn();
      else _readyCallbacks.push(fn);
    },

    /** Pre-loaded translations (for login/register pages that inline them). */
    load: function (lang, data) {
      _translations[lang] = data;
    }
  };

  /* ── initialization ──────────────────────────────── */

  // Detect saved preference
  var saved = localStorage.getItem(STORAGE_KEY);
  if (saved && SUPPORTED.includes(saved)) {
    _currentLang = saved;
  }

  // Fetch both translation files, then apply
  function fetchJson(url) {
    return fetch(url).then(function (r) { return r.json(); });
  }

  Promise.all([
    fetchJson("/static/i18n/zh.json"),
    fetchJson("/static/i18n/en.json")
  ]).then(function (results) {
    _translations.zh = results[0];
    _translations.en = results[1];
    document.documentElement.lang = _translations[_currentLang]?.meta?.lang || _currentLang;
    i18n.applyAll();
    _ready = true;
    _readyCallbacks.forEach(function (fn) { fn(); });
  }).catch(function (err) {
    console.warn("[i18n] Failed to load translations:", err);
    _ready = true;
    _readyCallbacks.forEach(function (fn) { fn(); });
  });

  window.i18n = i18n;
})();
