# AGENTS.md — Project instructions

## Site

- The deployed site lives in `site/` at `https://oscarw27.github.io/5294INOR6Y_G1/`.
- **Remember the sitemap:** `site/sitemap.xml` must stay in sync with the site. When a page is added, removed, renamed, or meaningfully changed under `site/`, update `sitemap.xml` in the same change (URLs, `<lastmod>` from the current date, priorities). A new `robots.txt` reference must not be broken. Validate the XML after editing (e.g. `python -c "import xml.dom.minidom; xml.dom.minidom.parse('site/sitemap.xml')"`).
