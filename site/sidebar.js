// Sidebar collapse to icons only; remembered per browser (a convenience, so storage errors are ignored).
(function () {
  var KEY = "runtracker_sidebar_collapsed";
  var root = document.documentElement;

  function stored() {
    try { return localStorage.getItem(KEY) === "1"; } catch (e) { return false; }
  }

  function apply(collapsed) {
    root.classList.toggle("sb-collapsed", collapsed);
    var btn = document.querySelector(".shell > .sidebar .sidebar-toggle");
    if (btn) {
      var label = collapsed ? "Expand sidebar" : "Collapse sidebar";
      btn.setAttribute("aria-expanded", String(!collapsed));
      btn.setAttribute("aria-label", label);
      btn.title = label;
    }
    // Icon-only links get a hover tooltip with their name.
    document.querySelectorAll(".shell > .sidebar .side-nav a").forEach(function (a) {
      var name = a.querySelector(".nav-label");
      if (collapsed && name) a.title = name.textContent; else a.removeAttribute("title");
    });
  }

  apply(stored()); // loaded in <head>: the class is set before the first paint, so no jump

  document.addEventListener("DOMContentLoaded", function () {
    apply(stored());
    var btn = document.querySelector(".shell > .sidebar .sidebar-toggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var collapsed = !root.classList.contains("sb-collapsed");
      try { localStorage.setItem(KEY, collapsed ? "1" : "0"); } catch (e) {}
      apply(collapsed);
      // Charts and the ontology viewer size themselves on resize; let them re-fit after the width change.
      setTimeout(function () { window.dispatchEvent(new Event("resize")); }, 200);
    });
  });
})();
