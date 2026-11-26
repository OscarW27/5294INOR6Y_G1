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

  // Log out on the non-dashboard pages: same keys as index.html's logout, then back to the login screen.
  var SESSION_KEY = "runtracker_session", RUNNER_MODE_KEY = "runtracker_runner_mode";
  function hasSession() {
    try { return !!localStorage.getItem(SESSION_KEY); } catch (e) { return false; }
  }
  root.classList.toggle("no-session", !hasSession()); // hides [data-logout] for visitors who are not logged in

  // Role subtitle + Beginner/Expert switch on pages other than the dashboard (index.html runs its own).
  var RUNNER_ROLES = { "Beginner runner": "Beginner", "Advanced runner": "Expert" };
  function session() {
    try { return JSON.parse(localStorage.getItem(SESSION_KEY)); } catch (e) { return null; }
  }
  function runnerMode(s) {
    var m = null;
    try { m = localStorage.getItem(RUNNER_MODE_KEY); } catch (e) {}
    return m === "Beginner" || m === "Expert" ? m : RUNNER_ROLES[s.role] || "Beginner";
  }
  function setModeButtons(sw, mode) {
    sw.querySelectorAll("button[data-mode]").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.mode === mode)); });
  }
  function initRoleAndMode() {
    var s = session();
    var label = document.querySelector(".shell > .sidebar [data-role-label]");
    if (label) label.textContent = s && s.role ? (RUNNER_ROLES[s.role] ? "Runner" : s.role) : "";
    var runner = !!(s && RUNNER_ROLES[s.role]);
    // Runner-only links (Run history); index.html toggles its own in showApp.
    if (!document.getElementById("main-nav")) {
      document.querySelectorAll(".shell > .sidebar [data-runner-only]").forEach(function (a) { a.hidden = !runner; });
    }
    var sw = document.querySelector(".shell > .sidebar [data-mode-switch]");
    if (!sw) return;
    sw.hidden = !runner;
    if (!runner) return;
    setModeButtons(sw, runnerMode(s));
    sw.addEventListener("click", function (e) {
      var b = e.target.closest("button[data-mode]");
      if (!b) return;
      try { localStorage.setItem(RUNNER_MODE_KEY, b.dataset.mode); } catch (err) {}
      setModeButtons(sw, b.dataset.mode);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    apply(stored());
    initRoleAndMode();
    document.querySelectorAll("[data-logout]").forEach(function (a) {
      a.addEventListener("click", function () {
        try { localStorage.removeItem(SESSION_KEY); localStorage.removeItem(RUNNER_MODE_KEY); } catch (e) {}
        // The link's href (index.html) then opens the login screen.
      });
    });
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
