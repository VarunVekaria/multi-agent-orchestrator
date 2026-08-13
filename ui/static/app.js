/* Composer behaviour and message rendering.
 *
 * Submitting shows the query and stops. The POST is fire-and-forget on purpose:
 * the server has nothing to say yet, so the UI must not sit waiting on it or
 * imply a reply is coming. */

(() => {
  "use strict";

  const root = document.documentElement;
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const send = document.getElementById("send");
  const thread = document.getElementById("thread");
  const threadScroll = document.getElementById("threadScroll");

  /** Grow with the content, then scroll once the cap is hit. */
  function resize() {
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight}px`;
  }

  function hasContent() {
    return input.value.trim().length > 0;
  }

  function syncSendButton() {
    send.disabled = !hasContent();
  }

  function appendMessage(role, text) {
    const item = document.createElement("li");
    item.className = `message ${role}`;

    const body = document.createElement("div");
    body.className = "body";
    // textContent, never innerHTML: a query containing markup is text, not DOM.
    body.textContent = text;

    item.appendChild(body);
    thread.appendChild(item);
    threadScroll.scrollTop = threadScroll.scrollHeight;
  }

  function submit() {
    const query = input.value.trim();
    if (!query) return;

    // Flip to the thread layout before the first message so the composer
    // animates into place once rather than twice.
    root.dataset.state = "thread";
    appendMessage("user", query);

    input.value = "";
    resize();
    syncSendButton();
    input.focus();

    // Acknowledged and logged server-side; nothing is done with the response.
    fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    }).catch((error) => {
      // The seam isn't load-bearing yet, so a failure here must not disrupt
      // what the user sees. Surface it where a developer will look.
      console.error("query submission failed:", error);
    });
  }

  input.addEventListener("input", () => {
    resize();
    syncSendButton();
  });

  input.addEventListener("keydown", (event) => {
    // Enter sends; Shift+Enter is a newline. isComposing guards IME input,
    // where Enter commits a candidate rather than ending the message.
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submit();
    }
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });

  resize();
  syncSendButton();
  input.focus();
})();
