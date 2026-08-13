/* Composer behaviour, message rendering, and the agent/task boards.
 *
 * The boards are rendered from PLACEHOLDER_PLAN below -- nothing here calls a
 * planner and no agent runs. Wiring this to real decomposition means replacing
 * that constant with the server's response; the rendering code already works
 * off whatever shape it is given. */

(() => {
  "use strict";

  const root = document.documentElement;
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const send = document.getElementById("send");
  const thread = document.getElementById("thread");
  const threadScroll = document.getElementById("threadScroll");

  /* Stand-in for a decomposed plan. Same shape the planner produces: agents,
   * each owning one or more tasks. Deliberately fixed -- it does not vary with
   * the query, and the board says so on screen so it is not mistaken for real
   * output. */
  const PLACEHOLDER_PLAN = {
    agents: [
      {
        id: "researcher",
        role: "Gathers source material and pulls out the facts that matter.",
        tasks: [
          {
            id: "collect_sources",
            description: "Find primary sources covering each option.",
          },
          {
            id: "extract_findings",
            description: "Pull the comparable facts out of each source.",
          },
        ],
      },
      {
        id: "analyst",
        role: "Turns raw findings into a defensible comparison.",
        tasks: [
          {
            id: "build_comparison",
            description: "Score each option against the agreed criteria.",
          },
        ],
      },
      {
        id: "writer",
        role: "Produces the finished write-up.",
        tasks: [
          {
            id: "draft_report",
            description: "Write the comparison as a short report.",
          },
          {
            id: "review_draft",
            description: "Check the draft against the original brief.",
          },
        ],
      },
    ],
  };

  /** Grow a textarea with its content, then let it scroll once capped. */
  function autoGrow(el) {
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }

  function hasContent() {
    return input.value.trim().length > 0;
  }

  function syncSendButton() {
    send.disabled = !hasContent();
  }

  function scrollToEnd() {
    threadScroll.scrollTop = threadScroll.scrollHeight;
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
    scrollToEnd();
  }

  /* ---- boards ---------------------------------------------------------- */

  function renderTask(agent, task) {
    const item = document.createElement("li");
    item.className = "task";

    const id = document.createElement("span");
    id.className = "task-id";
    id.textContent = task.id;

    const text = document.createElement("textarea");
    text.className = "task-text";
    text.rows = 1;
    // .value, not innerHTML -- a description is text even if it contains markup.
    text.value = task.description;
    text.setAttribute(
      "aria-label",
      `Description of task ${task.id}, owned by agent ${agent.id}`
    );

    text.addEventListener("input", () => {
      // Write the edit straight back into the plan object. The DOM is a view;
      // this keeps the model the single source of truth so that whatever
      // consumes the plan later reads edits, not the original text.
      task.description = text.value;
      autoGrow(text);
    });

    item.append(id, text);
    return item;
  }

  function renderAgent(agent) {
    const card = document.createElement("section");
    card.className = "agent";
    card.setAttribute("aria-label", `Agent ${agent.id}`);

    const name = document.createElement("h2");
    name.className = "agent-name";
    name.textContent = agent.id;

    const role = document.createElement("p");
    role.className = "agent-role";
    role.textContent = agent.role;

    const count = document.createElement("span");
    count.className = "agent-count";
    count.textContent = `${agent.tasks.length} ${
      agent.tasks.length === 1 ? "task" : "tasks"
    }`;

    const tasks = document.createElement("ul");
    tasks.className = "tasks";
    for (const task of agent.tasks) tasks.appendChild(renderTask(agent, task));

    card.append(name, role, count, tasks);
    return card;
  }

  function appendBoard(plan) {
    const item = document.createElement("li");
    item.className = "board-item";

    const board = document.createElement("div");
    board.className = "board";
    // The model this board renders, kept on the element so it can be read back
    // without a global. Edits mutate it in place.
    board._plan = plan;

    const taskTotal = plan.agents.reduce((n, a) => n + a.tasks.length, 0);

    const note = document.createElement("p");
    note.className = "board-note";
    note.textContent =
      `${plan.agents.length} agents · ${taskTotal} tasks · ` +
      "placeholder — not generated from your query yet";

    const grid = document.createElement("div");
    grid.className = "agents";
    for (const agent of plan.agents) grid.appendChild(renderAgent(agent));

    board.append(note, grid);
    item.appendChild(board);
    thread.appendChild(item);

    // Sizing needs layout, so it has to happen after the board is in the DOM.
    for (const el of grid.querySelectorAll(".task-text")) autoGrow(el);
    scrollToEnd();
  }

  /* ---- submit ---------------------------------------------------------- */

  function submit() {
    const query = input.value.trim();
    if (!query) return;

    // Flip to the thread layout before the first message so the composer
    // animates into place once rather than twice.
    root.dataset.state = "thread";
    appendMessage("user", query);
    // Cloned per board so editing one never reaches back into the constant or
    // into a board rendered earlier.
    appendBoard(structuredClone(PLACEHOLDER_PLAN));

    input.value = "";
    autoGrow(input);
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
    autoGrow(input);
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

  autoGrow(input);
  syncSendButton();
  input.focus();
})();
