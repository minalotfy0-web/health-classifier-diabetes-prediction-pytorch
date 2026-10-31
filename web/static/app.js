const form = document.getElementById("form");
const result = document.getElementById("result");
const submit = form.querySelector("button[type=submit]");

function show(className, html) {
  result.className = className;
  result.innerHTML = html;
  result.hidden = false;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function payload() {
  const body = {};
  for (const input of form.elements) {
    if (!input.name) continue;
    input.removeAttribute("aria-invalid");
    // Empty fields are sent as null (lab value not measured) or left out (server default).
    if (input.value.trim() !== "") body[input.name] = Number(input.value);
  }
  return body;
}

function describeError(detail) {
  if (!Array.isArray(detail)) return escapeHtml(String(detail));
  return detail
    .map((item) => {
      const field = item.loc[item.loc.length - 1];
      const input = form.elements[field];
      if (input) input.setAttribute("aria-invalid", "true");
      return `${escapeHtml(String(field))}: ${escapeHtml(item.msg)}`;
    })
    .join("<br>");
}

form.addEventListener("reset", () => {
  result.hidden = true;
  for (const input of form.elements) input.removeAttribute("aria-invalid");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submit.disabled = true;
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload()),
    });
    const data = await response.json();
    if (!response.ok) {
      show("error", `<h2>Check your input</h2><p>${describeError(data.detail)}</p>`);
      return;
    }
    const percent = (data.probability * 100).toFixed(1);
    show(
      data.prediction ? "positive" : "negative",
      `<h2>${escapeHtml(data.label)}</h2>
       <div class="prob">${percent}%</div>
       <div class="meta">Estimated probability of diabetes. Decision threshold
       ${(data.threshold * 100).toFixed(1)}%. Model: ${escapeHtml(data.model)}.
       Research project only, not for clinical use.</div>`,
    );
  } catch (error) {
    show("error", "<h2>Request failed</h2><p>Could not reach the server. Please try again.</p>");
  } finally {
    submit.disabled = false;
  }
});
