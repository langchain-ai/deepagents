async function createPlaywrightCompatRuntime(stagehand) {
  const stats = { calls: {}, misses: {} };
  const record = __name((bucket, method) => {
    stats[bucket][method] = (stats[bucket][method] ?? 0) + 1;
  }, "record");
  const matcher = __name(
    (value, exact = false) =>
      value instanceof RegExp
        ? { kind: "regexp", source: value.source, flags: value.flags }
        : { kind: "string", value: String(value), exact },
    "matcher",
  );
  const unsupported = __name((surface, method) => {
    const name = `${surface}.${String(method)}`;
    record("misses", name);
    throw new Error(`Playwright compatibility facade does not implement ${name}`);
  }, "unsupported");
  const guard = __name(
    (surface, target) =>
      new Proxy(target, {
        get(current, property, receiver) {
          if (property === "then") return void 0;
          if (Reflect.has(current, property)) return Reflect.get(current, property, receiver);
          return (..._args) => unsupported(surface, property);
        },
      }),
    "guard",
  );
  async function executeQueryInPage(input) {
    const normalize = __name((value) => value.replace(/\s+/gu, " ").trim(), "normalize");
    const matches = __name((value, expected) => {
      const normalized = normalize(value);
      if (expected.kind === "regexp") {
        return new RegExp(expected.source, expected.flags).test(normalized);
      }
      const target = normalize(expected.value);
      return expected.exact
        ? normalized === target
        : normalized.toLocaleLowerCase().includes(target.toLocaleLowerCase());
    }, "matches");
    const visible = __name((element) => {
      const style = getComputedStyle(element);
      if (style.visibility === "hidden" || style.display === "none") return false;
      const rect = element.getBoundingClientRect?.();
      return Boolean(rect && rect.width > 0 && rect.height > 0);
    }, "visible");
    const dedupe = __name((elements2) => {
      const seen = new Set();
      return elements2.filter((element) => {
        if (seen.has(element)) return false;
        seen.add(element);
        return true;
      });
    }, "dedupe");
    const queryCssDeep = __name((root, selector) => {
      const direct = [...root.querySelectorAll(selector)];
      const ownShadow =
        root instanceof Element && root.shadowRoot ? queryCssDeep(root.shadowRoot, selector) : [];
      const nested = [...root.querySelectorAll("*")].flatMap((element) =>
        element.shadowRoot ? queryCssDeep(element.shadowRoot, selector) : [],
      );
      return dedupe([...direct, ...ownShadow, ...nested]);
    }, "queryCssDeep");
    const smallestTextMatches = __name(
      (elements2, expected) =>
        elements2.filter(
          (element) =>
            matches(element.textContent ?? "", expected) &&
            ![...element.children].some((child) => matches(child.textContent ?? "", expected)),
        ),
      "smallestTextMatches",
    );
    const queryXPath = __name((root, expression) => {
      const documentNode = root instanceof Document ? root : root.ownerDocument;
      const result = documentNode.evaluate(
        expression,
        root,
        null,
        XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,
        null,
      );
      const elements2 = [];
      for (let index = 0; index < result.snapshotLength; index += 1) {
        const node = result.snapshotItem(index);
        if (node instanceof Element) elements2.push(node);
      }
      return elements2;
    }, "queryXPath");
    const splitSelectorList = __name((selector) => {
      const parts = [];
      let start = 0;
      let depth = 0;
      let quote = "";
      for (let index = 0; index < selector.length; index += 1) {
        const character = selector[index];
        if (quote) {
          if (character === quote && selector[index - 1] !== "\\") quote = "";
          continue;
        }
        if (character === '"' || character === "'") {
          quote = character;
        } else if (character === "(" || character === "[") {
          depth += 1;
        } else if (character === ")" || character === "]") {
          depth = Math.max(0, depth - 1);
        } else if (character === "," && depth === 0) {
          parts.push(selector.slice(start, index));
          start = index + 1;
        }
      }
      parts.push(selector.slice(start));
      return parts;
    }, "splitSelectorList");
    const querySelector = __name((root, rawSelector) => {
      const selector = rawSelector.trim();
      if (selector === "..") {
        return root instanceof Element && root.parentElement ? [root.parentElement] : [];
      }
      if (/^(?:xpath=|\/|\()/u.test(selector)) {
        return queryXPath(root, selector.replace(/^xpath=/u, ""));
      }
      if (/^text=/iu.test(selector)) {
        const text = selector.replace(/^text=/iu, "").replace(/^(?:"(.*)"|'(.*)')$/u, "$1$2");
        const regexp = text.match(/^\/(.*)\/([dgimsuvy]*)$/u);
        return smallestTextMatches(
          queryCssDeep(root, "*"),
          regexp
            ? { kind: "regexp", source: regexp[1] ?? "", flags: regexp[2] ?? "" }
            : { kind: "string", value: text, exact: false },
        );
      }
      return dedupe(
        splitSelectorList(selector).flatMap((rawPart) => {
          let part = rawPart.trim().replace(/^css=/u, "");
          if (/^text=/iu.test(part)) {
            const text = part.replace(/^text=/iu, "").replace(/^(?:"(.*)"|'(.*)')$/u, "$1$2");
            const regexp = text.match(/^\/(.*)\/([dgimsuvy]*)$/u);
            return smallestTextMatches(
              queryCssDeep(root, "*"),
              regexp
                ? { kind: "regexp", source: regexp[1] ?? "", flags: regexp[2] ?? "" }
                : { kind: "string", value: text, exact: false },
            );
          }
          const requireVisible = /:visible\b/u.test(part);
          part = part.replace(/:visible\b/gu, "").replace(/\s*>>>?\s*/gu, " ");
          const pseudo = part.match(/^(.*?):(has-text|text-is|text)\((['"])(.*?)\3\)(.*)$/u);
          let elements2;
          if (pseudo) {
            const anchors = queryCssDeep(root, pseudo[1]?.trim() || "*").filter((element) =>
              matches(element.textContent ?? "", {
                kind: "string",
                value: pseudo[4] ?? "",
                exact: pseudo[2] === "text-is",
              }),
            );
            const suffix = pseudo[5]?.trim();
            elements2 = suffix
              ? dedupe(anchors.flatMap((anchor) => queryCssDeep(anchor, suffix)))
              : anchors;
          } else {
            elements2 = queryCssDeep(root, part || "*");
          }
          return requireVisible ? elements2.filter(visible) : elements2;
        }),
      );
    }, "querySelector");
    const implicitRole = __name((element) => {
      const explicit = element.getAttribute("role")?.trim().split(/\s+/u)[0];
      if (explicit) return explicit;
      const tag = element.tagName.toLocaleLowerCase();
      if (tag === "button") return "button";
      if (tag === "a" && element.hasAttribute("href")) return "link";
      if (tag === "img") return "img";
      if (/^h[1-6]$/u.test(tag)) return "heading";
      if (tag === "textarea") return "textbox";
      if (tag === "select") return element.hasAttribute("multiple") ? "listbox" : "combobox";
      if (tag === "option") return "option";
      if (tag === "ul" || tag === "ol") return "list";
      if (tag === "li") return "listitem";
      if (tag === "table") return "table";
      if (tag === "tr") return "row";
      if (tag === "th") return "columnheader";
      if (tag === "td") return "cell";
      if (tag === "nav") return "navigation";
      if (tag === "main") return "main";
      if (tag === "form") return "form";
      if (tag === "input") {
        const type = (element.getAttribute("type") || "text").toLocaleLowerCase();
        if (["button", "submit", "reset", "image"].includes(type)) return "button";
        if (type === "checkbox") return "checkbox";
        if (type === "radio") return "radio";
        if (type === "range") return "slider";
        if (type === "number") return "spinbutton";
        if (type === "search") return "searchbox";
        if (!["hidden", "file"].includes(type)) return "textbox";
      }
      return void 0;
    }, "implicitRole");
    const labelText = __name((element) => {
      const labelledBy = element.getAttribute("aria-labelledby");
      if (labelledBy) {
        const text = labelledBy
          .split(/\s+/u)
          .map((id) => element.ownerDocument.getElementById(id)?.textContent ?? "")
          .join(" ");
        if (normalize(text)) return text;
      }
      const htmlElement = element;
      if (htmlElement.labels?.length) {
        return [...htmlElement.labels].map((label) => label.textContent ?? "").join(" ");
      }
      return "";
    }, "labelText");
    const accessibleName = __name((element) => {
      const ariaLabel = element.getAttribute("aria-label");
      if (ariaLabel) return ariaLabel;
      const label = labelText(element);
      if (normalize(label)) return label;
      if (element instanceof HTMLImageElement && element.alt) return element.alt;
      if (
        element instanceof HTMLInputElement &&
        ["button", "submit", "reset"].includes(element.type)
      ) {
        return element.value;
      }
      return element.textContent || element.getAttribute("title") || "";
    }, "accessibleName");
    const descendants = __name(
      (roots) => dedupe(roots.flatMap((root) => queryCssDeep(root, "*"))),
      "descendants",
    );
    const resolve = __name((steps, initialRoots = [document]) => {
      let current = [];
      let roots = initialRoots;
      for (const step of steps) {
        if (step.kind === "selector") {
          current = dedupe(roots.flatMap((root) => querySelector(root, step.value)));
        } else if (step.kind === "text") {
          current = smallestTextMatches(descendants(roots), step.matcher);
        } else if (step.kind === "attribute") {
          current = descendants(roots).filter((element) =>
            matches(element.getAttribute(step.name) ?? "", step.matcher),
          );
        } else if (step.kind === "label") {
          current = descendants(roots).filter((element) =>
            matches(labelText(element), step.matcher),
          );
        } else if (step.kind === "role") {
          current = descendants(roots).filter((element) => {
            if (implicitRole(element) !== step.role) return false;
            if (!step.includeHidden && !visible(element)) return false;
            if (step.name && !matches(accessibleName(element), step.name)) return false;
            if (step.checked !== void 0 && element.checked !== step.checked) return false;
            if (step.disabled !== void 0 && element.disabled !== step.disabled) return false;
            if (step.selected !== void 0 && element.selected !== step.selected) return false;
            if (
              step.expanded !== void 0 &&
              element.getAttribute("aria-expanded") !== String(step.expanded)
            )
              return false;
            if (
              step.pressed !== void 0 &&
              element.getAttribute("aria-pressed") !== String(step.pressed)
            )
              return false;
            if (step.level !== void 0 && Number(element.tagName.slice(1)) !== step.level)
              return false;
            return true;
          });
        } else if (step.kind === "filter") {
          current = current.filter((element) => {
            const text = element.textContent ?? "";
            if (step.hasText && !matches(text, step.hasText)) return false;
            if (step.hasNotText && matches(text, step.hasNotText)) return false;
            if (step.visible !== void 0 && visible(element) !== step.visible) return false;
            if (step.has && resolve(step.has, [element]).length === 0) return false;
            if (step.hasNot && resolve(step.hasNot, [element]).length > 0) return false;
            return true;
          });
        } else if (step.kind === "nth") {
          const index = step.index < 0 ? current.length + step.index : step.index;
          current = index >= 0 && index < current.length ? [current[index]] : [];
        }
        roots = current;
      }
      return current;
    }, "resolve");
    if (input.operation === "pageContent") {
      return { count: 1, value: document.documentElement.outerHTML };
    }
    if (input.operation === "pageEvaluateHandle") {
      const fn = (0, eval)(`(${input.functionSource})`);
      const result = await fn(input.argument);
      if (result instanceof Element) {
        const token = input.token;
        result.setAttribute("data-stagehand-pw-compat", token);
        return { count: 1, token, handleKind: "element" };
      }
      return { count: 1, value: result, handleKind: "value" };
    }
    const elements = resolve(input.plan ?? []);
    const first = elements[0];
    if (input.operation === "inspect") {
      return { count: elements.length, visible: first ? visible(first) : false };
    }
    if (input.operation === "untag") {
      document
        .querySelectorAll(`[data-stagehand-pw-compat="${CSS.escape(input.token ?? "")}"]`)
        .forEach((element) => element.removeAttribute("data-stagehand-pw-compat"));
      return { count: 0 };
    }
    if (input.operation === "tag") {
      if (!first) return { count: 0 };
      first.setAttribute("data-stagehand-pw-compat", input.token);
      return { count: elements.length, token: input.token };
    }
    if (input.operation === "tagAll") {
      const prefix = input.token;
      const tokens = elements.map((element, index) => {
        const token = `${prefix}-${index}`;
        element.setAttribute("data-stagehand-pw-compat", token);
        return token;
      });
      return { count: elements.length, values: tokens };
    }
    if (input.operation === "allTextContents") {
      return {
        count: elements.length,
        values: elements.map((element) => element.textContent ?? ""),
      };
    }
    if (input.operation === "allInnerTexts") {
      return { count: elements.length, values: elements.map((element) => element.innerText) };
    }
    if (input.operation === "evaluateAll") {
      const fn = (0, eval)(`(${input.functionSource})`);
      return { count: elements.length, value: await fn(elements, input.argument) };
    }
    if (!first) return { count: 0 };
    if (input.operation === "textContent")
      return { count: elements.length, value: first.textContent };
    if (input.operation === "innerText") return { count: elements.length, value: first.innerText };
    if (input.operation === "innerHTML") return { count: elements.length, value: first.innerHTML };
    if (input.operation === "inputValue") return { count: elements.length, value: first.value };
    if (input.operation === "isChecked")
      return { count: elements.length, value: Boolean(first.checked) };
    if (input.operation === "isDisabled")
      return {
        count: elements.length,
        value:
          first.matches(":disabled") ||
          first.getAttribute("aria-disabled")?.toLocaleLowerCase() === "true",
      };
    if (input.operation === "isEnabled")
      return {
        count: elements.length,
        value:
          !first.matches(":disabled") &&
          first.getAttribute("aria-disabled")?.toLocaleLowerCase() !== "true",
      };
    if (input.operation === "getAttribute")
      return { count: elements.length, value: first.getAttribute(input.attribute) };
    if (input.operation === "boundingBox") {
      const rect = first.getBoundingClientRect?.();
      return {
        count: elements.length,
        value: rect ? { x: rect.x, y: rect.y, width: rect.width, height: rect.height } : null,
      };
    }
    if (input.operation === "focus") {
      first.focus();
      return { count: elements.length };
    }
    if (input.operation === "blur") {
      first.blur();
      return { count: elements.length };
    }
    if (input.operation === "selectText") {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(first);
      selection?.removeAllRanges();
      selection?.addRange(range);
      return { count: elements.length };
    }
    if (input.operation === "domClick") {
      first.click();
      return { count: elements.length };
    }
    if (input.operation === "scrollIntoView") {
      first.scrollIntoView({ block: "center", inline: "center" });
      return { count: elements.length };
    }
    if (input.operation === "evaluate" || input.operation === "elementEvaluateHandle") {
      const fn = (0, eval)(`(${input.functionSource})`);
      const result = await fn(first, input.argument);
      if (input.operation === "elementEvaluateHandle" && result instanceof Element) {
        const token = input.token;
        result.setAttribute("data-stagehand-pw-compat", token);
        return { count: elements.length, token, handleKind: "element" };
      }
      return { count: elements.length, value: result, handleKind: "value" };
    }
    return { count: elements.length };
  }
  __name(executeQueryInPage, "executeQueryInPage");
  const buildQueryEvaluationExpression = __name((query) => {
    const functionSource = JSON.stringify(Function.prototype.toString.call(executeQueryInPage));
    const querySource = JSON.stringify(query);
    return `(async () => {
      const identity = (target) => target;
      for (let index = 0; index <= 32; index += 1) {
        globalThis[index === 0 ? "__name" : "__name" + index] = identity;
      }
      try {
        const execute = (0, eval)("(" + ${functionSource} + ")");
        return await execute(${querySource});
      } catch (error) {
        return {
          count: 0,
          error: {
            name: typeof error?.name === "string" ? error.name : "Error",
            message: typeof error?.message === "string" ? error.message : String(error),
            ...(typeof error?.stack === "string" ? { stack: error.stack } : {}),
          },
        };
      }
    })()`;
  }, "buildQueryEvaluationExpression");
  const rawContext = stagehand.context;
  const pageKey = __name((page) => page.pageId ?? page, "pageKey");
  const compatPages = new Map();
  const closedPages = new Set();
  const contextPageListeners = new Map();
  const screenshotArtifacts = [];
  const encodeBase64 = __name((bytes) => {
    let binary = "";
    const chunkSize = 32768;
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
    }
    return btoa(binary);
  }, "encodeBase64");
  class CompatLocator {
    constructor(plan, state) {
      this.plan = plan;
      this.state = state;
    }
    plan;
    state;
    static {
      __name(this, "CompatLocator");
    }
    derived(step) {
      return locatorProxy(new CompatLocator([...this.plan, step], this.state));
    }
    locator(selector, options = {}) {
      record("calls", "locator.locator");
      const located = this.derived({ kind: "selector", value: selector });
      return Object.keys(options).length > 0 ? located.filter(options) : located;
    }
    getByText(value, options = {}) {
      record("calls", "locator.getByText");
      return this.derived({ kind: "text", matcher: matcher(value, options.exact) });
    }
    getByRole(role, options = {}) {
      record("calls", "locator.getByRole");
      return this.derived({
        kind: "role",
        role,
        ...(options.name === void 0 ? {} : { name: matcher(options.name, options.exact === true) }),
        ...(typeof options.checked === "boolean" ? { checked: options.checked } : {}),
        ...(typeof options.disabled === "boolean" ? { disabled: options.disabled } : {}),
        ...(typeof options.selected === "boolean" ? { selected: options.selected } : {}),
        ...(typeof options.expanded === "boolean" ? { expanded: options.expanded } : {}),
        ...(typeof options.pressed === "boolean" ? { pressed: options.pressed } : {}),
        ...(typeof options.includeHidden === "boolean"
          ? { includeHidden: options.includeHidden }
          : {}),
        ...(typeof options.level === "number" ? { level: options.level } : {}),
      });
    }
    getByLabel(value, options = {}) {
      record("calls", "locator.getByLabel");
      return this.derived({ kind: "label", matcher: matcher(value, options.exact) });
    }
    byAttribute(name, value, exact) {
      return this.derived({ kind: "attribute", name, matcher: matcher(value, exact) });
    }
    getByPlaceholder(value, options = {}) {
      record("calls", "locator.getByPlaceholder");
      return this.byAttribute("placeholder", value, options.exact);
    }
    getByAltText(value, options = {}) {
      record("calls", "locator.getByAltText");
      return this.byAttribute("alt", value, options.exact);
    }
    getByTitle(value, options = {}) {
      record("calls", "locator.getByTitle");
      return this.byAttribute("title", value, options.exact);
    }
    getByTestId(value) {
      record("calls", "locator.getByTestId");
      return this.byAttribute("data-testid", value, true);
    }
    filter(options) {
      record("calls", "locator.filter");
      const has = options.has instanceof CompatLocator ? options.has.plan : void 0;
      const hasNot = options.hasNot instanceof CompatLocator ? options.hasNot.plan : void 0;
      return this.derived({
        kind: "filter",
        ...(options.hasText === void 0 ? {} : { hasText: matcher(options.hasText) }),
        ...(options.hasNotText === void 0 ? {} : { hasNotText: matcher(options.hasNotText) }),
        ...(has ? { has } : {}),
        ...(hasNot ? { hasNot } : {}),
        ...(typeof options.visible === "boolean" ? { visible: options.visible } : {}),
      });
    }
    first() {
      record("calls", "locator.first");
      return this.derived({ kind: "nth", index: 0 });
    }
    last() {
      record("calls", "locator.last");
      return this.derived({ kind: "nth", index: -1 });
    }
    nth(index) {
      record("calls", "locator.nth");
      return this.derived({ kind: "nth", index });
    }
    async count() {
      record("calls", "locator.count");
      return (await this.state.execute(this.plan, "inspect")).count;
    }
    async all() {
      record("calls", "locator.all");
      const prefix = crypto.randomUUID();
      const result = await this.state.execute(this.plan, "tagAll", { token: prefix });
      return result.values.map((token) =>
        locatorProxy(
          new CompatLocator(
            [
              { kind: "selector", value: `[data-stagehand-pw-compat="${token}"]` },
              { kind: "nth", index: 0 },
            ],
            this.state,
          ),
        ),
      );
    }
    async allTextContents() {
      record("calls", "locator.allTextContents");
      return (await this.state.execute(this.plan, "allTextContents")).values;
    }
    async allInnerTexts() {
      record("calls", "locator.allInnerTexts");
      return (await this.state.execute(this.plan, "allInnerTexts")).values;
    }
    async singleValue(operation, extra = {}) {
      const result = await this.state.execute(this.plan, operation, extra);
      if (result.count === 0) throw new Error(`locator.${operation}: no element matched`);
      return result.value;
    }
    async textContent() {
      record("calls", "locator.textContent");
      return await this.singleValue("textContent");
    }
    async innerText() {
      record("calls", "locator.innerText");
      return await this.singleValue("innerText");
    }
    async innerHTML() {
      record("calls", "locator.innerHTML");
      return await this.singleValue("innerHTML");
    }
    async inputValue() {
      record("calls", "locator.inputValue");
      return await this.singleValue("inputValue");
    }
    async getAttribute(name) {
      record("calls", "locator.getAttribute");
      return await this.singleValue("getAttribute", { attribute: name });
    }
    async isVisible() {
      record("calls", "locator.isVisible");
      const result = await this.state.execute(this.plan, "inspect");
      return result.count > 0 && result.visible === true;
    }
    async isChecked() {
      record("calls", "locator.isChecked");
      return await this.singleValue("isChecked");
    }
    async isDisabled() {
      record("calls", "locator.isDisabled");
      return await this.singleValue("isDisabled");
    }
    async isEnabled() {
      record("calls", "locator.isEnabled");
      return await this.singleValue("isEnabled");
    }
    async boundingBox() {
      record("calls", "locator.boundingBox");
      const result = await this.state.execute(this.plan, "boundingBox");
      return result.count === 0 ? null : result.value;
    }
    getBoundingClientRect() {
      record("calls", "locator.getBoundingClientRect");
      return this.boundingBox();
    }
    async evaluate(fn, arg) {
      record("calls", "locator.evaluate");
      return await this.singleValue("evaluate", {
        functionSource: Function.prototype.toString.call(fn),
        ...(arg === void 0 ? {} : { argument: arg }),
      });
    }
    async evaluateAll(fn, arg) {
      record("calls", "locator.evaluateAll");
      return (
        await this.state.execute(this.plan, "evaluateAll", {
          functionSource: Function.prototype.toString.call(fn),
          ...(arg === void 0 ? {} : { argument: arg }),
        })
      ).value;
    }
    async evaluateHandle(fn, arg) {
      record("calls", "locator.evaluateHandle");
      const token = crypto.randomUUID();
      const result = await this.state.execute(this.plan, "elementEvaluateHandle", {
        functionSource: Function.prototype.toString.call(fn),
        ...(arg === void 0 ? {} : { argument: arg }),
        token,
      });
      return jsHandle(result, this.state);
    }
    async focus() {
      record("calls", "locator.focus");
      await this.singleValue("focus");
    }
    async blur() {
      record("calls", "locator.blur");
      await this.singleValue("blur");
    }
    async selectText() {
      record("calls", "locator.selectText");
      await this.singleValue("selectText");
    }
    async waitFor(options = {}) {
      record("calls", "locator.waitFor");
      const state = options.state ?? "visible";
      const timeout = options.timeout ?? 3e4;
      const deadline = Date.now() + timeout;
      do {
        const result = await this.state.execute(this.plan, "inspect");
        const ready =
          state === "attached"
            ? result.count > 0
            : state === "detached"
              ? result.count === 0
              : state === "hidden"
                ? result.count === 0 || result.visible !== true
                : result.count > 0 && result.visible === true;
        if (ready) return;
        await this.state.rawPage.waitForTimeout(50);
      } while (Date.now() < deadline);
      throw new Error(`locator.waitFor: timed out after ${timeout}ms waiting for ${state}`);
    }
    async scrollIntoViewIfNeeded() {
      record("calls", "locator.scrollIntoViewIfNeeded");
      await this.singleValue("scrollIntoView");
    }
    async withTaggedTarget(method, action, options = {}) {
      record("calls", method);
      const timeout = options.timeout ?? 3e4;
      const deadline = Date.now() + timeout;
      let result = { count: 0 };
      let lastActionError;
      while (Date.now() <= deadline) {
        result = await this.state.execute(this.plan, "inspect");
        if (result.count > 1) {
          throw new Error(`${method}: strict mode violation: ${result.count} elements matched`);
        }
        if (result.count === 1) {
          await this.state.execute(this.plan, "scrollIntoView").catch(() => void 0);
          const token = crypto.randomUUID();
          await this.state.execute(this.plan, "tag", { token });
          let actionSucceeded = false;
          try {
            await action(this.state.rawPage.locator(`[data-stagehand-pw-compat="${token}"]`));
            actionSucceeded = true;
          } catch (error) {
            lastActionError = error;
            const message = error instanceof Error ? error.message : String(error);
            const retryable =
              /(?:not found|could not find|no (?:element|node)|detached|not visible|(?:box model|layout object)|execution context|session|target closed|timed? out|timeout)/iu.test(
                message,
              );
            if (!retryable || Date.now() >= deadline) throw error;
          } finally {
            await this.state.execute([], "untag", { token }).catch(() => void 0);
          }
          if (actionSucceeded) {
            await this.state.refreshUrl();
            return;
          }
        }
        if (Date.now() < deadline) await this.state.rawPage.waitForTimeout(50);
      }
      if (lastActionError) throw lastActionError;
      throw new Error(`${method}: no element matched within ${timeout}ms`);
    }
    async click(options = {}) {
      if (options.force === true) {
        record("calls", "locator.click");
        const result = await this.state.execute(this.plan, "inspect");
        if (result.count === 0) throw new Error("locator.click: no element matched");
        if (result.count > 1) {
          throw new Error(`locator.click: strict mode violation: ${result.count} elements matched`);
        }
        await this.state.execute(this.plan, "domClick");
        await this.state.refreshUrl();
        return;
      }
      await this.withTaggedTarget(
        "locator.click",
        (locator) =>
          locator.click({
            ...(typeof options.button === "string" ? { button: options.button } : {}),
            ...(typeof options.clickCount === "number" ? { clickCount: options.clickCount } : {}),
          }),
        options,
      );
    }
    async fill(value, options = {}) {
      await this.withTaggedTarget("locator.fill", (locator) => locator.fill(value), options);
    }
    async type(value, options = {}) {
      await this.withTaggedTarget(
        "locator.type",
        (locator) =>
          locator.type(
            value,
            typeof options.delay === "number" ? { delay: options.delay } : void 0,
          ),
        options,
      );
    }
    pressSequentially(value, options = {}) {
      record("calls", "locator.pressSequentially");
      return this.type(value, options);
    }
    async press(key, options = {}) {
      await this.withTaggedTarget(
        "locator.press",
        async (locator) => {
          await locator.click();
          await this.state.rawPage.keyPress(
            key,
            typeof options.delay === "number" ? { delay: options.delay } : void 0,
          );
        },
        options,
      );
    }
    async hover(options = {}) {
      await this.withTaggedTarget("locator.hover", (locator) => locator.hover(), options);
    }
    async selectOption(values, options = {}) {
      const requested = Array.isArray(values) ? values : [values];
      const indices = requested
        .map((value, position) =>
          typeof value === "object" && value !== null && typeof value.index === "number"
            ? { position, index: value.index }
            : null,
        )
        .filter((value) => value !== null);
      const indexedValues =
        indices.length === 0
          ? []
          : await this.evaluate((element, requestedIndices) => {
              const select = element;
              return requestedIndices.map(({ index }) => select.options[index]?.value ?? null);
            }, indices);
      const normalized = requested.map((value, position) => {
        if (typeof value === "string") return value;
        if (typeof value.value === "string") return value.value;
        if (typeof value.label === "string") return value.label;
        const indexedPosition = indices.findIndex((entry) => entry.position === position);
        const indexedValue = indexedValues[indexedPosition];
        if (typeof indexedValue === "string") return indexedValue;
        throw new Error(`locator.selectOption: option index ${String(value.index)} did not match`);
      });
      let selected = [];
      await this.withTaggedTarget(
        "locator.selectOption",
        async (locator) => {
          selected = await locator.selectOption(Array.isArray(values) ? normalized : normalized[0]);
        },
        options,
      );
      return selected;
    }
    async setInputFiles(files, options = {}) {
      await this.withTaggedTarget(
        "locator.setInputFiles",
        (locator) => locator.setInputFiles(files),
        options,
      );
    }
    async check(options = {}) {
      record("calls", "locator.check");
      if (!(await this.isChecked())) await this.click(options);
    }
    async uncheck(options = {}) {
      record("calls", "locator.uncheck");
      if (await this.isChecked()) await this.click(options);
    }
    async clear(options = {}) {
      record("calls", "locator.clear");
      await this.fill("", options);
    }
    async $(selector) {
      record("calls", "element.$");
      const locator = (await this.locator(selector).all())[0];
      return locator ? markElementHandle(locator, this.state) : null;
    }
    async $$(selector) {
      record("calls", "element.$$");
      return (await this.locator(selector).all()).map((locator) =>
        markElementHandle(locator, this.state),
      );
    }
    async $eval(selector, fn, arg) {
      record("calls", "element.$eval");
      return this.locator(selector).first().evaluate(fn, arg);
    }
    async $$eval(selector, fn, arg) {
      record("calls", "element.$$eval");
      return this.locator(selector).evaluateAll(fn, arg);
    }
  }
  const locatorProxy = __name((locator) => guard("locator", locator), "locatorProxy");
  const handleMetadata = new WeakMap();
  const pageStateMetadata = new WeakMap();
  const markElementHandle = __name((locator, state) => {
    handleMetadata.set(locator, {
      element: locator,
      result: { count: 1, handleKind: "element" },
      state,
    });
    return locator;
  }, "markElementHandle");
  const jsHandle = __name((result, state) => {
    const element =
      result.handleKind === "element" && result.token
        ? locatorProxy(
            new CompatLocator(
              [
                { kind: "selector", value: `[data-stagehand-pw-compat="${result.token}"]` },
                { kind: "nth", index: 0 },
              ],
              state,
            ),
          )
        : null;
    const target = {
      asElement: __name(() => element, "asElement"),
      jsonValue: __name(async () => result.value, "jsonValue"),
      evaluate: __name(
        (fn, arg) =>
          element
            ? element.evaluate(fn, arg)
            : state.rawPage.evaluate(
                (payload) => {
                  const evaluate = (0, eval)(`(${payload.functionSource})`);
                  return evaluate(payload.value, payload.argument);
                },
                {
                  functionSource: Function.prototype.toString.call(fn),
                  ...(result.value === void 0 ? {} : { value: result.value }),
                  ...(arg === void 0 ? {} : { argument: arg }),
                },
              ),
        "evaluate",
      ),
      evaluateHandle: __name(
        (fn, arg) =>
          element?.evaluateHandle(fn, arg) ?? unsupported("jsHandle", "evaluateHandle(value)"),
        "evaluateHandle",
      ),
      dispose: __name(async () => {
        if (result.token) await state.execute([], "untag", { token: result.token });
      }, "dispose"),
      $: __name((selector) => element?.$(selector) ?? null, "$"),
      $$: __name(async (selector) => element?.$$(selector) ?? [], "$$"),
    };
    const handle = guard("jsHandle", target);
    handleMetadata.set(handle, { element, result, state });
    return handle;
  }, "jsHandle");
  let waitForNewPage;
  const createPage = __name(async (page) => {
    const key = pageKey(page);
    const existing = compatPages.get(key);
    if (existing) return existing;
    const state = {
      rawPage: page,
      cachedUrl: await page.url(),
      viewport: await page.evaluate("({ width: innerWidth, height: innerHeight })"),
      closed: false,
      execute: __name(async (plan, operation, extra = {}) => {
        const result = await page.evaluate(
          buildQueryEvaluationExpression({ plan, operation, ...extra }),
        );
        if (result.error) {
          const error = new Error(result.error.message);
          error.name = result.error.name;
          if (result.error.stack) error.stack = result.error.stack;
          throw error;
        }
        return result;
      }, "execute"),
      refreshUrl: __name(async () => {
        state.cachedUrl = await page.url();
        for (const candidate of await rawContext.pages()) await createPage(candidate);
      }, "refreshUrl"),
    };
    const root = __name(() => locatorProxy(new CompatLocator([], state)), "root");
    const requestFetch = __name(async (url, options = {}) => {
      record("calls", "request.fetch");
      if (rawContext.request) return await rawContext.request.fetch(url, options);
      const response = await page.evaluate(
        async (payload) => {
          const result = await fetch(payload.url, payload.options);
          return {
            body: await result.text(),
            headers: Object.fromEntries(result.headers.entries()),
            ok: result.ok,
            status: result.status,
            statusText: result.statusText,
            url: result.url,
          };
        },
        { url, options },
      );
      const value = response;
      return guard("apiResponse", {
        ok: __name(() => value.ok, "ok"),
        status: __name(() => value.status, "status"),
        statusText: __name(() => value.statusText, "statusText"),
        url: __name(() => value.url, "url"),
        headers: __name(() => ({ ...value.headers }), "headers"),
        text: __name(async () => value.body, "text"),
        json: __name(async () => JSON.parse(value.body), "json"),
        body: __name(async () => new TextEncoder().encode(value.body), "body"),
      });
    }, "requestFetch");
    const request = guard("request", {
      fetch: __name((url, options) => requestFetch(url, options), "fetch"),
      get: __name((url, options = {}) => requestFetch(url, { ...options, method: "GET" }), "get"),
      post: __name(
        (url, options = {}) => requestFetch(url, { ...options, method: "POST" }),
        "post",
      ),
    });
    const eventSubscriptions = new Map();
    const routeSubscriptions = [];
    const encodeTextBase64 = __name((value) => {
      const bytes = new TextEncoder().encode(value);
      let binary = "";
      for (const byte of bytes) binary += String.fromCharCode(byte);
      return btoa(binary);
    }, "encodeTextBase64");
    const networkRequests = new Map();
    const requestFromEvent = __name((event) => {
      const params = event.params ?? {};
      const descriptor = params.request ?? {};
      const headers = descriptor.headers ?? {};
      const requestId = String(params.requestId ?? "");
      const request2 = guard("request", {
        url: __name(() => String(descriptor.url ?? ""), "url"),
        method: __name(() => String(descriptor.method ?? "GET"), "method"),
        headers: __name(
          () =>
            Object.fromEntries(
              Object.entries(headers).map(([name, value]) => [name, String(value)]),
            ),
          "headers",
        ),
        postData: __name(
          () => (descriptor.postData === void 0 ? null : String(descriptor.postData)),
          "postData",
        ),
        resourceType: __name(() => String(params.type ?? "other").toLowerCase(), "resourceType"),
        isNavigationRequest: __name(() => params.type === "Document", "isNavigationRequest"),
      });
      if (requestId) networkRequests.set(requestId, request2);
      return request2;
    }, "requestFromEvent");
    const responseFromEvent = __name((event) => {
      const params = event.params ?? {};
      const descriptor = params.response ?? {};
      const headers = descriptor.headers ?? {};
      const request2 = networkRequests.get(String(params.requestId ?? ""));
      const status = Number(descriptor.status ?? 0);
      return guard("response", {
        url: __name(() => String(descriptor.url ?? ""), "url"),
        status: __name(() => status, "status"),
        statusText: __name(() => String(descriptor.statusText ?? ""), "statusText"),
        ok: __name(() => status >= 200 && status <= 299, "ok"),
        headers: __name(
          () =>
            Object.fromEntries(
              Object.entries(headers).map(([name, value]) => [name, String(value)]),
            ),
          "headers",
        ),
        request: __name(() => request2, "request"),
      });
    }, "responseFromEvent");
    const failedRequestFromEvent = __name((event) => {
      const params = event.params ?? {};
      const request2 = networkRequests.get(String(params.requestId ?? ""));
      if (!request2) return requestFromEvent(event);
      return new Proxy(request2, {
        get(target, property, receiver) {
          if (property === "failure") {
            return () => ({ errorText: String(params.errorText ?? "Request failed") });
          }
          return Reflect.get(target, property, receiver);
        },
      });
    }, "failedRequestFromEvent");
    const consoleMessageFromEvent = __name((event) => {
      const params = event.params ?? {};
      const args = Array.isArray(params.args) ? params.args : [];
      const text = args
        .map((entry) => {
          const value = entry ?? {};
          if (value.value !== void 0) return String(value.value);
          if (value.description !== void 0) return String(value.description);
          return String(value.type ?? "");
        })
        .join(" ");
      return guard("consoleMessage", {
        type: __name(() => String(params.type ?? "log"), "type"),
        text: __name(() => text, "text"),
      });
    }, "consoleMessageFromEvent");
    const downloadFromEvent = __name((event) => {
      const params = event.params ?? {};
      const guid = String(params.guid ?? "");
      return guard("download", {
        url: __name(() => String(params.url ?? ""), "url"),
        suggestedFilename: __name(
          () => String(params.suggestedFilename ?? (guid || "download")),
          "suggestedFilename",
        ),
        failure: __name(async () => null, "failure"),
        path: __name(async () => null, "path"),
        cancel: __name(async () => {
          if (guid) await page.sendCDP("Browser.cancelDownload", { guid });
        }, "cancel"),
        saveAs: __name(async () => unsupported("download", "saveAs"), "saveAs"),
      });
    }, "downloadFromEvent");
    const pageErrorFromEvent = __name((event) => {
      const details = event.params?.exceptionDetails ?? {};
      const exception = details.exception ?? {};
      const error = new Error(
        String(exception.description ?? exception.value ?? details.text ?? "Page error"),
      );
      error.name = String(exception.className ?? "Error");
      return error;
    }, "pageErrorFromEvent");
    const subscribeEvent = __name((event, listener, once) => {
      if (
        event !== "download" &&
        event !== "console" &&
        event !== "request" &&
        event !== "response" &&
        event !== "requestfailed" &&
        event !== "pageerror" &&
        event !== "framenavigated"
      ) {
        unsupported("page", `on(${event})`);
      }
      let wrapped;
      wrapped = __name((value) => {
        if (once) {
          const subscriptions2 = eventSubscriptions.get(event);
          const subscription2 = subscriptions2?.get(listener);
          subscriptions2?.delete(listener);
          if (subscriptions2?.size === 0) eventSubscriptions.delete(event);
          void subscription2?.then((active) => active.unsubscribe());
        }
        const facadeValue =
          event === "download"
            ? downloadFromEvent(value)
            : event === "console"
              ? consoleMessageFromEvent(value)
              : event === "request"
                ? requestFromEvent(value)
                : event === "response"
                  ? responseFromEvent(value)
                  : event === "requestfailed"
                    ? failedRequestFromEvent(value)
                    : event === "pageerror"
                      ? pageErrorFromEvent(value)
                      : pageProxy;
        return listener(facadeValue);
      }, "wrapped");
      const primarySubscription =
        event === "pageerror"
          ? page.onCDP("Runtime.exceptionThrown", wrapped)
          : event === "framenavigated"
            ? page.onCDP("Page.frameNavigated", (value) => {
                const frame = value.params?.frame ?? {};
                if (frame.parentId === void 0) return wrapped(value);
              })
            : page.on(event, wrapped);
      const requestSubscription =
        event === "response" || event === "requestfailed"
          ? page.onCDP("Network.requestWillBeSent", requestFromEvent)
          : void 0;
      const subscription = requestSubscription
        ? Promise.all([primarySubscription, requestSubscription]).then(([primary, requests]) => ({
            unsubscribe: __name(async () => {
              await Promise.all([primary.unsubscribe(), requests.unsubscribe()]);
            }, "unsubscribe"),
          }))
        : primarySubscription;
      const subscriptions = eventSubscriptions.get(event) ?? new Map();
      subscriptions.set(listener, subscription);
      eventSubscriptions.set(event, subscriptions);
      return subscription;
    }, "subscribeEvent");
    const waitForResponse = __name(async (predicate, options = {}) => {
      record("calls", "page.waitForResponse");
      const timeout = options.timeout ?? 3e4;
      return await new Promise((resolve, reject) => {
        let settled = false;
        let subscription;
        const finish = __name((result, error) => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          const subscriptions = eventSubscriptions.get("response");
          subscriptions?.delete(listener);
          if (subscriptions?.size === 0) eventSubscriptions.delete("response");
          void subscription.then((active) => active.unsubscribe()).catch(() => void 0);
          if (error) reject(error);
          else resolve(result);
        }, "finish");
        const listener = __name(async (response) => {
          try {
            if (predicate instanceof RegExp) predicate.lastIndex = 0;
            const matched =
              typeof predicate === "function"
                ? await predicate(response)
                : predicate instanceof RegExp
                  ? predicate.test(response.url())
                  : response.url() === predicate;
            if (matched) finish(response);
          } catch (error) {
            finish(void 0, error instanceof Error ? error : new Error(String(error)));
          }
        }, "listener");
        const timer = setTimeout(
          () => finish(void 0, new Error(`page.waitForResponse: timed out after ${timeout}ms`)),
          timeout,
        );
        subscription = subscribeEvent("response", listener, false);
        void subscription.catch((error) =>
          finish(void 0, error instanceof Error ? error : new Error(String(error))),
        );
      });
    }, "waitForResponse");
    const waitForPageEvent = __name(async (event, options = {}) => {
      if (event === "popup") return await waitForNewPage(options);
      if (event !== "download") return unsupported("page", `waitForEvent(${event})`);
      await page.sendCDP("Page.enable").catch(() => void 0);
      await page.sendCDP("Page.setDownloadBehavior", { behavior: "allow" }).catch(() => void 0);
      const timeout = options.timeout ?? 3e4;
      return await new Promise((resolve, reject) => {
        let subscription;
        const listener = __name((download) => {
          clearTimeout(timer);
          resolve(download);
        }, "listener");
        const timer = setTimeout(() => {
          const subscriptions = eventSubscriptions.get(event);
          subscriptions?.delete(listener);
          if (subscriptions?.size === 0) eventSubscriptions.delete(event);
          void subscription.then((active) => active.unsubscribe());
          reject(new Error(`page.waitForEvent(download): timed out after ${timeout}ms`));
        }, timeout);
        subscription = subscribeEvent("download", listener, true);
      });
    }, "waitForPageEvent");
    const pageObject = {
      goto: __name(async (url, options) => {
        record("calls", "page.goto");
        const response = await page.goto(url, options);
        await state.refreshUrl();
        return response;
      }, "goto"),
      reload: __name(async (options) => {
        record("calls", "page.reload");
        const response = await page.reload(options);
        await state.refreshUrl();
        return response;
      }, "reload"),
      goBack: __name(async (options) => {
        record("calls", "page.goBack");
        const response = await page.goBack(options);
        await state.refreshUrl();
        return response;
      }, "goBack"),
      goForward: __name(async (options) => {
        record("calls", "page.goForward");
        const response = await page.goForward(options);
        await state.refreshUrl();
        return response;
      }, "goForward"),
      url: __name(() => {
        record("calls", "page.url");
        return state.cachedUrl;
      }, "url"),
      title: __name(() => {
        record("calls", "page.title");
        return page.title();
      }, "title"),
      content: __name(async () => {
        record("calls", "page.content");
        return (await state.execute([], "pageContent")).value;
      }, "content"),
      evaluate: __name((fn, arg) => {
        record("calls", "page.evaluate");
        const metadata = arg && typeof arg === "object" ? handleMetadata.get(arg) : void 0;
        if (metadata?.element && typeof fn === "function") {
          return metadata.element.evaluate(fn);
        }
        return page.evaluate(fn, arg);
      }, "evaluate"),
      evaluateHandle: __name(async (fn, arg) => {
        record("calls", "page.evaluateHandle");
        const metadata = arg && typeof arg === "object" ? handleMetadata.get(arg) : void 0;
        if (metadata?.element) return metadata.element.evaluateHandle(fn);
        const token = crypto.randomUUID();
        return jsHandle(
          await state.execute([], "pageEvaluateHandle", {
            functionSource: Function.prototype.toString.call(fn),
            ...(arg === void 0 ? {} : { argument: arg }),
            token,
          }),
          state,
        );
      }, "evaluateHandle"),
      locator: __name((selector, options = {}) => {
        record("calls", "page.locator");
        const located = locatorProxy(
          new CompatLocator([{ kind: "selector", value: selector }], state),
        );
        return Object.keys(options).length > 0 ? located.filter(options) : located;
      }, "locator"),
      getByText: __name((value, options) => {
        record("calls", "page.getByText");
        return root().getByText(value, options);
      }, "getByText"),
      getByRole: __name((role, options) => {
        record("calls", "page.getByRole");
        return root().getByRole(role, options);
      }, "getByRole"),
      getByLabel: __name((value, options) => {
        record("calls", "page.getByLabel");
        return root().getByLabel(value, options);
      }, "getByLabel"),
      getByPlaceholder: __name(
        (value, options) => root().getByPlaceholder(value, options),
        "getByPlaceholder",
      ),
      getByAltText: __name((value, options) => root().getByAltText(value, options), "getByAltText"),
      getByTitle: __name((value, options) => root().getByTitle(value, options), "getByTitle"),
      getByTestId: __name((value) => root().getByTestId(value), "getByTestId"),
      $: __name(async (selector) => {
        record("calls", "page.$");
        const locator = (await pageObject.locator(selector).all())[0];
        return locator ? markElementHandle(locator, state) : null;
      }, "$"),
      $$: __name(async (selector) => {
        record("calls", "page.$$");
        return (await pageObject.locator(selector).all()).map((locator) =>
          markElementHandle(locator, state),
        );
      }, "$$"),
      $x: __name(async (expression) => {
        record("calls", "page.$x");
        return (await pageObject.locator(`xpath=${expression}`).all()).map((locator) =>
          markElementHandle(locator, state),
        );
      }, "$x"),
      $eval: __name((selector, fn, arg) => {
        record("calls", "page.$eval");
        return pageObject.locator(selector).first().evaluate(fn, arg);
      }, "$eval"),
      $$eval: __name((selector, fn, arg) => {
        record("calls", "page.$$eval");
        return pageObject.locator(selector).evaluateAll(fn, arg);
      }, "$$eval"),
      click: __name((selector, options) => pageObject.locator(selector).click(options), "click"),
      hover: __name((selector, options) => pageObject.locator(selector).hover(options), "hover"),
      fill: __name(
        (selector, value, options) => pageObject.locator(selector).fill(value, options),
        "fill",
      ),
      type: __name(
        (selector, value, options) => pageObject.locator(selector).type(value, options),
        "type",
      ),
      press: __name(
        (selector, key2, options) => pageObject.locator(selector).press(key2, options),
        "press",
      ),
      focus: __name((selector) => pageObject.locator(selector).focus(), "focus"),
      check: __name((selector, options) => pageObject.locator(selector).check(options), "check"),
      uncheck: __name(
        (selector, options) => pageObject.locator(selector).uncheck(options),
        "uncheck",
      ),
      selectOption: __name(
        (selector, values, options) => pageObject.locator(selector).selectOption(values, options),
        "selectOption",
      ),
      getAttribute: __name(
        (selector, name) => pageObject.locator(selector).getAttribute(name),
        "getAttribute",
      ),
      textContent: __name((selector) => pageObject.locator(selector).textContent(), "textContent"),
      innerText: __name((selector) => pageObject.locator(selector).innerText(), "innerText"),
      inputValue: __name((selector) => pageObject.locator(selector).inputValue(), "inputValue"),
      isVisible: __name((selector) => pageObject.locator(selector).isVisible(), "isVisible"),
      isChecked: __name((selector) => pageObject.locator(selector).isChecked(), "isChecked"),
      isDisabled: __name((selector) => pageObject.locator(selector).isDisabled(), "isDisabled"),
      isEnabled: __name((selector) => pageObject.locator(selector).isEnabled(), "isEnabled"),
      screenshot: __name(async (options = {}) => {
        record("calls", "page.screenshot");
        const { path: requestedPath, ...supported } = options;
        const inferredType =
          supported.type === void 0 &&
          typeof requestedPath === "string" &&
          /\.jpe?g$/iu.test(requestedPath)
            ? "jpeg"
            : void 0;
        const bytes = await page.screenshot({
          ...supported,
          ...(inferredType ? { type: inferredType } : {}),
        });
        if (typeof requestedPath === "string") {
          screenshotArtifacts.push({ path: requestedPath, base64: encodeBase64(bytes) });
        }
        return bytes;
      }, "screenshot"),
      waitForTimeout: __name(async (ms) => {
        record("calls", "page.waitForTimeout");
        await page.waitForTimeout(ms);
        for (const candidate of await rawContext.pages()) await createPage(candidate);
      }, "waitForTimeout"),
      waitForLoadState: __name((state2 = "load", options = {}) => {
        record("calls", "page.waitForLoadState");
        return page.waitForLoadState(state2, options.timeout);
      }, "waitForLoadState"),
      waitForSelector: __name(async (selector, options) => {
        record("calls", "page.waitForSelector");
        return (await page.waitForSelector(selector, options))
          ? pageObject.locator(selector).first()
          : null;
      }, "waitForSelector"),
      waitForNavigation: __name(async (options = {}) => {
        record("calls", "page.waitForNavigation");
        const before = state.cachedUrl;
        const deadline = Date.now() + (options.timeout ?? 3e4);
        while (Date.now() < deadline) {
          const next = await page.url();
          if (next !== before) {
            state.cachedUrl = next;
            if (options.waitUntil) await page.waitForLoadState(options.waitUntil, options.timeout);
            return null;
          }
          await page.waitForTimeout(50);
        }
        throw new Error("page.waitForNavigation: timed out");
      }, "waitForNavigation"),
      waitForResponse,
      waitForEvent: waitForPageEvent,
      setViewportSize: __name(async (size) => {
        record("calls", "page.setViewportSize");
        state.viewport = { width: size.width, height: size.height };
        await page.setViewportSize(size.width, size.height);
      }, "setViewportSize"),
      viewportSize: __name(() => {
        record("calls", "page.viewportSize");
        return state.viewport;
      }, "viewportSize"),
      setExtraHTTPHeaders: __name((headers) => {
        record("calls", "page.setExtraHTTPHeaders");
        return page.setExtraHTTPHeaders(headers);
      }, "setExtraHTTPHeaders"),
      addInitScript: __name((script, arg) => {
        record("calls", "page.addInitScript");
        return page.addInitScript(script, arg);
      }, "addInitScript"),
      close: __name(async () => {
        record("calls", "page.close");
        await page.close();
        state.closed = true;
        closedPages.add(key);
      }, "close"),
      isClosed: __name(() => state.closed, "isClosed"),
      name: __name(() => "", "name"),
      context: __name(() => context, "context"),
      bringToFront: __name(async () => {
        record("calls", "page.bringToFront");
        await rawContext.setActivePage(page);
      }, "bringToFront"),
      frames: __name(() => {
        record("calls", "page.frames");
        return [pageProxy];
      }, "frames"),
      on: __name((event, listener) => {
        record("calls", "page.on");
        subscribeEvent(event, listener, false);
        return pageProxy;
      }, "on"),
      once: __name((event, listener) => {
        record("calls", "page.once");
        subscribeEvent(event, listener, true);
        return pageProxy;
      }, "once"),
      off: __name((event, listener) => {
        record("calls", "page.off");
        const subscriptions = eventSubscriptions.get(event);
        const subscription = subscriptions?.get(listener);
        subscriptions?.delete(listener);
        if (subscriptions?.size === 0) eventSubscriptions.delete(event);
        void subscription?.then((active) => active.unsubscribe());
        return pageProxy;
      }, "off"),
      removeListener: __name((event, listener) => {
        record("calls", "page.removeListener");
        const subscriptions = eventSubscriptions.get(event);
        const subscription = subscriptions?.get(listener);
        subscriptions?.delete(listener);
        if (subscriptions?.size === 0) eventSubscriptions.delete(event);
        void subscription?.then((active) => active.unsubscribe());
        return pageProxy;
      }, "removeListener"),
      route: __name(async (pattern, handler) => {
        record("calls", "page.route");
        const urlPattern = typeof pattern === "string" ? pattern : "*";
        await page.sendCDP("Fetch.enable", { patterns: [{ urlPattern }] });
        const subscription = page.onCDP("Fetch.requestPaused", async (event) => {
          const params = event.params ?? {};
          const requestId = String(params.requestId ?? "");
          const cdpRequest = params.request ?? {};
          let handled = false;
          const requestHeaders = cdpRequest.headers ?? {};
          const route = {
            request: __name(
              () => ({
                url: __name(() => String(cdpRequest.url ?? ""), "url"),
                method: __name(() => String(cdpRequest.method ?? "GET"), "method"),
                headers: __name(() => ({ ...requestHeaders }), "headers"),
                postData: __name(
                  () => (cdpRequest.postData === void 0 ? null : String(cdpRequest.postData)),
                  "postData",
                ),
              }),
              "request",
            ),
            continue: __name(async (options = {}) => {
              handled = true;
              const headers = options.headers;
              await page.sendCDP("Fetch.continueRequest", {
                requestId,
                ...(headers
                  ? { headers: Object.entries(headers).map(([name, value]) => ({ name, value })) }
                  : {}),
                ...(typeof options.url === "string" ? { url: options.url } : {}),
                ...(typeof options.method === "string" ? { method: options.method } : {}),
                ...(typeof options.postData === "string"
                  ? { postData: encodeTextBase64(options.postData) }
                  : {}),
              });
            }, "continue"),
            abort: __name(async (errorReason = "Failed") => {
              handled = true;
              await page.sendCDP("Fetch.failRequest", { requestId, errorReason });
            }, "abort"),
            fulfill: __name(async (options = {}) => {
              handled = true;
              const headers = { ...(options.headers ?? {}) };
              const responseBody =
                options.json === void 0
                  ? typeof options.body === "string"
                    ? options.body
                    : void 0
                  : JSON.stringify(options.json);
              if (options.json !== void 0 && !("content-type" in headers)) {
                headers["content-type"] = "application/json";
              }
              const body = responseBody === void 0 ? void 0 : encodeTextBase64(responseBody);
              await page.sendCDP("Fetch.fulfillRequest", {
                requestId,
                responseCode: typeof options.status === "number" ? options.status : 200,
                responseHeaders: Object.entries(headers).map(([name, value]) => ({ name, value })),
                ...(body === void 0 ? {} : { body }),
              });
            }, "fulfill"),
          };
          await handler(route);
          if (!handled) await route.continue();
        });
        await subscription;
        routeSubscriptions.push({ pattern, handler, subscription });
      }, "route"),
      unroute: __name(async (pattern, handler) => {
        const matches = routeSubscriptions.filter(
          (entry) => entry.pattern === pattern && (handler === void 0 || entry.handler === handler),
        );
        for (const entry of matches) {
          routeSubscriptions.splice(routeSubscriptions.indexOf(entry), 1);
          await (await entry.subscription).unsubscribe();
        }
        if (routeSubscriptions.length === 0) await page.sendCDP("Fetch.disable");
      }, "unroute"),
      request,
      accessibility: { snapshot: __name((options) => page.snapshot(options), "snapshot") },
      keyboard: {
        type: __name((text, options) => page.type(text, options), "type"),
        insertText: __name((text) => page.type(text), "insertText"),
        press: __name((key2, options) => page.keyPress(key2, options), "press"),
      },
      mouse: (() => {
        let x = 0;
        let y = 0;
        let down = false;
        return {
          click: __name((nextX, nextY, options) => page.click(nextX, nextY, options), "click"),
          move: __name(async (nextX, nextY) => {
            x = nextX;
            y = nextY;
            await page.hover(x, y);
          }, "move"),
          wheel: __name((deltaX, deltaY) => page.scroll(x, y, deltaX, deltaY), "wheel"),
          down: __name(async () => {
            down = true;
          }, "down"),
          up: __name(async () => {
            if (down) await page.click(x, y);
            down = false;
          }, "up"),
        };
      })(),
    };
    const pageProxy = guard("page", pageObject);
    pageStateMetadata.set(pageProxy, state);
    compatPages.set(key, pageProxy);
    closedPages.delete(key);
    for (const [listener, once] of contextPageListeners) {
      listener(pageProxy);
      if (once) contextPageListeners.delete(listener);
    }
    return pageProxy;
  }, "createPage");
  const initialPage = await createPage(stagehand.page);
  const contextRequest = initialPage.request;
  const initialRawPages = await rawContext.pages();
  for (const page of initialRawPages) await createPage(page);
  waitForNewPage = __name(async (options = {}) => {
    const existing = new Set(compatPages.keys());
    const timeout = options.timeout ?? 3e4;
    const deadline = Date.now() + timeout;
    do {
      for (const candidate of await rawContext.pages()) {
        const key = pageKey(candidate);
        if (!existing.has(key)) return await createPage(candidate);
      }
      await new Promise((resolve) => setTimeout(resolve, 50));
    } while (Date.now() < deadline);
    throw new Error(`context.waitForEvent(page): timed out after ${timeout}ms`);
  }, "waitForNewPage");
  const contextObject = {
    pages: __name(() => {
      record("calls", "context.pages");
      return [...compatPages.entries()]
        .filter(([key]) => !closedPages.has(key))
        .map(([, page]) => page);
    }, "pages"),
    newPage: __name(async () => {
      record("calls", "context.newPage");
      return await createPage(await rawContext.newPage());
    }, "newPage"),
    cookies: __name((urls) => {
      record("calls", "context.cookies");
      return rawContext.cookies(urls);
    }, "cookies"),
    addCookies: __name((cookies) => rawContext.addCookies(cookies), "addCookies"),
    clearCookies: __name((options) => {
      record("calls", "context.clearCookies");
      return rawContext.clearCookies(options);
    }, "clearCookies"),
    setExtraHTTPHeaders: __name(
      (headers) => rawContext.setExtraHTTPHeaders(headers),
      "setExtraHTTPHeaders",
    ),
    addInitScript: __name((script, arg) => {
      record("calls", "context.addInitScript");
      return rawContext.addInitScript(script, arg);
    }, "addInitScript"),
    waitForEvent: __name((event, options) => {
      if (event !== "page") return unsupported("context", `waitForEvent(${event})`);
      return waitForNewPage(options);
    }, "waitForEvent"),
    on: __name((event, listener) => {
      record("calls", "context.on");
      if (event !== "page") return unsupported("context", `on(${event})`);
      contextPageListeners.set(listener, false);
      return context;
    }, "on"),
    once: __name((event, listener) => {
      record("calls", "context.once");
      if (event !== "page") return unsupported("context", `once(${event})`);
      contextPageListeners.set(listener, true);
      return context;
    }, "once"),
    off: __name((event, listener) => {
      record("calls", "context.off");
      if (event !== "page") return unsupported("context", `off(${event})`);
      contextPageListeners.delete(listener);
      return context;
    }, "off"),
    removeListener: __name((event, listener) => {
      record("calls", "context.removeListener");
      if (event !== "page") return unsupported("context", `removeListener(${event})`);
      contextPageListeners.delete(listener);
      return context;
    }, "removeListener"),
    newCDPSession: __name(async (compatPage) => {
      record("calls", "context.newCDPSession");
      const target = pageStateMetadata.get(compatPage);
      if (!target) throw new Error("context.newCDPSession: page does not belong to this context");
      const subscriptions = [];
      return guard("cdpSession", {
        send: __name((method, params) => target.rawPage.sendCDP(method, params), "send"),
        on: __name((method, listener) => {
          subscriptions.push(target.rawPage.onCDP(method, (event) => listener(event.params ?? {})));
        }, "on"),
        detach: __name(async () => {
          await Promise.all(
            subscriptions.map(async (subscription) => (await subscription).unsubscribe()),
          );
        }, "detach"),
      });
    }, "newCDPSession"),
    request: contextRequest,
  };
  const context = guard("context", contextObject);
  const browser = guard("browser", {
    contexts: __name(() => [context], "contexts"),
    isConnected: __name(() => true, "isConnected"),
    newContext: __name(async () => {
      record("calls", "browser.newContext");
      return context;
    }, "newContext"),
    newPage: __name(() => contextObject.newPage(), "newPage"),
  });
  return {
    page: initialPage,
    context,
    browser,
    telemetry: __name(
      () => ({ calls: { ...stats.calls }, misses: { ...stats.misses } }),
      "telemetry",
    ),
    artifacts: __name(() => screenshotArtifacts.map((artifact) => ({ ...artifact })), "artifacts"),
  };
}
