# Deep research request

How can we balance the need for durability and warranty guarantees for EV batteries with evolving regulatory requirements, especially ACC regulations (ACC II), while staying on track with our production timelines through 2035?

## Who you are

You are working on behalf of:

- **Name:** Sofia Rodriguez
- **Role:** Product Compliance Engineer
- **Department:** Product Development
- **Seniority:** Junior
- **Email:** sofia.rodriguez@elexionautomotive.com
- **Responsibilities:** Conducts compliance testing and certification for Elexion Automotive's electric vehicles, collaborates with design teams to ensure regulatory requirements are met, and assists in the development of compliance documentation.

## Company context

- **Company:** Elexion Automotive
- **Industry:** Automotive -- Electric Vehicles
- **Headquarters:** Austin, Texas, USA
- **Size:** Enterprise
- **Employees:** 1900
- **Annual revenue:** $350M - $500M
- **Market position:** Elexion is an emerging force in the North American EV market, offering affordable and high-performance electric vehicles with regulatory-first design and localized production.
- **Description:** Elexion Automotive designs and manufactures premium and mid-range electric vehicles tailored for the North American market. With a focus on sustainability, user-centric technology, and cross-border compliance, Elexion aims to accelerate the transition to zero-emission transportation.
- **Key products and services:** Electric vehicles; Plug-in hybrid electric vehicles; Home charging stations
- **Target markets:** Mid-income families in North America; Provincial and municipal governments
- **Compliance certifications:** EPA Greenhouse Gas Emissions Compliance; Motor Vehicle Safety Oversight Program; CARB Zero-Emission Vehicle (ZEV) Program; ISO 26262 – Road Vehicles Functional Safety

Today's date is 2025-08-27.

## Where to research

Your company's systems are running and reachable over the network. **Nothing is on this
machine's filesystem** — you have to query the applications. Each one has its own login,
with its password already exported in the environment:

- **email** (log in as `sofia.rodriguez`, password in `$DRBENCH_EMAIL_PASS`) — the mailbox for this login, over IMAP on `drbench:1143` (Python's `imaplib` works well). The same mail is browsable at `http://drbench:8085`.
- **nextcloud** (log in as `sofia.rodriguez`, password in `$DRBENCH_NEXTCLOUD_PASS`) — the company cloud drive. HTTP Basic auth over WebDAV. List one level with `curl -u "$DRBENCH_NEXTCLOUD_sofia.rodriguez:$DRBENCH_NEXTCLOUD_PASS" -X PROPFIND -H 'Depth: 1' -H 'Host: localhost' http://drbench:8081/remote.php/dav/files/sofia.rodriguez/`, then repeat on any directory it returns (they end in `/`) to walk deeper, and GET any file path to download it. Both extra headers matter: this server answers `400 Bad Request` to `Depth: infinity`, and it only trusts the Host `localhost`, so a request addressed to `drbench:8081` is rejected without the `Host` override.

Documents are PDF, DOCX, XLSX, PPTX, and JSONL mail exports, so anything you download is
binary. Convert it with `extract-text <path>`. Not every document is relevant — the
systems hold unrelated material alongside what you need.

You also have internet access and a `web_search` tool. Some of what this question needs
is public information that exists nowhere in the company's systems, so research the open
web as well.

If a service seems unreachable, check `http://drbench:8099/health` — it returns 200 only when every
application is up.

## What to deliver

Write a research report to `/app/report.md` as Markdown.

- Ground every factual claim in a source, cited inline with a bracketed number
  (`[1]`, `[2]`, ...).
- End the report with a `## References` section listing each number against its source,
  in one of these exact forms. Scoring resolves each citation back to the source it
  names, and a citation it cannot resolve counts as unsupported no matter how accurate
  the claim is:
  - **a document** — its file name, e.g. `food-safety-compliance.pdf`
  - **a web page** — its full URL
  - **an email** — `RoundCube-<sender address>-<a recipient address>-<Subject>`, e.g.
    `RoundCube-david.lee@example.com-emily.patel@example.com-Re: Q2 Compliance Update`.
    Use the sender's **email address**, not their display name, and copy the subject
    line **exactly** as it appears in the mailbox — both are matched character for
    character.
  - **a chat message** — `MatterMost-<channel>-<team>-<user>`
- Report only what your sources support. Uncited assertions, and claims you cannot trace
  back to a document, email, chat message, or web page, do not count in your favour.
- Cover the question thoroughly: the report is scored on how many of the findings a
  domain expert would consider essential you actually surface, and on whether you kept
  the irrelevant material out.
