# Deep research request

How can Elexion Automotive increase customer trust through after-sales support while balancing the need for exceptional customer care with efficient and cost-effective service?

## Who you are

You are working on behalf of:

- **Name:** Ryan Thompson
- **Role:** Customer Service Team Lead
- **Department:** Customer Experience
- **Seniority:** Mid
- **Email:** ryan.thompson@elexionautomotive.com
- **Responsibilities:** Lead a team of customer service representatives, monitor performance metrics, and implement strategies to enhance customer experience and satisfaction.

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
machine's filesystem** — you have to query the applications. Each one has its own login:

- **email** (log in as `ryan.thompson` / `my_drbench_pwd`) — the mailbox for this login, over IMAP on `drbench:1143` (Python's `imaplib` works well). The same mail is browsable at `http://drbench:8085`.
- **file_system** (log in as `ryan.thompson` / `my_drbench_pwd`) — local and shared drives, exposed through a file browser at `http://drbench:8090`.
- **mattermost** (log in as `ryan.thompson` / `my_drbench_pwd`) — team chat. `POST http://drbench:8082/api/v4/users/login` with a JSON body of `login_id` and `password` returns a session token in the `Token` response header; send it back as `Authorization: Bearer <token>`.
- **nextcloud** (log in as `ryan.thompson` / `my_drbench_pwd`) — the company cloud drive. HTTP Basic auth over WebDAV. List one level with `curl -u ryan.thompson:my_drbench_pwd -X PROPFIND -H 'Depth: 1' -H 'Host: localhost' http://drbench:8081/remote.php/dav/files/ryan.thompson/`, then repeat on any directory it returns (they end in `/`) to walk deeper, and GET any file path to download it. Both extra headers matter: this server answers `400 Bad Request` to `Depth: infinity`, and it only trusts the Host `localhost`, so a request addressed to `drbench:8081` is rejected without the `Host` override.

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
- End the report with a `## References` section listing each number against the
  document's file name or the URL it came from. Use the file name, not the number, in
  that list.
- Report only what your sources support. Uncited assertions, and claims you cannot trace
  back to a document or web page, do not count in your favour.
- Cover the question thoroughly: the report is scored on how many of the findings a
  domain expert would consider essential you actually surface, and on whether you kept
  the irrelevant material out.
