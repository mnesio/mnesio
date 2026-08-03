// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import remarkGfm from "remark-gfm";

// Deploy target. `github.com/mnesio/mnesio` is a *project* repo, so GitHub
// Pages serves it under `/mnesio/` — hence `BASE = "/mnesio"`, and the
// `remarkBaseLinks` plugin below prefixes every internal link so nothing 404s.
// Live at: https://mnesio.github.io/mnesio/
//
// To move to a custom domain (e.g. https://mnesio.dev) served at the root:
// set `BASE = ""`, set `site` to that origin, and add the domain in
// Settings → Pages → Custom domain (which writes a CNAME file).
const BASE = "/mnesio";

/**
 * Prefix root-absolute markdown links with the deploy `base`.
 *
 * Astro rewrites the links *it* generates (assets, the sidebar), but not
 * hand-written `[text](/concepts/foo/)` links in content — those would 404 on a
 * project-path deploy. Rewriting them here keeps the content portable: prose
 * stays written against the site root, and switching between a root and a
 * sub-path deploy is a one-line change to `BASE` rather than an edit to every
 * link. A no-op when `BASE` is empty (root deploy).
 *
 * Walks the mdast directly so this needs no extra dependency.
 */
function remarkBaseLinks() {
  const prefix = BASE.replace(/\/$/, "");
  return (tree) => {
    if (!prefix) return;
    const visit = (node) => {
      if (node.type === "link" && typeof node.url === "string") {
        const u = node.url;
        // Internal, root-absolute, not already prefixed, not protocol-relative.
        if (u.startsWith("/") && !u.startsWith("//") && !u.startsWith(`${prefix}/`)) {
          node.url = prefix + u;
        }
      }
      for (const child of node.children ?? []) visit(child);
    };
    visit(tree);
  };
}

export default defineConfig({
  site: "https://mnesio.github.io",
  base: BASE || "/",
  // GFM (tables, strikethrough, task lists) for both .md and .mdx. The MDX
  // integration inherits this markdown config by default.
  markdown: {
    remarkPlugins: [remarkGfm, remarkBaseLinks],
  },
  integrations: [
    starlight({
      title: "mnesio",
      tagline: "A memory that gets verifiably better.",
      description:
        "A self-improving long-term memory layer for AI agents — append-only, bi-temporal, erasable, and verifiably better over time.",
      logo: { src: "./src/assets/logo.svg", alt: "mnesio" },
      customCss: ["./src/styles/custom.css"],
      components: {
        // Org footer (copyright, license, link columns) on every page.
        Footer: "./src/components/Footer.astro",
      },
      // Social share card (1200×630) for link previews on LinkedIn / X /
      // Slack / Hacker News. Absolute URL because scrapers don't resolve
      // relative paths. Update the origin if the site moves to a domain.
      head: [
        {
          tag: "meta",
          attrs: {
            property: "og:image",
            content: "https://mnesio.github.io/mnesio/brand/og.png",
          },
        },
        {
          tag: "meta",
          attrs: {
            name: "twitter:image",
            content: "https://mnesio.github.io/mnesio/brand/og.png",
          },
        },
        {
          tag: "meta",
          attrs: { name: "twitter:card", content: "summary_large_image" },
        },
      ],
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/mnesio/mnesio",
        },
      ],
      editLink: {
        baseUrl: "https://github.com/mnesio/mnesio/edit/main/website/",
      },
      lastUpdated: true,
      sidebar: [
        {
          label: "Start",
          items: [
            { label: "Why mnesio", slug: "start/why" },
            { label: "Getting started", slug: "start/getting-started" },
            { label: "Quickstart (MCP)", slug: "start/quickstart" },
          ],
        },
        {
          label: "Concepts",
          items: [
            { label: "Architecture", slug: "concepts/architecture" },
            { label: "The seven hard rules", slug: "concepts/hard-rules" },
            { label: "The procedural wedge", slug: "concepts/wedge" },
          ],
        },
        {
          label: "Guides",
          items: [
            { label: "Agent integration (MCP)", slug: "guides/integration" },
            { label: "Code memory", slug: "guides/code-memory" },
            { label: "OpenClaw & Hermes", slug: "guides/openclaw-hermes" },
            { label: "KV cartridges (GPU)", slug: "guides/kv-cartridges" },
          ],
        },
        {
          label: "Reference",
          items: [
            { label: "How mnesio differs", slug: "reference/comparison" },
            { label: "Benchmarks", slug: "reference/benchmarks" },
          ],
        },
      ],
    }),
  ],
});
