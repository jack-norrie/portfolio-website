import mdx from "@astrojs/mdx";
import react from "@astrojs/react";
import sitemap from "@astrojs/sitemap";
import tailwindcss from "@tailwindcss/vite";
import AutoImport from "astro-auto-import";
import { defineConfig } from "astro/config";
import remarkCollapse from "remark-collapse";
import remarkToc from "remark-toc";
import remarkMath from "remark-math";
import remarkObsidianCallout from "remark-obsidian-callout";
import rehypeMathjax from "rehype-mathjax";
import sharp from "sharp";
import { visit } from "unist-util-visit";
import config from "./src/config/config.json";

const remarkPublicAssets = () => (tree) => {
  visit(tree, ["image", "link"], (node) => {
    if (!node.url || typeof node.url !== "string") {
      return;
    }

    if (node.url.startsWith("/public/")) {
      node.url = node.url.replace(/^\/public\//, "/");
    } else if (node.url.startsWith("public/")) {
      node.url = node.url.replace(/^public\//, "/");
    }
  });
};

const remarkObsidianEmbeds = () => (tree) => {
  visit(tree, "paragraph", (node, index, parent) => {
    if (!parent || typeof index !== "number") {
      return;
    }

    if (
      !node.children ||
      node.children.some((child) => child.type !== "text")
    ) {
      return;
    }

    const text = node.children
      .map((child) => (typeof child.value === "string" ? child.value : ""))
      .join("");

    const embedRegex = /!\[\[(.+?)\]\]/g;
    if (!embedRegex.test(text)) {
      return;
    }

    embedRegex.lastIndex = 0;
    const newChildren = [];
    let lastIndex = 0;
    let match = embedRegex.exec(text);

    while (match) {
      const [raw, url] = match;
      const matchIndex = match.index;

      const before = text.slice(lastIndex, matchIndex);
      if (before) {
        newChildren.push({ type: "text", value: before });
      }

      if (!/\.excalidraw$/i.test(url)) {
        const filename = url.split("/").pop() ?? url;
        const alt = filename.replace(/\.[^/.]+$/, "");
        newChildren.push({ type: "image", url, alt, title: null });
      }

      lastIndex = matchIndex + raw.length;
      match = embedRegex.exec(text);
    }

    const after = text.slice(lastIndex);
    if (after) {
      newChildren.push({ type: "text", value: after });
    }

    const hasNonWhitespace = newChildren.some((child) => {
      if (child.type !== "text") {
        return true;
      }
      return typeof child.value === "string" && child.value.trim() !== "";
    });

    if (!hasNonWhitespace) {
      parent.children.splice(index, 1);
      return [index, 0];
    }

    node.children = newChildren;
  });
};

// https://astro.build/config
export default defineConfig({
  site: config.site.base_url ? config.site.base_url : "http://examplesite.com",
  base: config.site.base_path ? config.site.base_path : "/",
  trailingSlash: config.site.trailing_slash ? "always" : "never",
  image: { service: sharp() },
  vite: { plugins: [tailwindcss()] },
  integrations: [
    react(),
    sitemap(),
    AutoImport({
      imports: [
        "@/shortcodes/Button",
        "@/shortcodes/Accordion",
        "@/shortcodes/Notice",
        "@/shortcodes/Video",
        "@/shortcodes/Youtube",
        "@/shortcodes/Tabs",
        "@/shortcodes/Tab",
      ],
    }),
    mdx(),
  ],
  markdown: {
    remarkPlugins: [
      remarkToc,
      [remarkCollapse, { test: "Table of contents" }],
      remarkMath,
      remarkObsidianCallout,
      remarkObsidianEmbeds,
      remarkPublicAssets,
    ],
    rehypePlugins: [rehypeMathjax],
    shikiConfig: {
      themes: {
        light: "github-light",
        dark: "github-dark-default",
      },
      wrap: true,
    },
    extendDefaultPlugins: true,
  },
});
