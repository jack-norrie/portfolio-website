import config from "@/config/config.json";
import Giscus from "@giscus/react";
import type {
  BooleanString,
  InputPosition,
  Loading,
  Mapping,
} from "@giscus/react";
import React, { useEffect, useState } from "react";

const GiscusComments = ({ className }: { className?: string }) => {
  const { giscus } = config;
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    // Check initial theme
    const isDark = document.documentElement.classList.contains("dark");
    setTheme(isDark ? "dark" : "light");

    // Watch for theme changes
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.attributeName === "class") {
          const isDark = document.documentElement.classList.contains("dark");
          setTheme(isDark ? "dark" : "light");
        }
      });
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    return () => observer.disconnect();
  }, []);

  // Use GitHub Dark themes that match the site theme
  const giscusTheme = theme === "dark" ? "dark" : "light";

  return (
    <div className={className}>
      <Giscus
        repo={giscus.repo as `${string}/${string}`}
        repoId={giscus.repoId}
        category={giscus.category}
        categoryId={giscus.categoryId}
        mapping={giscus.mapping as Mapping}
        strict={giscus.strict as BooleanString}
        reactionsEnabled={giscus.reactionsEnabled as BooleanString}
        emitMetadata={giscus.emitMetadata as BooleanString}
        inputPosition={giscus.inputPosition as InputPosition}
        theme={giscusTheme}
        lang={giscus.lang}
        loading={giscus.loading as Loading}
      />
    </div>
  );
};

export default GiscusComments;
