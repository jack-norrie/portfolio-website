import config from "@/config/config.json";
import Giscus from "@giscus/react";
import React from "react";

const GiscusComments = ({ className }: { className?: string }) => {
  const { giscus } = config;

  if (!giscus.enable) {
    return null;
  }

  return (
    <div className={className}>
      <Giscus
        repo={giscus.repo}
        repoId={giscus.repoId}
        category={giscus.category}
        categoryId={giscus.categoryId}
        mapping={giscus.mapping}
        strict={giscus.strict}
        reactionsEnabled={giscus.reactionsEnabled}
        emitMetadata={giscus.emitMetadata}
        inputPosition={giscus.inputPosition}
        theme={giscus.theme}
        lang={giscus.lang}
        loading={giscus.loading}
      />
    </div>
  );
};

export default GiscusComments;
