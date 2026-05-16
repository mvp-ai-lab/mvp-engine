import type { MouseEvent } from "react";
import { useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

type PageEntry = {
  left: string;
  right: string;
};

type PageContent = {
  left: string;
  right: string;
};

const emptyPage: PageContent = {
  left: "",
  right: "",
};

const chapterTitles = ["I. Overview", "II. Setup", "III. Engine", "III. Build", "IV. Skills", "V. Recipe", "VI. Ask"];

function assetUrl(path: string) {
  return `${import.meta.env.BASE_URL}${path.replace(/^\//, "")}`;
}

function MarkdownPage({ markdown }: { markdown: string }) {
  const copyCodeBlock = async (event: MouseEvent<HTMLButtonElement>) => {
    const pre = event.currentTarget.nextElementSibling;
    if (!pre?.textContent) {
      return;
    }

    await navigator.clipboard.writeText(pre.textContent);
  };

  return (
    <ReactMarkdown
      components={{
        code: ({ children, className }) =>
          className ? (
            <code className="font-mono text-[inherit]">{children}</code>
          ) : (
            <code className="rounded border border-[rgba(87,54,24,0.18)] bg-[rgba(255,238,178,0.18)] px-[0.22em] py-[0.04em] font-mono text-[0.74em] text-[rgba(74,47,24,0.8)] max-[760px]:border-[#f1d795]/20 max-[760px]:bg-[#f8df9c]/12 max-[760px]:text-[#f7dfaa]">
              {children}
            </code>
          ),
        h1: ({ children }) => (
          <h1 className="mb-[1.05cqw] text-[3.05cqw] font-bold leading-[1.08] text-[rgba(65,40,19,0.98)] max-[760px]:mb-5 max-[760px]:text-[2.5rem] max-[760px]:text-[#ffe8b0]">
            {children}
          </h1>
        ),
        h2: ({ children }) => (
          <h2 className="mb-[0.55cqw] mt-[1.05cqw] text-[2cqw] font-bold not-italic leading-[1.2] text-[rgba(77,49,24,0.9)] max-[760px]:mb-3 max-[760px]:mt-7 max-[760px]:text-[1.75rem] max-[760px]:text-[#f5d99a]">
            {children}
          </h2>
        ),
        img: ({ alt, src }) => {
          const imageSrc = src ?? "";
          const resolvedImageSrc = assetUrl(imageSrc);
          const mobileAspectRatio = imageSrc.includes("04-fig") ? "1362 / 1003" : "1672 / 626";

          return (
            <>
              <img
                className="mx-auto mt-[1.1cqw] block max-h-[16cqw] w-[86%] object-contain opacity-80 mix-blend-multiply max-[760px]:hidden"
                src={resolvedImageSrc}
                alt={alt ?? ""}
              />
              <span
                className="mx-auto mt-5 hidden w-full bg-[#ead7ad] opacity-78 max-[760px]:block"
                role={alt ? "img" : undefined}
                aria-label={alt ?? undefined}
                style={{
                  aspectRatio: mobileAspectRatio,
                  maskImage: `url(${resolvedImageSrc})`,
                  maskPosition: "center",
                  maskRepeat: "no-repeat",
                  maskSize: "contain",
                  WebkitMaskImage: `url(${resolvedImageSrc})`,
                  WebkitMaskPosition: "center",
                  WebkitMaskRepeat: "no-repeat",
                  WebkitMaskSize: "contain",
                }}
              />
            </>
          );
        },
        li: ({ children }) => (
          <li className="mt-[0.32cqw] pl-[0.1em] text-[1.18cqw] leading-[1.45] text-[rgba(94,61,32,0.86)] marker:text-[0.8em] marker:text-[rgba(96,61,30,0.72)] max-[760px]:mt-2 max-[760px]:text-[1.05rem] max-[760px]:leading-7 max-[760px]:text-[#ead7ad] marker:max-[760px]:text-[#f5d99a]/70">
            {children}
          </li>
        ),
        ol: ({ children }) => <ol className="mt-[0.82cqw] list-decimal pl-[1.45em] marker:[font-variant-numeric:tabular-nums]">{children}</ol>,
        p: ({ children }) => (
          <p className="mt-[0.72cqw] text-justify text-[1.34cqw] leading-[1.55] text-[rgba(90,58,31,0.9)] max-[760px]:mt-4 max-[760px]:text-left max-[760px]:text-[1.08rem] max-[760px]:leading-8 max-[760px]:text-[#ead7ad]">
            {children}
          </p>
        ),
        pre: ({ children }) => (
          <div className="relative mt-[0.82cqw]">
            <button
              className="absolute right-[0.45cqw] top-[0.35cqw] z-[1] rounded border border-[rgba(87,54,24,0.22)] bg-[rgba(255,238,178,0.28)] px-[0.42cqw] py-[0.12cqw] font-mono text-[0.76cqw] leading-[1.2] text-[rgba(74,47,24,0.72)] hover:bg-[rgba(255,238,178,0.42)] hover:text-[rgba(62,38,18,0.9)] max-[760px]:right-2 max-[760px]:top-2 max-[760px]:px-2 max-[760px]:py-1 max-[760px]:text-xs max-[760px]:text-[#f4dda6]"
              type="button"
              onClick={copyCodeBlock}
            >
              Copy
            </button>
            <pre className="m-0 overflow-hidden rounded-[5px] border border-[rgba(87,54,24,0.24)] border-l-[3px] border-l-[rgba(87,54,24,0.36)] bg-[linear-gradient(rgba(104,70,34,0.13),rgba(75,46,21,0.1)),rgba(255,238,178,0.16)] p-[0.65cqw_0.75cqw] font-mono text-[0.86cqw] leading-[1.45] text-[rgba(74,47,24,0.78)] shadow-[inset_0_0_12px_rgba(74,44,18,0.08)] [white-space:pre-wrap] max-[760px]:overflow-x-auto max-[760px]:border-[#f1d795]/20 max-[760px]:bg-[#090604]/70 max-[760px]:p-4 max-[760px]:text-sm max-[760px]:text-[#f3deb0]">
              {children}
            </pre>
          </div>
        ),
        ul: ({ children }) => <ul className="mt-[0.82cqw] list-disc pl-[1.35em]">{children}</ul>,
      }}
    >
      {markdown}
    </ReactMarkdown>
  );
}

export default function App() {
  const [pages, setPages] = useState<PageContent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showLoadingMask, setShowLoadingMask] = useState(true);
  const [pageIndex, setPageIndex] = useState(0);
  const [turnDirection, setTurnDirection] = useState<"next" | "prev" | null>(null);

  useEffect(() => {
    let isMounted = true;

    const loadPages = async () => {
      const manifestResponse = await fetch(assetUrl("/tutorial-pages/pages.json"));
      const manifest = (await manifestResponse.json()) as PageEntry[];
      const loadedPages = await Promise.all(
        manifest.map(async (page) => {
          const [leftResponse, rightResponse] = await Promise.all([fetch(assetUrl(page.left)), fetch(assetUrl(page.right))]);
          return {
            left: await leftResponse.text(),
            right: await rightResponse.text(),
          };
        }),
      );

      if (isMounted) {
        setPages(loadedPages);
        setIsLoading(false);
      }
    };

    void loadPages();

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (isLoading) {
      setShowLoadingMask(true);
      return;
    }

    const maskTimer = window.setTimeout(() => {
      setShowLoadingMask(false);
    }, 520);

    return () => window.clearTimeout(maskTimer);
  }, [isLoading]);

  const turnPage = useCallback(
    (direction: "next" | "prev") => {
      if (turnDirection !== null) {
        return;
      }

      const targetIndex = direction === "next" ? pageIndex + 1 : pageIndex - 1;
      if (targetIndex < 0 || targetIndex >= pages.length) {
        return;
      }

      setTurnDirection(direction);
      window.setTimeout(() => {
        setPageIndex(targetIndex);
      }, 260);
      window.setTimeout(() => {
        setTurnDirection(null);
      }, 640);
    },
    [pageIndex, pages.length, turnDirection],
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "ArrowRight") {
        turnPage("next");
      }

      if (event.key === "ArrowLeft") {
        turnPage("prev");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [turnPage]);

  const currentPage = pages[pageIndex] ?? emptyPage;
  const totalPages = pages.length;
  const readingProgress = totalPages <= 1 ? 0 : (pageIndex / (totalPages - 1)) * 100;

  const jumpToPage = (targetIndex: number) => {
    if (targetIndex === pageIndex || targetIndex < 0 || targetIndex >= totalPages || turnDirection !== null) {
      return;
    }

    setTurnDirection(targetIndex > pageIndex ? "next" : "prev");
    window.setTimeout(() => {
      setPageIndex(targetIndex);
    }, 260);
    window.setTimeout(() => {
      setTurnDirection(null);
    }, 640);
  };

  return (
    <main className="relative min-h-screen overflow-hidden bg-black font-sans text-white max-[760px]:overflow-x-hidden max-[760px]:overflow-y-auto">
      <img
        className="pointer-events-none fixed inset-0 z-0 h-screen w-screen select-none object-cover object-center opacity-[0.8]"
        src={assetUrl("/background.png")}
        alt=""
        aria-hidden="true"
      />
      <section
        className="relative z-[1] flex min-h-screen items-center justify-center px-4 py-6 max-[760px]:block max-[760px]:px-5 max-[760px]:py-10"
        aria-label="MVP-Engine interface"
      >
        <div className="relative aspect-[1448/1086] w-[min(116vw,1680px,132vh)] [container-type:inline-size] max-[760px]:aspect-auto max-[760px]:w-full">
          <nav
            className="absolute left-1/2 top-[-2.2%] z-40 w-[66%] -translate-x-1/2 font-book text-[#f3dfb0] drop-shadow-[0_2px_5px_rgba(0,0,0,0.85)] max-[760px]:hidden"
            aria-label="Chapter navigation"
          >
            <div className="relative mx-auto h-[2.55cqw] min-h-7">
              <div className="absolute left-[4.25cqw] right-[4.25cqw] top-[0.46cqw] h-[0.14cqw] min-h-px rounded-full bg-[#2a190c]/75 shadow-[0_0_0_1px_rgba(255,225,150,0.18)]" />
              <div
                className="absolute left-[4.25cqw] top-[0.46cqw] h-[0.14cqw] min-h-px rounded-full bg-[#ffd36d] shadow-[0_0_10px_rgba(255,211,109,0.72)] transition-[width] duration-300"
                style={{ width: `calc((100% - 8.5cqw) * ${readingProgress / 100})` }}
              />
              <div className="relative flex justify-between">
                {pages.map((_, index) => {
                  const isCurrent = index === pageIndex;
                  const isRead = index <= pageIndex;
                  return (
                    <button
                      className="group flex w-[8.5cqw] min-w-10 flex-col items-center gap-[0.28cqw] text-center"
                      type="button"
                      key={`${chapterTitles[index] ?? "Chapter"}-${index}`}
                      aria-label={`Go to chapter ${index + 1}`}
                      aria-current={isCurrent ? "step" : undefined}
                      onClick={() => jumpToPage(index)}
                    >
                      <span
                        className={`h-[0.86cqw] min-h-2.5 w-[0.86cqw] min-w-2.5 rounded-full border transition ${
                          isCurrent
                            ? "border-[#fff1ba] bg-[#ffe08c] shadow-[0_0_12px_rgba(255,221,134,0.85)]"
                            : isRead
                              ? "border-[#ffe08c]/70 bg-[#d9a64d]"
                              : "border-[#d8b168]/55 bg-[#21170e]/70 group-hover:border-[#ffe08c]/80"
                        }`}
                      />
                      <span
                        className={`text-[0.82cqw] leading-none max-[760px]:text-[1.25cqw] ${
                          isCurrent ? "text-[#fff0bd]" : "text-[#5b4730]/52"
                        }`}
                      >
                        {chapterTitles[index] ?? `Ch ${index + 1}`}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </nav>
          <img
            className="pointer-events-none absolute inset-0 z-10 h-full w-full scale-x-[1.08] select-none object-contain [filter:drop-shadow(0_20px_24px_rgba(0,0,0,0.62))_drop-shadow(0_0_6px_rgba(255,225,128,0.22))_drop-shadow(0_0_13px_rgba(105,235,255,0.15))_drop-shadow(0_0_24px_rgba(172,118,255,0.13))] max-[760px]:hidden"
            src={assetUrl("/main-panel-cutout-v2.png")}
            alt=""
            aria-hidden="true"
          />
          <div
            className={`absolute left-1/2 top-1/2 z-[26] grid h-[68%] w-[82%] -translate-x-1/2 -translate-y-[47%] scale-x-[1.08] grid-cols-2 gap-[14%] py-[2%] font-book text-[#5a3a1f] opacity-90 mix-blend-multiply [text-shadow:0_0_0.35px_rgba(72,43,18,0.42)] max-[760px]:hidden ${
              turnDirection === null ? "" : "animate-page-fade"
            }`}
          >
            <article className="relative z-10 min-w-0 overflow-hidden pr-[3%] max-[760px]:overflow-visible max-[760px]:pr-0">
              <MarkdownPage markdown={currentPage.left} />
            </article>
            <article className="relative z-10 min-w-0 overflow-hidden pl-[3%] max-[760px]:overflow-visible max-[760px]:pl-0">
              <MarkdownPage markdown={currentPage.right} />
            </article>
          </div>
          <div className="hidden max-[760px]:relative max-[760px]:z-10 max-[760px]:block max-[760px]:pb-28 max-[760px]:font-book max-[760px]:text-[#ead7ad] max-[760px]:[text-shadow:0_2px_12px_rgba(0,0,0,0.95)]">
            {pages.map((page, index) => (
              <section className="py-8 first:pt-0" key={`${chapterTitles[index] ?? "Chapter"}-${index}`}>
                {index > 0 ? <div className="mb-10 h-px w-full bg-[#f2d99b]/24 shadow-[0_0_10px_rgba(242,217,155,0.18)]" /> : null}
                <p className="mb-5 text-sm uppercase tracking-[0.18em] text-[#f2d99b]/46">
                  {chapterTitles[index] ?? `Chapter ${index + 1}`}
                </p>
                <article className="min-w-0">
                  <MarkdownPage markdown={page.left} />
                </article>
                <article className="mt-10 min-w-0">
                  <MarkdownPage markdown={page.right} />
                </article>
              </section>
            ))}
          </div>
          <button
            className="absolute left-[9%] top-[18%] z-[24] h-[64%] w-[35%] bg-transparent max-[760px]:hidden"
            type="button"
            aria-label="Previous page"
            onClick={() => turnPage("prev")}
          />
          <button
            className="absolute right-[9%] top-[18%] z-[24] h-[64%] w-[35%] bg-transparent max-[760px]:hidden"
            type="button"
            aria-label="Next page"
            onClick={() => turnPage("next")}
          />
          <div className="pointer-events-none absolute bottom-[11.5%] left-1/2 z-[19] -translate-x-1/2 scale-x-[1.08] font-book text-[1cqw] text-[#6b4725]/65 mix-blend-multiply max-[760px]:hidden">
            {totalPages > 0 ? `${pageIndex + 1} / ${totalPages}` : ""}
          </div>
          <img
            className="pointer-events-none absolute bottom-[-3%] left-[-6.5%] z-20 w-[28%] scale-x-[1.08] select-none object-contain object-left-bottom [filter:drop-shadow(0_18px_18px_rgba(0,0,0,0.78))_drop-shadow(0_0_14px_rgba(0,0,0,0.58))] max-[760px]:hidden"
            src={assetUrl("/foreground_lb-cutout.png")}
            alt=""
            aria-hidden="true"
          />
          <div className="pointer-events-none absolute bottom-[27.5%] left-[-2.55%] z-30 h-[13%] w-[9%] translate-x-[-50%] translate-y-1/2 scale-x-[1.08] max-[760px]:hidden">
            <span className="absolute left-1/2 top-[58%] h-[64%] w-[76%] -translate-x-1/2 -translate-y-1/2 animate-candle-glow rounded-full bg-[radial-gradient(circle,rgba(255,243,171,0.95)_0_10%,rgba(255,166,46,0.48)_30%,rgba(255,119,0,0.18)_50%,transparent_72%)] mix-blend-screen blur-[8px]" />
            <span className="absolute bottom-[44%] left-1/2 ml-[-11%] h-[37%] w-[17%] animate-candle-smoke rounded-full bg-[radial-gradient(ellipse_at_center,rgba(235,231,210,0.58),rgba(235,231,210,0))] opacity-0 blur-md" />
            <span className="absolute bottom-[44%] left-1/2 ml-[1%] h-[37%] w-[17%] animate-candle-smoke rounded-full bg-[radial-gradient(ellipse_at_center,rgba(235,231,210,0.5),rgba(235,231,210,0))] opacity-0 blur-md [animation-delay:1.15s]" />
            <span className="absolute bottom-[44%] left-1/2 ml-[11%] h-[37%] w-[17%] animate-candle-smoke rounded-full bg-[radial-gradient(ellipse_at_center,rgba(235,231,210,0.44),rgba(235,231,210,0))] opacity-0 blur-md [animation-delay:2.3s]" />
          </div>
          <div className="pointer-events-none absolute bottom-[-2.7%] right-[-4.3%] z-20 w-[16%] scale-x-[1.08] select-none max-[760px]:fixed max-[760px]:bottom-3 max-[760px]:right-3 max-[760px]:z-20 max-[760px]:w-[18vw] max-[760px]:max-w-20 max-[760px]:scale-x-100">
            <img
              className="w-full object-contain object-right-bottom [filter:drop-shadow(0_18px_18px_rgba(0,0,0,0.78))_drop-shadow(0_0_14px_rgba(0,0,0,0.58))]"
              src={assetUrl("/foreground_rb-cutout.png")}
              alt=""
              aria-hidden="true"
            />
            <span className="absolute left-[90.4%] top-[52.3%] h-[11%] w-[14%] -translate-x-1/2 -translate-y-1/2 animate-staff-glow rounded-full bg-[radial-gradient(circle,rgba(244,184,255,0.86)_0_8%,rgba(193,76,255,0.42)_30%,rgba(110,62,255,0.18)_54%,transparent_76%)] mix-blend-screen blur-[5px]" />
            <span className="absolute left-[90.4%] top-[52.3%] h-[3.7%] w-[4.7%] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle,rgba(255,239,255,0.95),rgba(204,87,255,0.64)_48%,transparent_72%)] mix-blend-screen blur-[1px]" />
            <span className="absolute left-[87.8%] top-[49.9%] h-[1.3%] w-[1.7%] animate-staff-glow rounded-full bg-[rgba(242,166,255,0.72)] mix-blend-screen blur-[1px] [animation-delay:0.45s]" />
            <span className="absolute left-[93.4%] top-[49.1%] h-[1%] w-[1.3%] animate-staff-glow rounded-full bg-[rgba(164,116,255,0.66)] mix-blend-screen blur-[1px] [animation-delay:1.05s]" />
          </div>
        </div>
      </section>
      {showLoadingMask ? (
        <div
          className={`fixed inset-0 z-[100] flex items-center justify-center bg-[#050302] text-center font-book text-[#f6dfad] transition-opacity duration-500 ${
            isLoading ? "opacity-100" : "pointer-events-none opacity-0"
          }`}
          aria-live="polite"
          aria-busy="true"
        >
          <div className="flex flex-col items-center gap-4">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-[#7d5b2c] border-t-[#ffe29a] shadow-[0_0_16px_rgba(255,212,121,0.32)]" />
            <p className="text-[clamp(1rem,1.6vw,1.4rem)] tracking-[0.08em] text-[#d8bc82]">Loading tutorial</p>
          </div>
        </div>
      ) : null}
    </main>
  );
}
