import type { MouseEvent } from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";

type PageEntry = {
  left: string;
  right: string;
};

type PageContent = {
  left: string;
  right: string;
};

type TurnDirection = "next" | "prev";
type PageSide = "left" | "right";
type PageTextureRect = {
  bottom: number;
  left: number;
  right: number;
  top: number;
};

type ArticlePlacement = {
  containerCssWidth: number;
  cssHeight: number;
  cssWidth: number;
  height: number;
  left: number;
  top: number;
  width: number;
};

type PageSnapshot = {
  canvas: HTMLCanvasElement;
  cssHeight: number;
  cssWidth: number;
};

type PageTurn = {
  backMarkdown: string;
  backSnapshot?: Promise<PageSnapshot>;
  direction: TurnDirection;
  frontMarkdown: string;
  frontSnapshot?: Promise<PageSnapshot>;
  fromIndex: number;
  id: number;
  toIndex: number;
};

const emptyPage: PageContent = {
  left: "",
  right: "",
};

const chapterTitles = ["I. Overview", "II. Setup", "III. Engine", "III. Build", "IV. Skills", "V. Recipe", "VI. Ask"];
const pageTurnDurationMs = 1920;
const pageTurnCoveredSwapMs = Math.round(pageTurnDurationMs * 0.75);
const textCanvasBleedCssPx = 12;
const pageMeshColumns = 72;
const pageMeshRows = 48;

const pageTurnRects: Record<TurnDirection, { height: number; left: number; top: number; width: number }> = {
  next: { height: 0.875, left: 0.5, top: 0.01825, width: 0.4375 },
  prev: { height: 0.875, left: 0.0625, top: 0.01825, width: 0.4375 },
};

const pageTextureRects: Record<PageSide, PageTextureRect> = {
  left: {
    bottom: 6291 / 7008,
    left: 575 / 4672,
    right: 1,
    top: 137 / 7008,
  },
  right: {
    bottom: 6302 / 7008,
    left: 0,
    right: 4101 / 4672,
    top: 119 / 7008,
  },
};

const desktopPageTextClassName = "font-book text-[#5a3a1f] [text-shadow:0_0_0.35px_rgba(72,43,18,0.42)]";
const fullPageTextureRect: PageTextureRect = { bottom: 1, left: 0, right: 1, top: 0 };

const turnVertexShader = `
attribute vec2 a_position;
attribute float a_depth;
attribute vec2 a_uv;
attribute float a_light;

varying vec2 v_uv;
varying float v_light;

void main() {
  gl_Position = vec4(a_position, a_depth, 1.0);
  v_uv = a_uv;
  v_light = a_light;
}
`;

const turnFragmentShader = `
precision mediump float;

uniform sampler2D u_front;
uniform sampler2D u_back;
uniform vec4 u_frontUvRect;
uniform vec4 u_backUvRect;
uniform vec3 u_paper;

varying vec2 v_uv;
varying float v_light;

vec2 mapPageUv(vec2 uv, vec4 rect) {
  return mix(rect.xy, rect.zw, uv);
}

void main() {
  vec4 front = texture2D(u_front, mapPageUv(v_uv, u_frontUvRect));
  vec4 back = texture2D(u_back, mapPageUv(vec2(1.0 - v_uv.x, v_uv.y), u_backUvRect));
  vec4 tex = gl_FrontFacing ? front : back;

  float pageAlpha = tex.a < 0.8 ? 0.0 : 1.0;

  if (pageAlpha < 0.01) {
    discard;
  }

  float textureMix = smoothstep(0.02, 0.82, tex.a);
  vec3 color = mix(u_paper, tex.rgb, textureMix);

  float softenedLight = mix(1.0, v_light, 0.62);
  float frontLight = clamp(softenedLight + 0.025, 0.84, 1.11);
  float backLight = clamp(softenedLight + 0.085, 0.90, 1.18);
  gl_FragColor = vec4(color * (gl_FrontFacing ? frontLight : backLight), pageAlpha);
}
`;

const turnImageCache = new Map<string, Promise<HTMLImageElement>>();

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

function easeInOutCubic(value: number) {
  return value < 0.5 ? 4 * value * value * value : 1 - Math.pow(-2 * value + 2, 3) / 2;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function smoothStep(edge0: number, edge1: number, value: number) {
  const x = clamp((value - edge0) / (edge1 - edge0), 0, 1);
  return x * x * (3 - 2 * x);
}

function nextFrame() {
  return new Promise<void>((resolve) => {
    window.requestAnimationFrame(() => resolve());
  });
}

async function waitForRenderedAssets(element: HTMLElement) {
  await document.fonts?.ready;

  const images = Array.from(element.querySelectorAll("img"));
  await Promise.all(
    images.map(async (image) => {
      if (image.complete) {
        return;
      }

      if (typeof image.decode === "function") {
        await image.decode().catch(() => undefined);
        return;
      }

      await new Promise<void>((resolve) => {
        image.onload = () => resolve();
        image.onerror = () => resolve();
      });
    }),
  );
}

function isVisibleElement(element: Element) {
  if (!(element instanceof HTMLElement)) {
    return false;
  }

  const style = window.getComputedStyle(element);
  return style.display !== "none" && style.visibility !== "hidden" && Number(style.opacity) > 0;
}

function isTransparentColor(color: string) {
  return color === "transparent" || color === "rgba(0, 0, 0, 0)" || /rgba\([^)]*,\s*0\)$/.test(color);
}

function drawRect(context: CanvasRenderingContext2D, rect: DOMRect, rootRect: DOMRect, color: string, alpha = 1) {
  if (rect.width <= 0 || rect.height <= 0 || isTransparentColor(color)) {
    return;
  }

  context.save();
  context.globalAlpha *= alpha;
  context.fillStyle = color;
  context.fillRect(rect.left - rootRect.left, rect.top - rootRect.top, rect.width, rect.height);
  context.restore();
}

function drawElementSurfaces(root: HTMLElement, context: CanvasRenderingContext2D, rootRect: DOMRect) {
  const elements = [root, ...Array.from(root.querySelectorAll<HTMLElement>("*"))];

  elements.forEach((element) => {
    if (!isVisibleElement(element)) {
      return;
    }

    const style = window.getComputedStyle(element);
    const backgroundColor = style.backgroundColor;
    const borderColor = style.borderColor;
    const borderWidth = Math.max(
      Number.parseFloat(style.borderTopWidth) || 0,
      Number.parseFloat(style.borderRightWidth) || 0,
      Number.parseFloat(style.borderBottomWidth) || 0,
      Number.parseFloat(style.borderLeftWidth) || 0,
    );

    Array.from(element.getClientRects()).forEach((rect) => {
      drawRect(context, rect, rootRect, backgroundColor, Number(style.opacity) || 1);

      if (borderWidth > 0 && !isTransparentColor(borderColor)) {
        context.save();
        context.globalAlpha *= Number(style.opacity) || 1;
        context.strokeStyle = borderColor;
        context.lineWidth = borderWidth;
        context.strokeRect(rect.left - rootRect.left, rect.top - rootRect.top, rect.width, rect.height);
        context.restore();
      }
    });
  });
}

function cssPixelValue(value: string) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}

function objectFitRect(image: HTMLImageElement, rect: DOMRect, style: CSSStyleDeclaration) {
  const sourceRatio = image.naturalWidth / image.naturalHeight;
  const shouldContain = style.objectFit === "contain" || style.objectFit === "scale-down";
  let width = rect.width;
  let height = rect.height;
  let left = rect.left;
  let top = rect.top;

  if (shouldContain) {
    const cssWidth = cssPixelValue(style.width) || rect.width;
    const cssHeight = cssPixelValue(style.height) || rect.height;
    const scaleX = rect.width / cssWidth;
    const scaleY = rect.height / cssHeight;
    const targetRatio = cssWidth / cssHeight;
    let fittedCssWidth = cssWidth;
    let fittedCssHeight = cssHeight;

    // object-fit is resolved before ancestor transforms such as the book's scaleX(1.08).
    if (sourceRatio > targetRatio) {
      fittedCssHeight = cssWidth / sourceRatio;
    } else {
      fittedCssWidth = cssHeight * sourceRatio;
    }

    width = fittedCssWidth * scaleX;
    height = fittedCssHeight * scaleY;
    left = rect.left + ((cssWidth - fittedCssWidth) / 2) * scaleX;
    top = rect.top + ((cssHeight - fittedCssHeight) / 2) * scaleY;
  }

  return {
    height,
    left,
    top,
    width,
  };
}

function drawElementImages(root: HTMLElement, context: CanvasRenderingContext2D, rootRect: DOMRect) {
  Array.from(root.querySelectorAll("img")).forEach((image) => {
    if (!isVisibleElement(image) || !image.complete || image.naturalWidth <= 0 || image.naturalHeight <= 0) {
      return;
    }

    const style = window.getComputedStyle(image);
    const rect = image.getBoundingClientRect();
    const fitted = objectFitRect(image, rect, style);
    context.save();
    context.globalAlpha *= Number(style.opacity) || 1;
    context.drawImage(image, fitted.left - rootRect.left, fitted.top - rootRect.top, fitted.width, fitted.height);
    context.restore();
  });
}

function canvasFontFromStyle(style: CSSStyleDeclaration) {
  const fontStyle = style.fontStyle === "normal" ? "" : style.fontStyle;
  const fontVariant = style.fontVariant === "normal" ? "" : style.fontVariant;
  const fontWeight = style.fontWeight === "normal" ? "" : style.fontWeight;
  return `${fontStyle} ${fontVariant} ${fontWeight} ${style.fontSize} ${style.fontFamily}`.replace(/\s+/g, " ").trim();
}

function drawText(context: CanvasRenderingContext2D, text: string, rect: DOMRect, rootRect: DOMRect, style: CSSStyleDeclaration) {
  if (!text || rect.width <= 0 || rect.height <= 0 || isTransparentColor(style.color)) {
    return;
  }

  context.save();
  context.font = canvasFontFromStyle(style);
  context.fillStyle = style.color;
  context.globalAlpha *= Number(style.opacity) || 1;
  context.textBaseline = "top";
  context.fillText(text, rect.left - rootRect.left, rect.top - rootRect.top);
  context.restore();
}

function drawListMarkers(root: HTMLElement, context: CanvasRenderingContext2D, rootRect: DOMRect) {
  Array.from(root.querySelectorAll("li")).forEach((item) => {
    if (!isVisibleElement(item)) {
      return;
    }

    const style = window.getComputedStyle(item);
    const rect = item.getBoundingClientRect();
    const parent = item.parentElement;
    const isOrdered = parent?.tagName.toLowerCase() === "ol";
    const siblings = parent ? Array.from(parent.children).filter((child) => child.tagName.toLowerCase() === "li") : [];
    const marker = isOrdered ? `${Math.max(1, siblings.indexOf(item) + 1)}.` : "•";
    const fontSize = Number.parseFloat(style.fontSize) || 14;
    const markerRect = new DOMRect(rect.left - fontSize * 0.95, rect.top, fontSize * 0.75, rect.height);
    drawText(context, marker, markerRect, rootRect, style);
  });
}

function drawTextNodes(root: HTMLElement, context: CanvasRenderingContext2D, rootRect: DOMRect) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let node = walker.nextNode();
  const range = document.createRange();

  while (node) {
    const text = node.textContent ?? "";
    const parent = node.parentElement;

    if (parent && isVisibleElement(parent) && text.trim()) {
      const style = window.getComputedStyle(parent);
      let offset = 0;

      for (const character of Array.from(text)) {
        const nextOffset = offset + character.length;

        if (!/\s/.test(character)) {
          range.setStart(node, offset);
          range.setEnd(node, nextOffset);

          Array.from(range.getClientRects()).forEach((rect) => {
            drawText(context, character, rect, rootRect, style);
          });
        }

        offset = nextOffset;
      }
    }

    node = walker.nextNode();
  }

  range.detach();
}

async function drawElementToCanvas(element: HTMLElement, cssWidth: number, cssHeight: number, bleedCssPx = 0) {
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  const canvas = document.createElement("canvas");
  const canvasCssWidth = cssWidth + bleedCssPx * 2;
  const canvasCssHeight = cssHeight + bleedCssPx * 2;
  const width = Math.max(1, Math.round(canvasCssWidth * scale));
  const height = Math.max(1, Math.round(canvasCssHeight * scale));
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d");

  if (!context) {
    return canvas;
  }

  const rootRect = element.getBoundingClientRect();
  context.scale(scale, scale);
  context.clearRect(0, 0, canvasCssWidth, canvasCssHeight);
  context.translate(bleedCssPx, bleedCssPx);

  drawElementSurfaces(element, context, rootRect);
  drawElementImages(element, context, rootRect);
  drawListMarkers(element, context, rootRect);
  drawTextNodes(element, context, rootRect);

  return canvas;
}

async function renderElementSnapshotToCanvas(element: HTMLElement) {
  const rect = element.getBoundingClientRect();
  return {
    canvas: await drawElementToCanvas(element, rect.width, rect.height, textCanvasBleedCssPx),
    cssHeight: rect.height,
    cssWidth: rect.width,
  };
}

function pageArticleClassName(side: PageSide) {
  return side === "left"
    ? "relative z-10 min-w-0 overflow-hidden pr-[3%] max-[760px]:overflow-visible max-[760px]:pr-0"
    : "relative z-10 min-w-0 overflow-hidden pl-[3%] max-[760px]:overflow-visible max-[760px]:pl-0";
}

async function renderHtmlToCanvas(html: string, side: PageSide, cssWidth: number, cssHeight: number, containerCssWidth: number) {
  const mount = document.createElement("div");
  mount.style.position = "fixed";
  mount.style.left = "-10000px";
  mount.style.top = "0";
  mount.style.width = `${containerCssWidth}px`;
  mount.style.height = `${cssHeight}px`;
  mount.style.containerType = "inline-size";
  mount.style.overflow = "hidden";
  mount.style.pointerEvents = "none";
  mount.style.zIndex = "-1";
  mount.innerHTML = `<div class="${desktopPageTextClassName}" style="height:${cssHeight}px;overflow:hidden;width:${cssWidth}px"><article class="${pageArticleClassName(
    side,
  )}" style="height:100%;width:100%">${html}</article></div>`;
  document.body.appendChild(mount);

  try {
    await nextFrame();
    await nextFrame();

    const renderedElement = mount.firstElementChild as HTMLElement | null;

    if (!renderedElement) {
      throw new Error("Unable to render page content.");
    }

    await waitForRenderedAssets(renderedElement);
    return await drawElementToCanvas(renderedElement, cssWidth, cssHeight, textCanvasBleedCssPx);
  } finally {
    mount.remove();
  }
}

async function renderMarkdownToCanvas(markdown: string, side: PageSide, cssWidth: number, cssHeight: number, containerCssWidth: number) {
  const mount = document.createElement("div");
  mount.style.position = "fixed";
  mount.style.left = "-10000px";
  mount.style.top = "0";
  mount.style.width = `${containerCssWidth}px`;
  mount.style.height = `${cssHeight}px`;
  mount.style.containerType = "inline-size";
  mount.style.overflow = "hidden";
  mount.style.pointerEvents = "none";
  mount.style.zIndex = "-1";
  document.body.appendChild(mount);

  const root = createRoot(mount);
  const articleClassName = pageArticleClassName(side);

  try {
    flushSync(() => {
      root.render(
        <div className={desktopPageTextClassName} style={{ height: `${cssHeight}px`, overflow: "hidden", width: `${cssWidth}px` }}>
          <article className={articleClassName} style={{ height: "100%", width: "100%" }}>
            <MarkdownPage markdown={markdown} />
          </article>
        </div>,
      );
    });

    await nextFrame();
    await nextFrame();

    const renderedElement = mount.firstElementChild as HTMLElement | null;

    if (!renderedElement) {
      throw new Error("Unable to render page content.");
    }

    await waitForRenderedAssets(renderedElement);
    return await drawElementToCanvas(renderedElement, cssWidth, cssHeight, textCanvasBleedCssPx);
  } finally {
    root.unmount();
    mount.remove();
  }
}

function getContainerQueryWidth(element: HTMLElement, fallbackWidth: number) {
  let current = element.parentElement;

  while (current) {
    const style = window.getComputedStyle(current);

    if (style.containerType !== "normal") {
      const styleWidth = Number.parseFloat(style.width);
      const rectWidth = current.getBoundingClientRect().width;
      return styleWidth || current.clientWidth || rectWidth || fallbackWidth;
    }

    current = current.parentElement;
  }

  return fallbackWidth;
}

function getArticlePlacement(canvas: HTMLCanvasElement, side: PageSide): ArticlePlacement {
  const article = document.querySelector<HTMLElement>(`[data-page-side="${side}"]`);
  const canvasRect = canvas.getBoundingClientRect();
  const rect = side === "right" ? pageTurnRects.next : pageTurnRects.prev;
  const pageLeft = canvasRect.left + rect.left * canvasRect.width;
  const pageTop = canvasRect.top + rect.top * canvasRect.height;
  const pageWidth = rect.width * canvasRect.width;
  const pageHeight = rect.height * canvasRect.height;

  if (!article || pageWidth <= 0 || pageHeight <= 0) {
    return {
      containerCssWidth: canvas.offsetWidth || canvasRect.width,
      cssHeight: 640,
      cssWidth: 480,
      height: 0.68,
      left: 0.08,
      top: 0.16,
      width: 0.8,
    };
  }

  const articleRect = article.getBoundingClientRect();
  const fallbackContainerWidth = canvas.offsetWidth || canvasRect.width;

  return {
    containerCssWidth: getContainerQueryWidth(article, fallbackContainerWidth),
    cssHeight: article.offsetHeight || articleRect.height,
    cssWidth: article.offsetWidth || articleRect.width,
    height: articleRect.height / pageHeight,
    left: (articleRect.left - pageLeft) / pageWidth,
    top: (articleRect.top - pageTop) / pageHeight,
    width: articleRect.width / pageWidth,
  };
}

async function createCompositedTurnTexture(
  image: HTMLImageElement,
  side: PageSide,
  markdown: string,
  placement: ArticlePlacement,
  html?: string,
  renderedSnapshot?: PageSnapshot | Promise<PageSnapshot>,
) {
  const crop = pageTextureRects[side];
  const sourceLeft = crop.left * image.naturalWidth;
  const sourceTop = crop.top * image.naturalHeight;
  const sourceWidth = (crop.right - crop.left) * image.naturalWidth;
  const sourceHeight = (crop.bottom - crop.top) * image.naturalHeight;
  const textureWidth = 1536;
  const textureHeight = Math.max(1, Math.round((textureWidth * sourceHeight) / sourceWidth));
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");

  canvas.width = textureWidth;
  canvas.height = textureHeight;
  context?.drawImage(image, sourceLeft, sourceTop, sourceWidth, sourceHeight, 0, 0, textureWidth, textureHeight);

  if (context && (renderedSnapshot || markdown.trim())) {
    const snapshot = renderedSnapshot ? await renderedSnapshot : undefined;
    const textCanvas =
      snapshot?.canvas ??
      (html
        ? await renderHtmlToCanvas(html, side, placement.cssWidth, placement.cssHeight, placement.containerCssWidth)
        : await renderMarkdownToCanvas(markdown, side, placement.cssWidth, placement.cssHeight, placement.containerCssWidth));
    const textCssWidth = snapshot?.cssWidth ?? placement.cssWidth;
    const textCssHeight = snapshot?.cssHeight ?? placement.cssHeight;
    const bleedX = (textCanvasBleedCssPx / Math.max(1, textCssWidth)) * placement.width * textureWidth;
    const bleedY = (textCanvasBleedCssPx / Math.max(1, textCssHeight)) * placement.height * textureHeight;
    context.save();
    context.globalAlpha = 0.9;
    context.globalCompositeOperation = "multiply";
    context.drawImage(
      textCanvas,
      placement.left * textureWidth - bleedX,
      placement.top * textureHeight - bleedY,
      placement.width * textureWidth + bleedX * 2,
      placement.height * textureHeight + bleedY * 2,
    );
    context.restore();
  }

  return canvas;
}

function loadTurnImage(src: string) {
  const cachedImage = turnImageCache.get(src);

  if (cachedImage) {
    return cachedImage;
  }

  const imagePromise = new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => {
      if (typeof image.decode === "function") {
        void image
          .decode()
          .catch(() => undefined)
          .then(() => resolve(image));
        return;
      }

      resolve(image);
    };
    image.onerror = () => reject(new Error(`Unable to load page texture: ${src}`));
    image.src = src;
  });

  turnImageCache.set(src, imagePromise);
  return imagePromise;
}

function pageTextureUrl(side: PageSide) {
  return assetUrl(`/main-panel-cutout-v2-${side}.png`);
}

function createTurnShader(gl: WebGLRenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) {
    throw new Error("Unable to create WebGL shader.");
  }

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? "Unknown shader compile error.";
    gl.deleteShader(shader);
    throw new Error(log);
  }

  return shader;
}

function createTurnProgram(gl: WebGLRenderingContext) {
  const vertexShader = createTurnShader(gl, gl.VERTEX_SHADER, turnVertexShader);
  const fragmentShader = createTurnShader(gl, gl.FRAGMENT_SHADER, turnFragmentShader);
  const program = gl.createProgram();

  if (!program) {
    throw new Error("Unable to create WebGL program.");
  }

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) ?? "Unknown WebGL program link error.";
    gl.deleteProgram(program);
    throw new Error(log);
  }

  return program;
}

function createTurnTexture(gl: WebGLRenderingContext, image: TexImageSource) {
  const texture = gl.createTexture();

  if (!texture) {
    throw new Error("Unable to create WebGL texture.");
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.bindTexture(gl.TEXTURE_2D, null);

  return texture;
}

function BookTurnCanvas({ onReady, turn }: { onReady: (turn: PageTurn) => void; turn: PageTurn }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = canvas?.getContext("webgl", {
      alpha: true,
      antialias: true,
      depth: true,
      premultipliedAlpha: false,
    });

    if (!canvas || !gl) {
      onReady(turn);
      return;
    }

    const frontSide: PageSide = turn.direction === "next" ? "right" : "left";
    const backSide: PageSide = turn.direction === "next" ? "left" : "right";
    const frontUrl = pageTextureUrl(frontSide);
    const backUrl = pageTextureUrl(backSide);
    const frontElement = document.querySelector<HTMLElement>(`[data-page-side="${frontSide}"]`);
    const frontHtml = frontElement?.innerHTML;
    const frontUvRect = fullPageTextureRect;
    const backUvRect = fullPageTextureRect;
    let animationFrame = 0;
    let disposed = false;
    let program: WebGLProgram | null = null;
    let frontTexture: WebGLTexture | null = null;
    let backTexture: WebGLTexture | null = null;
    let positionBuffer: WebGLBuffer | null = null;
    let depthBuffer: WebGLBuffer | null = null;
    let uvBuffer: WebGLBuffer | null = null;
    let lightBuffer: WebGLBuffer | null = null;
    let indexBuffer: WebGLBuffer | null = null;

    const vertexCount = (pageMeshColumns + 1) * (pageMeshRows + 1);
    const positions = new Float32Array(vertexCount * 2);
    const depths = new Float32Array(vertexCount);
    const uvs = new Float32Array(vertexCount * 2);
    const lights = new Float32Array(vertexCount);
    const indices = new Uint16Array(pageMeshColumns * pageMeshRows * 6);

    for (let row = 0; row <= pageMeshRows; row += 1) {
      for (let column = 0; column <= pageMeshColumns; column += 1) {
        const vertex = row * (pageMeshColumns + 1) + column;
        uvs[vertex * 2] = column / pageMeshColumns;
        uvs[vertex * 2 + 1] = row / pageMeshRows;
      }
    }

    let indexCursor = 0;
    for (let row = 0; row < pageMeshRows; row += 1) {
      for (let column = 0; column < pageMeshColumns; column += 1) {
        const i0 = row * (pageMeshColumns + 1) + column;
        const i1 = i0 + 1;
        const i2 = i0 + pageMeshColumns + 1;
        const i3 = i2 + 1;
        indices[indexCursor] = i0;
        indices[indexCursor + 1] = i2;
        indices[indexCursor + 2] = i1;
        indices[indexCursor + 3] = i1;
        indices[indexCursor + 4] = i2;
        indices[indexCursor + 5] = i3;
        indexCursor += 6;
      }
    }

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.round(rect.width * pixelRatio));
      const height = Math.max(1, Math.round(rect.height * pixelRatio));

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      gl.viewport(0, 0, width, height);
    };

    const fillMesh = (progress: number) => {
      const rect = pageTurnRects[turn.direction];
      const left = rect.left * 2 - 1;
      const right = (rect.left + rect.width) * 2 - 1;
      const top = 1 - rect.top * 2;
      const bottom = 1 - (rect.top + rect.height) * 2;
      const hingeX = turn.direction === "next" ? left : right;
      const outward = turn.direction === "next" ? 1 : -1;
      const pageWidth = Math.abs(right - left);
      const pageHeight = Math.abs(top - bottom);
      const centerY = (top + bottom) / 2;
      const eased = easeInOutCubic(progress);
      const baseAngle = Math.PI * eased;
      const turnWave = Math.sin(Math.PI * progress);
      const activeWave = Math.pow(turnWave, 0.72);
      const curlAmount = 0.46 * activeWave * (1 - progress * 0.16);
      const leadCornerWave = Math.sin(Math.min(1, progress * 1.45) * Math.PI) * Math.pow(1 - progress, 0.35);
      const trailCornerWave = Math.sin(Math.min(1, progress * 1.1) * Math.PI) * Math.pow(1 - progress, 0.55);
      const leadCornerRow = turn.direction === "next" ? 1 : 0;
      const trailCornerRow = 1 - leadCornerRow;
      const perspective = 3.1;

      for (let row = 0; row <= pageMeshRows; row += 1) {
        const v = row / pageMeshRows;
        const baseY = top + (bottom - top) * v;
        const leadCornerY = Math.max(0, 1 - Math.abs(v - leadCornerRow) * 2.45);
        const trailCornerY = Math.max(0, 1 - Math.abs(v - trailCornerRow) * 2.2);
        const rowBias = v - 0.5;

        for (let column = 0; column <= pageMeshColumns; column += 1) {
          const u = column / pageMeshColumns;
          const vertex = row * (pageMeshColumns + 1) + column;
          const distanceFromSpine = turn.direction === "next" ? u : 1 - u;
          const spineRelease = smoothStep(0.06, 0.24, distanceFromSpine);
          const edgeCurve = Math.sin(distanceFromSpine * Math.PI);
          const freeEdge = Math.pow(distanceFromSpine, 1.18);
          const ridgeA = Math.sin((distanceFromSpine * 2.2 + v * 1.35 + progress * 1.4) * Math.PI);
          const ridgeB = Math.sin((distanceFromSpine * 4.9 - v * 2.7 + progress * 0.85) * Math.PI);
          const ridgeC = Math.sin((distanceFromSpine * 7.4 + v * 4.1 - progress * 1.7) * Math.PI);
          const clothRipple = (ridgeA * 0.56 + ridgeB * 0.31 + ridgeC * 0.13) * activeWave * freeEdge * (0.18 + edgeCurve * 0.82);
          const diagonalFold = Math.pow(clamp(Math.sin((distanceFromSpine * 1.35 + (1 - v) * 0.9 - progress * 0.55) * Math.PI), 0, 1), 1.8);
          const leadCornerLift = Math.pow(distanceFromSpine, 1.72) * Math.pow(leadCornerY, 1.45) * leadCornerWave;
          const trailCornerLift = Math.pow(distanceFromSpine, 2.05) * Math.pow(trailCornerY, 1.65) * trailCornerWave * 0.48;
          const cornerLift = leadCornerLift + trailCornerLift;
          const angle =
            baseAngle +
            curlAmount * edgeCurve * (0.18 + distanceFromSpine * 0.82) +
            clothRipple * 0.34 +
            diagonalFold * activeWave * freeEdge * 0.18;
          const x3 = hingeX + outward * pageWidth * distanceFromSpine * Math.cos(angle);
          let z3 = pageWidth * distanceFromSpine * Math.abs(Math.sin(angle)) * 0.42;
          z3 += cornerLift * pageWidth * 0.24;
          z3 += Math.abs(clothRipple) * pageWidth * 0.13 + diagonalFold * pageWidth * activeWave * freeEdge * 0.1;

          const rowCurl = rowBias * Math.sin(angle) * pageHeight * (0.026 + activeWave * 0.018) * spineRelease;
          const clothYShift = (ridgeA * 0.018 + ridgeB * 0.012) * pageHeight * activeWave * freeEdge * spineRelease;
          const cornerYShift =
            ((leadCornerRow === 1 ? -1 : 1) * leadCornerLift * pageHeight * 0.05 +
              (trailCornerRow === 1 ? -1 : 1) * trailCornerLift * pageHeight * 0.035) *
            spineRelease;
          const y3 = baseY + rowCurl + clothYShift + cornerYShift;
          const projectedScale = perspective / (perspective - z3);
          const x2 = hingeX + (x3 - hingeX) * projectedScale;
          const y2 = centerY + (y3 - centerY) * projectedScale;

          positions[vertex * 2] = x2;
          positions[vertex * 2 + 1] = y2;
          depths[vertex] = Math.max(-0.9, Math.min(0.9, -z3 * 0.42));
          lights[vertex] =
            0.94 +
            Math.cos(angle - 0.32) * 0.12 -
            Math.abs(Math.sin(angle)) * 0.16 +
            cornerLift * 0.08 +
            clothRipple * 0.1 -
            diagonalFold * 0.1;
        }
      }
    };

    const bindAttribute = (buffer: WebGLBuffer | null, location: number, size: number) => {
      if (!buffer || location < 0) {
        return;
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
    };

    const renderFrame = (startedAt: number, now: number) => {
      if (disposed || !program || !frontTexture || !backTexture || !positionBuffer || !depthBuffer || !uvBuffer || !lightBuffer || !indexBuffer) {
        return;
      }

      resize();
      fillMesh(Math.min(1, (now - startedAt) / pageTurnDurationMs));

      gl.clearColor(0, 0, 0, 0);
      gl.clearDepth(1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      gl.useProgram(program);

      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, depthBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, depths, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, lightBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, lights, gl.DYNAMIC_DRAW);

      bindAttribute(positionBuffer, gl.getAttribLocation(program, "a_position"), 2);
      bindAttribute(depthBuffer, gl.getAttribLocation(program, "a_depth"), 1);
      bindAttribute(uvBuffer, gl.getAttribLocation(program, "a_uv"), 2);
      bindAttribute(lightBuffer, gl.getAttribLocation(program, "a_light"), 1);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, frontTexture);
      gl.uniform1i(gl.getUniformLocation(program, "u_front"), 0);
      gl.uniform4f(gl.getUniformLocation(program, "u_frontUvRect"), frontUvRect.left, frontUvRect.top, frontUvRect.right, frontUvRect.bottom);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, backTexture);
      gl.uniform1i(gl.getUniformLocation(program, "u_back"), 1);
      gl.uniform4f(gl.getUniformLocation(program, "u_backUvRect"), backUvRect.left, backUvRect.top, backUvRect.right, backUvRect.bottom);
      gl.uniform3f(gl.getUniformLocation(program, "u_paper"), 0.88, 0.74, 0.46);

      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
      gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0);

      if (now - startedAt < pageTurnDurationMs) {
        animationFrame = window.requestAnimationFrame((nextNow) => renderFrame(startedAt, nextNow));
      }
    };

    void Promise.all([loadTurnImage(frontUrl), loadTurnImage(backUrl)])
      .then(async ([frontImage, backImage]) => {
        if (disposed) {
          return;
        }

        const frontPlacement = getArticlePlacement(canvas, frontSide);
        const backPlacement = getArticlePlacement(canvas, backSide);
        const frontSnapshot = turn.frontSnapshot ?? (frontElement ? renderElementSnapshotToCanvas(frontElement) : undefined);

        if (disposed) {
          return;
        }

        const [frontCanvas, backCanvas] = await Promise.all([
          createCompositedTurnTexture(frontImage, frontSide, turn.frontMarkdown, frontPlacement, frontHtml, frontSnapshot),
          createCompositedTurnTexture(backImage, backSide, turn.backMarkdown, backPlacement, undefined, turn.backSnapshot),
        ]);

        if (disposed) {
          return;
        }

        program = createTurnProgram(gl);
        frontTexture = createTurnTexture(gl, frontCanvas);
        backTexture = createTurnTexture(gl, backCanvas);
        positionBuffer = gl.createBuffer();
        depthBuffer = gl.createBuffer();
        uvBuffer = gl.createBuffer();
        lightBuffer = gl.createBuffer();
        indexBuffer = gl.createBuffer();

        gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

        const startedAt = performance.now();
        renderFrame(startedAt, startedAt);
        onReady(turn);
      })
      .catch((error: unknown) => {
        console.error(error);
      });

    return () => {
      disposed = true;
      window.cancelAnimationFrame(animationFrame);

      if (frontTexture) gl.deleteTexture(frontTexture);
      if (backTexture) gl.deleteTexture(backTexture);
      if (positionBuffer) gl.deleteBuffer(positionBuffer);
      if (depthBuffer) gl.deleteBuffer(depthBuffer);
      if (uvBuffer) gl.deleteBuffer(uvBuffer);
      if (lightBuffer) gl.deleteBuffer(lightBuffer);
      if (indexBuffer) gl.deleteBuffer(indexBuffer);
      if (program) gl.deleteProgram(program);
    };
  }, [onReady, turn]);

  return <canvas className="book-turn-canvas" ref={canvasRef} />;
}

function BookTurnLayer({ onReady, turn }: { onReady: (turn: PageTurn) => void; turn: PageTurn | null }) {
  if (turn === null) {
    return null;
  }

  return (
    <div className="book-turn-layer max-[760px]:hidden" aria-hidden="true" data-render-surface="main-panel-texture">
      <BookTurnCanvas key={turn.id} onReady={onReady} turn={turn} />
    </div>
  );
}

export default function App() {
  const [pages, setPages] = useState<PageContent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showLoadingMask, setShowLoadingMask] = useState(true);
  const [pageIndex, setPageIndex] = useState(0);
  const [visiblePageIndexes, setVisiblePageIndexes] = useState({ left: 0, right: 0 });
  const [activeTurn, setActiveTurn] = useState<PageTurn | null>(null);
  const turnIdRef = useRef(0);
  const turnTimersRef = useRef<number[]>([]);
  const pageSnapshotCacheRef = useRef(new Map<string, Promise<PageSnapshot>>());

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
    void Promise.all([loadTurnImage(pageTextureUrl("left")), loadTurnImage(pageTextureUrl("right"))]);
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

  const clearTurnTimers = useCallback(() => {
    turnTimersRef.current.forEach((timer) => window.clearTimeout(timer));
    turnTimersRef.current = [];
  }, []);

  useEffect(() => clearTurnTimers, [clearTurnTimers]);

  useEffect(() => {
    const clearSnapshotCache = () => pageSnapshotCacheRef.current.clear();
    window.addEventListener("resize", clearSnapshotCache);
    return () => window.removeEventListener("resize", clearSnapshotCache);
  }, []);

  const cacheVisiblePageSnapshot = useCallback((page: number, side: PageSide) => {
    const article = document.querySelector<HTMLElement>(`[data-page-side="${side}"]`);
    const snapshot = article ? renderElementSnapshotToCanvas(article) : undefined;

    if (snapshot) {
      pageSnapshotCacheRef.current.set(`${page}:${side}`, snapshot);
    }

    return snapshot;
  }, []);

  const startTurn = useCallback(
    (targetIndex: number) => {
      if (activeTurn !== null || targetIndex === pageIndex || targetIndex < 0 || targetIndex >= pages.length) {
        return;
      }

      clearTurnTimers();
      const direction: TurnDirection = targetIndex > pageIndex ? "next" : "prev";
      const frontSide: PageSide = direction === "next" ? "right" : "left";
      const backSide: PageSide = direction === "next" ? "left" : "right";
      const sourcePage = pages[pageIndex] ?? emptyPage;
      const targetPage = pages[targetIndex] ?? emptyPage;
      const leftSnapshot = cacheVisiblePageSnapshot(visiblePageIndexes.left, "left");
      const rightSnapshot = cacheVisiblePageSnapshot(visiblePageIndexes.right, "right");
      const frontSnapshot = frontSide === "left" ? leftSnapshot : rightSnapshot;
      const backSnapshot = pageSnapshotCacheRef.current.get(`${targetIndex}:${backSide}`);
      turnIdRef.current += 1;
      setActiveTurn({
        backMarkdown: targetPage[backSide],
        backSnapshot,
        direction,
        frontMarkdown: sourcePage[frontSide],
        frontSnapshot,
        fromIndex: pageIndex,
        id: turnIdRef.current,
        toIndex: targetIndex,
      });
    },
    [activeTurn, cacheVisiblePageSnapshot, clearTurnTimers, pageIndex, pages, visiblePageIndexes.left, visiblePageIndexes.right],
  );

  const handleTurnReady = useCallback(
    (turn: PageTurn) => {
      clearTurnTimers();

      flushSync(() => {
        setVisiblePageIndexes(turn.direction === "next" ? { left: turn.fromIndex, right: turn.toIndex } : { left: turn.toIndex, right: turn.fromIndex });
      });

      turnTimersRef.current = [
        window.setTimeout(() => {
          setPageIndex(turn.toIndex);
          setVisiblePageIndexes({ left: turn.toIndex, right: turn.toIndex });
        }, pageTurnCoveredSwapMs),
        window.setTimeout(() => {
          setActiveTurn(null);
          turnTimersRef.current = [];
        }, pageTurnDurationMs),
      ];
    },
    [clearTurnTimers],
  );

  const turnPage = useCallback(
    (direction: TurnDirection) => {
      startTurn(direction === "next" ? pageIndex + 1 : pageIndex - 1);
    },
    [pageIndex, startTurn],
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

  const visibleLeftPage = pages[visiblePageIndexes.left] ?? emptyPage;
  const visibleRightPage = pages[visiblePageIndexes.right] ?? emptyPage;
  const totalPages = pages.length;
  const readingProgress = totalPages <= 1 ? 0 : (pageIndex / (totalPages - 1)) * 100;

  const jumpToPage = (targetIndex: number) => {
    startTurn(targetIndex);
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
          <div className="absolute left-1/2 top-1/2 z-[26] grid h-[68%] w-[82%] -translate-x-1/2 -translate-y-[47%] scale-x-[1.08] grid-cols-2 gap-[14%] py-[2%] font-book text-[#5a3a1f] opacity-90 mix-blend-multiply [text-shadow:0_0_0.35px_rgba(72,43,18,0.42)] max-[760px]:hidden">
            <article className="relative z-10 min-w-0 overflow-hidden pr-[3%] max-[760px]:overflow-visible max-[760px]:pr-0" data-page-side="left">
              <MarkdownPage markdown={visibleLeftPage.left} />
            </article>
            <article className="relative z-10 min-w-0 overflow-hidden pl-[3%] max-[760px]:overflow-visible max-[760px]:pl-0" data-page-side="right">
              <MarkdownPage markdown={visibleRightPage.right} />
            </article>
          </div>
          <BookTurnLayer onReady={handleTurnReady} turn={activeTurn} />
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
            className="absolute left-[9%] top-[18%] z-[29] h-[64%] w-[35%] bg-transparent max-[760px]:hidden"
            type="button"
            aria-label="Previous page"
            onClick={() => turnPage("prev")}
          />
          <button
            className="absolute right-[9%] top-[18%] z-[29] h-[64%] w-[35%] bg-transparent max-[760px]:hidden"
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
