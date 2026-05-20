import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        cauldron: "#7dff8c",
        ember: "#f5b642",
        mana: "#7f7cff",
        arcane: "#b456ff",
        ink: "#1a1228",
      },
      fontFamily: {
        book: ['"IM Fell DW Pica"', "serif"],
        pixel: ["ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
      },
      keyframes: {
        candleGlow: {
          "0%, 100%": {
            opacity: "0.58",
            transform: "translate(-50%, -50%) scale(0.92)",
          },
          "50%": {
            opacity: "0.92",
            transform: "translate(-50%, -50%) scale(1.08)",
          },
        },
        candleSmoke: {
          "0%": {
            opacity: "0",
            transform: "translate(0, 0) scale(0.5)",
          },
          "20%": {
            opacity: "0.42",
          },
          "65%": {
            opacity: "0.24",
          },
          "100%": {
            opacity: "0",
            transform: "translate(-0.8rem, -5.8rem) scale(1.55)",
          },
        },
        staffGlow: {
          "0%, 100%": {
            opacity: "0.36",
            transform: "translate(-50%, -50%) scale(0.82)",
          },
          "50%": {
            opacity: "0.72",
            transform: "translate(-50%, -50%) scale(1.16)",
          },
        },
        pageFade: {
          "0%, 100%": {
            opacity: "1",
          },
          "45%": {
            opacity: "0",
          },
        },
      },
      animation: {
        "candle-glow": "candleGlow 1.8s ease-in-out infinite",
        "candle-smoke": "candleSmoke 3.8s ease-in-out infinite",
        "staff-glow": "staffGlow 2.2s ease-in-out infinite",
        "page-fade": "pageFade 420ms ease-in-out",
      },
      boxShadow: {
        pixel: "0 0 0 2px #07040d, 0 0 0 4px #c99433, 0 10px 0 #07040d",
        glow: "0 0 18px rgba(125, 255, 140, 0.55), 0 0 44px rgba(58, 224, 115, 0.25)",
      },
    },
  },
  plugins: [],
} satisfies Config;
