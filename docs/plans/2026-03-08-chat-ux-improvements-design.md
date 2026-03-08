# Chat UX Improvements — Thinking Indicator + Inline Source Footnotes

**Date:** 2026-03-08
**Status:** Approved

## Problem

1. **No loading feedback** — When a message is sent, an empty assistant bubble appears with no visual indicator until the first token streams in. Looks broken.
2. **Source cards too heavy** — Vertical stack of full-width collapsible cards dominates the response area, pushing the chat input far down.

## Design

### 1. Thinking Indicator

Show a frosted glass pill with animated shimmer and bouncing dots when the assistant is processing but no tokens have arrived yet.

**State derivation** (no store changes):
- `isStreaming && content === ""` → ThinkingIndicator visible
- `isStreaming && content.length > 0` → slim blinking cursor
- `!isStreaming` → nothing

**Component: `ThinkingIndicator.tsx`**
- Glass-style pill (`bg-muted/40 backdrop-blur`)
- Three dots with staggered bounce animation
- "Thinking..." label in muted text
- Shimmer gradient sweep across the pill
- Fades out when first token arrives

**Streaming cursor upgrade:**
- Replace chunky pulsing block (`w-2 h-4`) with slim blinking line (`w-0.5 h-4`)

**CSS addition:**
- `@keyframes shimmer` — gradient sweep for the thinking pill

### 2. Inline Footnotes + Source Bar

Replace vertical source card stack with inline superscript footnotes and a compact source summary bar.

**Inline footnotes:**
- Custom `react-markdown` component detects `[1]`, `[2]` patterns in rendered text
- Renders as small superscript number chips (primary color pill)
- Hover → Radix `HoverCard` popover with: doc name, page/section, content preview, confidence
- Click → popover stays pinned

**Source summary bar (`SourceBar.tsx`):**
- Single horizontal row below message text
- Label: "N sources" + row of compact pills
- Each pill: source number + truncated doc name
- One line total — replaces the multi-card vertical stack

**Removed:** `SourceCard.tsx` (replaced by SourceFootnote + SourceBar)

## Files

| Action | File |
|--------|------|
| Create | `components/chat/ThinkingIndicator.tsx` |
| Create | `components/chat/SourceFootnote.tsx` |
| Create | `components/chat/SourceBar.tsx` |
| Edit   | `components/chat/MessageBubble.tsx` |
| Edit   | `index.css` |
| Delete | `components/chat/SourceCard.tsx` |

## Non-goals

- No backend changes (pipeline stepper would require SSE intermediate events)
- No side drawer for sources (too complex for mobile)
- AgentTrace stays as-is (already compact and toggleable)
