"use client";

import { Brain } from "lucide-react";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "@/components/neurim/theme-toggle";

export function TopBar({
  sessionId,
  connected,
  showBrain,
  onToggleBrain,
}: {
  sessionId: string | null;
  connected: boolean;
  showBrain: boolean;
  onToggleBrain: (value: boolean) => void;
}) {
  return (
    <div className="flex items-center px-1 py-4 border-b border-border">
      <div className="flex items-center gap-2">
        <Brain size={18} className="text-approach" />
        <span className="font-sans tracking-tight font-semibold text-foreground">NEURIM</span>
        <span className="font-serif italic text-sm ml-2 text-[#98a2b3] dark:text-muted-foreground">
          session {sessionId ? sessionId.slice(0, 12) : "—"}
        </span>
      </div>

      <div className="ml-auto flex items-center gap-4">
        <button
          type="button"
          aria-pressed={showBrain}
          onClick={() => onToggleBrain(!showBrain)}
          className="flex items-center gap-2"
        >
          <span className="font-sans font-medium text-[12px] text-[#475467] dark:text-muted-foreground">
            Alpha asymmetry
          </span>
          <span
            className={cn(
              "relative inline-flex h-4 w-8 shrink-0 items-center rounded-full transition-colors",
              showBrain ? "bg-primary" : "bg-[#d3d7e0] dark:bg-white/10"
            )}
          >
            <span
              className={cn(
                "inline-block h-3 w-3 transform rounded-full bg-white transition-transform",
                showBrain ? "translate-x-4" : "translate-x-0.5"
              )}
            />
          </span>
        </button>

        <div className="flex items-center gap-2">
          <span
            className={cn(
              "h-[7px] w-[7px] rounded-full",
              connected ? "bg-primary animate-blink" : "bg-black/20 dark:bg-white/20"
            )}
          />
          <span className="font-mono uppercase text-[11px] tracking-wide text-[#98a2b3] dark:text-muted-foreground">
            {connected ? "Render live" : "Render offline"}
          </span>
        </div>

        <ThemeToggle className="h-8 w-8 rounded-md" />

      </div>
    </div>
  );
}
