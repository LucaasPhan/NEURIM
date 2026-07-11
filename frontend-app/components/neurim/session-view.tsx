"use client";

import { RefreshCw, RotateCcw } from "lucide-react";
import { TopBar } from "@/components/neurim/top-bar";
import { PromptBubble } from "@/components/neurim/prompt-bubble";
import { ProcessingState } from "@/components/neurim/processing-state";
import { HeroCandidate } from "@/components/neurim/hero-candidate";
import { SignalRail } from "@/components/neurim/signal-rail";
import { SteerInput } from "@/components/neurim/steer-input";
import type { NeurimSession } from "@/hooks/use-session";

export function SessionView({ session }: { session: NeurimSession }) {
  if (session.phase === "completed") return <FinalResult session={session} />;

  if (session.phase === "finalizing") {
    return (
      <main className="mx-auto flex min-h-screen max-w-3xl flex-col items-center justify-center px-4 py-12">
        <PromptBubble prompt={session.submittedPrompt} />
        <div className="mt-12 w-full">
          <ProcessingState statusText={session.statusText} />
        </div>
      </main>
    );
  }

  const state = session.frame?.state ?? "explore";
  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 pb-36 sm:px-6">
      <TopBar
        sessionId={session.sessionId}
        connected={session.connected}
        showBrain={session.showBrain}
        onToggleBrain={session.setShowBrain}
      />
      <div className="mt-10"><PromptBubble prompt={session.submittedPrompt} /></div>
      <div className="mt-10 flex-1">
        {session.phase === "processing" ? (
          <ProcessingState statusText={session.statusText} />
        ) : (
          <div className="flex flex-col items-stretch gap-6 md:flex-row">
            <div className="md:flex-[1.35]">
              <HeroCandidate frameSrc={session.frameSrc} state={state} fps={session.fps} modelLabel="Live PNG" />
            </div>
            <SignalRail
              reward={session.reward}
              state={state}
              features={session.frame?.eeg_features}
              showBrain={session.showBrain}
            />
          </div>
        )}
      </div>
      <div className="fixed inset-x-0 bottom-0 z-20 bg-gradient-to-t from-background via-background/95 to-transparent px-4 pb-6 pt-16">
        <div className="mx-auto max-w-3xl">
          <SteerInput onSubmit={session.startSession} disabled={session.isSubmitting} />
        </div>
      </div>
    </main>
  );
}

function FinalResult({ session }: { session: NeurimSession }) {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col items-center justify-center px-4 py-10 sm:px-6">
      <PromptBubble prompt={session.submittedPrompt} />
      <div className="mt-8 w-full max-w-[min(72vh,640px)]">
        <div className="relative aspect-square overflow-hidden rounded-lg bg-black shadow-[0_24px_70px_rgba(16,24,40,.18)]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={session.finalSrc ?? ""} alt="Final NEURIM result" className="h-full w-full object-contain" />
        </div>
        <div className="mt-4 text-center font-mono text-xs text-muted-foreground">
          {session.resultRefined ? "OpenAI finalized" : "Unrefined final frame"}
        </div>
        {!session.resultRefined && session.finalizeError && (
          <p className="mt-2 text-center text-sm text-destructive">{session.finalizeError}</p>
        )}
      </div>
      <div className="mt-6 flex flex-wrap justify-center gap-3">
        {!session.resultRefined && (
          <button
            type="button"
            onClick={session.retryFinalization}
            disabled={session.isRetryingFinalization}
            className="inline-flex h-10 items-center gap-2 rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground disabled:opacity-50"
          >
            <RefreshCw size={16} className={session.isRetryingFinalization ? "animate-spin" : ""} />
            Retry refinement
          </button>
        )}
        <button
          type="button"
          onClick={session.reset}
          className="inline-flex h-10 items-center gap-2 rounded-md border border-border px-4 text-sm font-medium"
        >
          <RotateCcw size={16} />
          New prompt
        </button>
      </div>
    </main>
  );
}
