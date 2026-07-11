"use client";

import dynamic from "next/dynamic";
import {
  AudioLines,
  Brain,
  ChevronDown,
  CircleStop,
  Loader2,
  Mic,
  Plus,
  Radio,
  Send,
  SlidersHorizontal,
  Unplug,
  Wifi,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

const BrainActivity3D = dynamic(
  () => import("@/components/brain-activity-3d").then((mod) => mod.BrainActivity3D),
  { ssr: false, loading: () => <Skeleton className="h-[430px] rounded-[24px]" /> }
);

export type EEGFeatures = {
  channels: Array<{
    name: string;
    value: number;
    alpha_power?: number;
    quality?: number;
    position: [number, number, number];
  }>;
  faa: {
    raw: number | null;
    reward: number;
    left_channel: string;
    right_channel: string;
  };
};

type FrameMessage = {
  frame_b64: string;
  z: number[];
  step_index: number;
  t: number;
  format: "jpeg" | "png" | string;
  state: "calibrate" | "explore" | "refine" | "settle" | "recover" | string;
  reward_estimate: number;
  eeg_features?: EEGFeatures | null;
};

type BackendSession = {
  running: boolean;
  pid: number | null;
  started_at: string | null;
  prompt: string | null;
  exit_code: number | null;
};

type SessionIntentResponse = {
  ok: boolean;
  session_id: string;
  prompt: string;
  max_steps: number;
  prompt_routing: "local_api_server" | string;
  render_contract: string;
  backend_url: string;
  backend_session: BackendSession;
};

const examplePrompts = [
  "A calm golden retriever puppy on white bedding",
  "A futuristic bioluminescent garden at dawn",
  "A soft cinematic portrait with warm studio light",
];

const epocPositions: Record<string, [number, number, number]> = {
  AF3: [-0.42, 0.88, 0.22],
  F7: [-0.86, 0.58, 0.04],
  F3: [-0.46, 0.55, 0.36],
  FC5: [-0.72, 0.22, 0.22],
  T7: [-0.95, -0.08, 0],
  P7: [-0.78, -0.58, 0.1],
  O1: [-0.34, -0.9, 0.18],
  O2: [0.34, -0.9, 0.18],
  P8: [0.78, -0.58, 0.1],
  T8: [0.95, -0.08, 0],
  FC6: [0.72, 0.22, 0.22],
  F4: [0.46, 0.55, 0.36],
  F8: [0.86, 0.58, 0.04],
  AF4: [0.42, 0.88, 0.22],
};

function decodeFrameSrc(msg: FrameMessage) {
  return `data:image/${msg.format || "jpeg"};base64,${msg.frame_b64}`;
}

function promptHash(input: string) {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function seededUnit(seed: number, index: number) {
  const value = Math.sin(seed * 0.00001 + index * 12.9898) * 43758.5453;
  return value - Math.floor(value);
}

function makeMockImageSrc(seed: number) {
  const hueA = Math.round(8 + seededUnit(seed, 1) * 36);
  const hueB = Math.round(178 + seededUnit(seed, 2) * 54);
  const hueC = Math.round(248 + seededUnit(seed, 3) * 42);
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 720">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="hsl(${hueA} 92% 68%)"/>
          <stop offset="50%" stop-color="hsl(${hueB} 70% 50%)"/>
          <stop offset="100%" stop-color="hsl(${hueC} 76% 60%)"/>
        </linearGradient>
        <radialGradient id="light" cx="68%" cy="18%" r="60%">
          <stop offset="0%" stop-color="white" stop-opacity="0.62"/>
          <stop offset="100%" stop-color="white" stop-opacity="0"/>
        </radialGradient>
        <filter id="soften">
          <feGaussianBlur stdDeviation="20"/>
        </filter>
      </defs>
      <rect width="720" height="720" rx="42" fill="url(#bg)"/>
      <rect width="720" height="720" rx="42" fill="url(#light)"/>
      <ellipse cx="300" cy="360" rx="190" ry="150" fill="white" opacity="0.18" filter="url(#soften)"/>
      <ellipse cx="500" cy="460" rx="160" ry="110" fill="black" opacity="0.10" filter="url(#soften)"/>
      <path d="M100 528 C230 410 322 594 462 476 C522 426 588 410 642 438" fill="none" stroke="white" stroke-width="18" stroke-linecap="round" opacity="0.28"/>
    </svg>
  `;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function makeMockSession(prompt: string) {
  const seed = promptHash(prompt);
  const reward = Number((-0.08 + seededUnit(seed, 8) * 0.72).toFixed(3));
  const rawFaa = Number((reward * 0.22 + (seededUnit(seed, 9) - 0.5) * 0.04).toFixed(3));
  const z = Array.from({ length: 8 }, (_, index) => Number((seededUnit(seed, index + 16) * 2 - 1).toFixed(3)));
  const channels = Object.entries(epocPositions).map(([name, position], index) => ({
    name,
    value: Number(((seededUnit(seed, index + 32) - 0.5) * 18).toFixed(3)),
    alpha_power: Number((0.2 + seededUnit(seed, index + 48) * 1.4 + Math.max(0, reward) * 0.35).toFixed(3)),
    quality: Number((0.72 + seededUnit(seed, index + 64) * 0.22).toFixed(3)),
    position,
  }));

  return {
    candidateSrc: makeMockImageSrc(seed),
    frame: {
      frame_b64: "mock",
      z,
      step_index: 1,
      t: Date.now() / 1000,
      format: "mock",
      state: "explore",
      reward_estimate: reward,
      eeg_features: {
        channels,
        faa: {
          raw: rawFaa,
          reward,
          left_channel: "F3",
          right_channel: "F4",
        },
      },
    } satisfies FrameMessage,
  };
}

export function NeurimDashboard() {
  const [url, setUrl] = useState("ws://localhost:8765");
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState("Not connected");
  const [frame, setFrame] = useState<FrameMessage | null>(null);
  const [frameSrc, setFrameSrc] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [promptDraft, setPromptDraft] = useState("");
  const [submittedPrompt, setSubmittedPrompt] = useState("");
  const [sessionIntent, setSessionIntent] = useState<SessionIntentResponse | null>(null);
  const [sessionStatus, setSessionStatus] = useState("Ready");
  const [isSavingIntent, setIsSavingIntent] = useState(false);
  const [showBrainActivity, setShowBrainActivity] = useState(true);
  const wsRef = useRef<WebSocket | null>(null);
  const frameCounter = useRef({ count: 0, startedAt: 0 });

  useEffect(() => {
    return () => wsRef.current?.close();
  }, []);

  function connect() {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(url);
    wsRef.current = ws;
    setStatus("Connecting");

    ws.addEventListener("open", () => {
      ws.send(JSON.stringify({ role: "display" }));
      setConnected(true);
      setStatus("Connected");
    });

    ws.addEventListener("message", (event) => {
      try {
        const msg = JSON.parse(event.data) as FrameMessage;
        if (!msg.frame_b64 || !Array.isArray(msg.z)) return;
        const now = Date.now();
        setFrame(msg);
        setFrameSrc(decodeFrameSrc(msg));
        if (frameCounter.current.startedAt === 0) {
          frameCounter.current = { count: 0, startedAt: now };
        }
        frameCounter.current.count += 1;
        const elapsed = now - frameCounter.current.startedAt;
        if (elapsed > 1000) {
          setFps(Math.round((frameCounter.current.count * 1000) / elapsed));
          frameCounter.current = { count: 0, startedAt: now };
        }
      } catch {
        return;
      }
    });

    ws.addEventListener("close", () => {
      setConnected(false);
      setStatus("Disconnected");
    });

    ws.addEventListener("error", () => {
      setConnected(false);
      setStatus("Connection error");
    });
  }

  function disconnect() {
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
    setStatus("Disconnected");
  }

  async function startLocalSession() {
    const prompt = promptDraft.trim();
    if (!prompt) {
      setSessionStatus("Write a prompt first.");
      return;
    }
    setIsSavingIntent(true);
    setSessionStatus("Starting local backend...");
    try {
      const response = await fetch("/api/session-intent", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const json = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(json.error || "Failed to save session intent");
      const intent = json as SessionIntentResponse;
      const acceptedPrompt = intent.prompt || prompt;
      const mock = makeMockSession(acceptedPrompt);
      setSessionIntent(intent);
      setSubmittedPrompt(acceptedPrompt);
      setPromptDraft("");
      setFrame(mock.frame);
      setFrameSrc(mock.candidateSrc);
      setFps(24);
      setShowBrainActivity(true);
      setSessionStatus(intent.backend_session.pid ? `api_server.py accepted prompt · pid ${intent.backend_session.pid}` : "api_server.py accepted prompt");
    } catch (error) {
      setSessionStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsSavingIntent(false);
    }
  }

  const hasSession = Boolean(sessionIntent);
  const reward = frame?.eeg_features?.faa.reward ?? frame?.reward_estimate ?? 0;
  const eegMode = connected && frame?.eeg_features ? "live" : frame?.eeg_features ? "mock" : "fallback";
  const previewBadge = connected ? "live" : frame ? "mock" : "idle";

  return (
    <main className="min-h-screen overflow-hidden">
      <nav className="fixed inset-x-0 top-0 z-20">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <div className="flex items-center gap-2 text-lg font-semibold">
            <Brain className="h-5 w-5" />
            NEURIM
            {hasSession ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : null}
          </div>
          {hasSession ? (
            <div className="rounded-full bg-muted px-4 py-2 text-sm text-muted-foreground">
              Local session · {sessionIntent?.session_id.slice(0, 8)}
            </div>
          ) : null}
        </div>
      </nav>

      {!hasSession ? (
        <section className="mx-auto flex min-h-screen max-w-6xl flex-col items-center justify-center px-4 py-24 text-center sm:px-6">
          <h1 className="max-w-5xl text-balance text-5xl font-semibold leading-[1.02] sm:text-6xl lg:text-7xl">
            Build an image with your brain signal
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-muted-foreground">
            Describe the visual direction. A local backend session starts after submit.
          </p>
          <PromptComposer
            value={promptDraft}
            onChange={setPromptDraft}
            onSubmit={startLocalSession}
            isSubmitting={isSavingIntent}
            examples={examplePrompts}
            mode="hero"
          />
          <p className="mt-4 text-sm text-muted-foreground">{sessionStatus}</p>
        </section>
      ) : (
        <section className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 pb-40 pt-24 sm:px-6">
          <div className="flex justify-end">
            <div className="max-w-[520px] rounded-[22px] bg-muted px-5 py-4 text-left text-base shadow-sm">
              {submittedPrompt}
            </div>
          </div>

          <div className="mt-16 max-w-3xl">
            <div className="mb-4 text-sm text-muted-foreground">{sessionStatus}</div>
            <h2 className="text-3xl font-semibold tracking-normal">Prompt sent to local backend.</h2>
            <p className="mt-4 max-w-2xl text-base leading-7 text-muted-foreground">
              api_server.py accepted the prompt and started the local session.
            </p>
          </div>

          <section className="grid grid-rows-[1fr] transition-[grid-template-rows,opacity,transform] duration-700 ease-out">
            <div className="overflow-hidden">
              <div className="mt-10 rounded-[30px] border bg-card p-4 shadow-[0_24px_90px_rgba(16,24,40,0.12)]">
                <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <div className="text-sm font-semibold">Live build preview</div>
                    <div className="text-sm text-muted-foreground">Local prompt accepted · brain activity and candidate frame</div>
                  </div>
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                    <Badge variant={connected ? "default" : "outline"} className="w-fit gap-1 rounded-full">
                      {connected ? <Wifi className="h-3.5 w-3.5" /> : <Unplug className="h-3.5 w-3.5" />}
                      {status}
                    </Badge>
                    <Input value={url} onChange={(event) => setUrl(event.target.value)} disabled={connected} className="h-10 bg-card font-mono sm:w-[250px]" />
                    {connected ? (
                      <Button variant="destructive" onClick={disconnect} className="h-10 shrink-0">
                        <CircleStop className="h-4 w-4" />
                        Disconnect
                      </Button>
                    ) : (
                      <Button onClick={connect} className="h-10 shrink-0">
                        <Radio className="h-4 w-4" />
                        Connect
                      </Button>
                    )}
                    <Button
                      type="button"
                      variant={showBrainActivity ? "default" : "secondary"}
                      onClick={() => setShowBrainActivity((value) => !value)}
                      aria-expanded={showBrainActivity}
                      className="h-10 shrink-0"
                    >
                      <Brain className="h-4 w-4" />
                      {showBrainActivity ? "Hide brain" : "Show brain"}
                    </Button>
                  </div>
                </div>

                <div className={cn("grid gap-4", showBrainActivity ? "lg:grid-cols-[1fr_1fr]" : "lg:grid-cols-1")}>
                  {showBrainActivity ? (
                    <div className="rounded-[24px] border bg-[#091013] p-3">
                      <div className="mb-3 flex items-center justify-between px-1">
                        <div>
                          <div className="text-sm font-semibold text-white">3D brain analogy</div>
                          <div className="text-xs text-white/60">Scroll or pinch to zoom</div>
                        </div>
                        <Badge variant="secondary" className="rounded-full">
                          {eegMode}
                        </Badge>
                      </div>
                      <BrainActivity3D features={frame?.eeg_features} reward={reward} className="h-[430px] border-0" />
                    </div>
                  ) : null}

                  <div className="rounded-[24px] border bg-[#0b0f14] p-3">
                    <div className="mb-3 flex items-center justify-between px-1">
                      <div>
                        <div className="text-sm font-semibold text-white">Live candidate</div>
                        <div className="text-xs text-white/60">Generated from the prompt</div>
                      </div>
                      <Badge variant="secondary" className="rounded-full">
                        {previewBadge} · {fps} fps
                      </Badge>
                    </div>
                    <div className="relative aspect-square overflow-hidden rounded-[18px] bg-black">
                      {frameSrc ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={frameSrc} alt="Live generated NEURIM frame" className="h-full w-full object-cover" />
                      ) : (
                        <div className="flex h-full items-center justify-center px-6 text-center text-sm text-white/58">
                          Waiting for generated frame.
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <div className="fixed inset-x-0 bottom-0 z-20 bg-gradient-to-t from-background via-background/95 to-transparent px-4 pb-5 pt-16">
            <PromptComposer
              value={promptDraft}
              onChange={setPromptDraft}
              onSubmit={startLocalSession}
              isSubmitting={isSavingIntent}
              mode="chat"
            />
            <p className="mx-auto mt-3 max-w-4xl text-center text-xs text-muted-foreground">
              NEURIM local sessions can be wrong. Check live headset and renderer outputs before using them.
            </p>
          </div>
        </section>
      )}
    </main>
  );
}

function PromptComposer({
  value,
  onChange,
  onSubmit,
  isSubmitting,
  examples = [],
  mode,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isSubmitting: boolean;
  examples?: string[];
  mode: "hero" | "chat";
}) {
  return (
    <form
      className={cn(
        "mx-auto w-full max-w-4xl border bg-card text-left shadow-[0_24px_90px_rgba(16,24,40,0.14)]",
        mode === "hero" ? "mt-10 rounded-[28px] p-3" : "rounded-[24px] p-4"
      )}
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={mode === "hero" ? "Ask NEURIM to build a visual concept..." : "Write a message..."}
        className={cn(
          "w-full resize-none border-0 bg-transparent outline-none placeholder:text-muted-foreground",
          mode === "hero" ? "min-h-32 rounded-[20px] px-4 py-4 text-lg leading-7" : "min-h-12 px-1 text-lg leading-7"
        )}
      />

      {mode === "hero" ? (
        <div className="flex flex-col gap-3 border-t px-2 pt-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap gap-2">
            {examples.map((prompt) => (
              <button
                key={prompt}
                type="button"
                onClick={() => onChange(prompt)}
                className="rounded-full border bg-muted px-3 py-1.5 text-xs font-medium text-muted-foreground transition hover:border-primary hover:text-foreground"
              >
                {prompt}
              </button>
            ))}
          </div>
          <Button type="submit" className="h-11 rounded-full px-5" disabled={isSubmitting}>
            {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            Build
          </Button>
        </div>
      ) : (
        <div className="flex items-center justify-between pt-2">
          <Button type="button" variant="ghost" size="icon" aria-label="Add context">
            <Plus className="h-5 w-5" />
          </Button>
          <div className="flex items-center gap-3">
            <button type="button" className="hidden items-center gap-1 text-sm font-medium text-muted-foreground sm:flex">
              NEURIM local
              <ChevronDown className="h-4 w-4" />
            </button>
            <Button type="button" variant="ghost" size="icon" aria-label="Settings">
              <SlidersHorizontal className="h-5 w-5" />
            </Button>
            <Button type="button" variant="ghost" size="icon" aria-label="Voice">
              <Mic className="h-5 w-5" />
            </Button>
            <Button type="button" variant="ghost" size="icon" aria-label="Audio">
              <AudioLines className="h-5 w-5" />
            </Button>
            <Button type="submit" size="icon" aria-label="Send prompt" disabled={isSubmitting}>
              {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      )}
    </form>
  );
}
