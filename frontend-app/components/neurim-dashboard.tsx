"use client";

import dynamic from "next/dynamic";
import { Activity, Brain, CircleStop, Gauge, Radio, Sparkles, Unplug, Wifi } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const BrainActivity3D = dynamic(
  () => import("@/components/brain-activity-3d").then((mod) => mod.BrainActivity3D),
  { ssr: false, loading: () => <Skeleton className="h-[330px] rounded-md" /> }
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

const stateLabels: Record<string, string> = {
  calibrate: "Calibrating",
  explore: "Exploring",
  refine: "Refining",
  settle: "Locked",
  recover: "Recovering",
};

function decodeFrameSrc(msg: FrameMessage) {
  return `data:image/${msg.format || "jpeg"};base64,${msg.frame_b64}`;
}

function rewardToPercent(value: number) {
  return Math.max(0, Math.min(100, ((value + 1) / 2) * 100));
}

function formatMs(value: number | null) {
  return value == null ? "n/a" : `${Math.round(value)} ms`;
}

function Sparkline({ values }: { values: number[] }) {
  const points = useMemo(() => {
    if (values.length < 2) return "";
    return values
      .map((value, index) => {
        const x = (index / Math.max(1, values.length - 1)) * 100;
        const y = 36 - ((Math.max(-1, Math.min(1, value)) + 1) / 2) * 32;
        return `${x},${y}`;
      })
      .join(" ");
  }, [values]);

  return (
    <svg viewBox="0 0 100 40" className="h-24 w-full overflow-visible">
      <line x1="0" x2="100" y1="20" y2="20" className="stroke-border" strokeWidth="0.7" />
      <line x1="0" x2="100" y1="7.2" y2="7.2" className="stroke-muted-foreground/40" strokeDasharray="2 2" strokeWidth="0.7" />
      {points ? <polyline points={points} fill="none" className="stroke-primary" strokeWidth="2.2" strokeLinecap="round" /> : null}
    </svg>
  );
}

function FaaRewardBar({ reward, raw }: { reward: number; raw?: number | null }) {
  const pct = rewardToPercent(reward);
  const positive = reward >= 0;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>-1 avoid</span>
        <span>0</span>
        <span>+1 approach</span>
      </div>
      <div className="relative">
        <Progress value={pct} className="h-3" />
        <div className="absolute left-1/2 top-[-5px] h-6 w-px bg-foreground/50" />
        <div className="absolute top-[-4px] h-5 w-1 rounded-full bg-white shadow" style={{ left: `calc(${pct}% - 2px)` }} />
      </div>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-md border bg-muted/40 p-3">
          <div className="text-xs text-muted-foreground">Reward</div>
          <div className={cn("font-mono text-xl font-semibold", positive ? "text-emerald-300" : "text-sky-300")}>
            {reward.toFixed(3)}
          </div>
        </div>
        <div className="rounded-md border bg-muted/40 p-3">
          <div className="text-xs text-muted-foreground">Raw FAA</div>
          <div className="font-mono text-xl font-semibold">{raw == null ? "n/a" : raw.toFixed(3)}</div>
        </div>
      </div>
    </div>
  );
}

function ZMeters({ z }: { z: number[] }) {
  return (
    <div className="space-y-2">
      {z.slice(0, 10).map((value, index) => {
        const bounded = Math.max(-1, Math.min(1, value));
        return (
          <div key={index} className="grid grid-cols-[42px_1fr_54px] items-center gap-3 text-xs">
            <span className="font-mono text-muted-foreground">z{index}</span>
            <div className="relative h-2 rounded-full bg-muted">
              <div className="absolute left-1/2 top-[-3px] h-4 w-px bg-border" />
              <div
                className={cn("absolute top-0 h-2 rounded-full", bounded >= 0 ? "bg-primary" : "bg-sky-400")}
                style={
                  bounded >= 0
                    ? { left: "50%", width: `${bounded * 50}%` }
                    : { left: `${50 + bounded * 50}%`, width: `${-bounded * 50}%` }
                }
              />
            </div>
            <span className="text-right font-mono text-muted-foreground">{bounded.toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}

export function NeurimDashboard() {
  const [url, setUrl] = useState("ws://localhost:8765");
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState("Not connected");
  const [frame, setFrame] = useState<FrameMessage | null>(null);
  const [frameSrc, setFrameSrc] = useState<string | null>(null);
  const [targetSrc, setTargetSrc] = useState<string | null>(null);
  const [targetMissing, setTargetMissing] = useState(false);
  const [rewardLog, setRewardLog] = useState<number[]>([]);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [fps, setFps] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const frameCounter = useRef({ count: 0, startedAt: 0 });

  useEffect(() => {
    return () => wsRef.current?.close();
  }, []);

  useEffect(() => {
    let cancelled = false;

    function refreshTarget() {
      const probe = new Image();
      const src = `/api/target-frame?t=${Date.now()}`;
      probe.onload = () => {
        if (!cancelled) {
          setTargetSrc(src);
          setTargetMissing(false);
        }
      };
      probe.onerror = () => {
        if (!cancelled) setTargetMissing(true);
      };
      probe.src = src;
    }

    refreshTarget();
    const id = window.setInterval(refreshTarget, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
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
        setLatencyMs(now - msg.t * 1000);
        setRewardLog((prev) => [...prev.slice(-79), msg.reward_estimate]);
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

  const reward = frame?.eeg_features?.faa.reward ?? frame?.reward_estimate ?? 0;
  const rawFaa = frame?.eeg_features?.faa.raw ?? null;
  const state = frame?.state ?? "calibrate";
  const hasRealEeg = Boolean(frame?.eeg_features?.channels?.length);

  return (
    <TooltipProvider>
      <main className="mx-auto flex min-h-screen max-w-[1500px] flex-col gap-5 px-5 py-5 lg:px-7">
        <header className="flex flex-col gap-4 border-b pb-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="mb-2 flex items-center gap-2">
              <Badge variant="secondary" className="gap-1">
                <Brain className="h-3.5 w-3.5" />
                NEURIM
              </Badge>
              <Badge variant={connected ? "default" : "outline"} className="gap-1">
                {connected ? <Wifi className="h-3.5 w-3.5" /> : <Unplug className="h-3.5 w-3.5" />}
                {status}
              </Badge>
            </div>
            <h1 className="text-2xl font-semibold tracking-normal lg:text-3xl">EEG reward control surface</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-muted-foreground">
              Live generated frames, FAA reward, optimizer telemetry, and 3D EEG activity from the NEURIM hub.
            </p>
          </div>

          <div className="flex w-full gap-2 lg:w-[520px]">
            <Input value={url} onChange={(event) => setUrl(event.target.value)} disabled={connected} className="font-mono" />
            {connected ? (
              <Button variant="destructive" onClick={disconnect}>
                <CircleStop className="h-4 w-4" />
                Disconnect
              </Button>
            ) : (
              <Button onClick={connect}>
                <Radio className="h-4 w-4" />
                Connect
              </Button>
            )}
          </div>
        </header>

        <section className="grid gap-5 xl:grid-cols-[minmax(0,1.25fr)_minmax(420px,0.75fr)]">
          <Card>
            <CardHeader className="flex-row items-center justify-between space-y-0">
              <div>
                <CardTitle>Target comparison</CardTitle>
                <CardDescription>Hidden target beside the live candidate for real-time comparison</CardDescription>
              </div>
              <Badge variant="outline" className="font-mono">
                {frame?.format?.toUpperCase() ?? "NO FRAME"} · {fps} fps
              </Badge>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 lg:grid-cols-2">
                <div>
                  <div className="mb-2 flex items-center justify-between">
                    <span className="font-mono text-xs uppercase tracking-wide text-muted-foreground">Target</span>
                    <Badge variant={targetMissing ? "outline" : "secondary"}>{targetMissing ? "missing" : "file"}</Badge>
                  </div>
                  <div className="relative aspect-square overflow-hidden rounded-md border bg-black">
                    {targetSrc ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={targetSrc} alt="Target frame" className="h-full w-full object-contain" />
                    ) : (
                      <div className="flex h-full items-center justify-center px-6 text-center text-sm text-muted-foreground">
                        Run scripted fake loop to create data/processed/target_frame.png
                      </div>
                    )}
                  </div>
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between">
                    <span className="font-mono text-xs uppercase tracking-wide text-muted-foreground">Live candidate</span>
                    <Badge variant={connected ? "default" : "outline"}>{connected ? "live" : "stale"}</Badge>
                  </div>
                  <div className="relative aspect-square overflow-hidden rounded-md border bg-black">
                    {frameSrc ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={frameSrc} alt="Live generated NEURIM frame" className="h-full w-full object-contain" />
                    ) : (
                      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                        Connect to the hub to receive frames
                      </div>
                    )}
                    <div className="absolute right-3 top-3 rounded-md border bg-background/80 px-2 py-1 font-mono text-xs">
                      step {frame?.step_index ?? "—"}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="space-y-5">
            <Card>
              <CardHeader className="flex-row items-center justify-between space-y-0">
                <div>
                  <CardTitle>FAA reward</CardTitle>
                  <CardDescription>Frontal alpha asymmetry signal</CardDescription>
                </div>
                <Badge className="capitalize" variant={state === "recover" ? "destructive" : "secondary"}>
                  {stateLabels[state] ?? state}
                </Badge>
              </CardHeader>
              <CardContent>
                <FaaRewardBar reward={reward} raw={rawFaa} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Brain activity map</CardTitle>
                <CardDescription>
                  {hasRealEeg ? "Driven by EEG alpha-band power" : "Synthetic fallback until EEG features arrive"}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <BrainActivity3D features={frame?.eeg_features} reward={reward} />
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="rounded-md border bg-muted/30 p-2">
                    <div className="text-muted-foreground">Channels</div>
                    <div className="font-mono text-lg">{frame?.eeg_features?.channels.length ?? 14}</div>
                  </div>
                  <div className="rounded-md border bg-muted/30 p-2">
                    <div className="text-muted-foreground">Latency</div>
                    <div className="font-mono text-lg">{formatMs(latencyMs)}</div>
                  </div>
                  <div className="rounded-md border bg-muted/30 p-2">
                    <div className="text-muted-foreground">State</div>
                    <div className="truncate font-mono text-lg">{state}</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        <section className="grid gap-5 lg:grid-cols-[0.8fr_1.2fr]">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Gauge className="h-4 w-4 text-primary" />
                Reward history
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Sparkline values={rewardLog} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-primary" />
                Optimizer telemetry
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="latent">
                <TabsList>
                  <TabsTrigger value="latent">Latent vector</TabsTrigger>
                  <TabsTrigger value="stream">Stream</TabsTrigger>
                </TabsList>
                <TabsContent value="latent">
                  <ZMeters z={frame?.z ?? []} />
                </TabsContent>
                <TabsContent value="stream">
                  <div className="grid gap-3 text-sm sm:grid-cols-4">
                    <Metric label="Step" value={String(frame?.step_index ?? "—")} />
                    <Metric label="Reward" value={reward.toFixed(3)} />
                    <Metric label="Format" value={frame?.format?.toUpperCase() ?? "—"} />
                    <Metric label="Dimensions" value={frame?.z ? `${frame.z.length}D` : "—"} />
                  </div>
                  <Separator className="my-4" />
                  <Alert>
                    <Sparkles className="mr-2 inline h-4 w-4 text-primary" />
                    <AlertTitle className="inline">Compatibility mode</AlertTitle>
                    <AlertDescription className="mt-1">
                      The dashboard accepts older frame messages without EEG features and switches the 3D panel to a synthetic fallback.
                    </AlertDescription>
                  </Alert>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </section>

        <footer className="pb-4 text-xs text-muted-foreground">
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="cursor-help underline decoration-dotted">Run the hub</span>
            </TooltipTrigger>
            <TooltipContent>python scripts/run_demo.py --serve --mock</TooltipContent>
          </Tooltip>{" "}
          and connect this dashboard to <code>ws://localhost:8765</code>.
        </footer>
      </main>
    </TooltipProvider>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border bg-muted/30 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 font-mono text-lg font-semibold">{value}</div>
    </div>
  );
}
