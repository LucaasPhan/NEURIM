import { NextResponse } from "next/server";
import WebSocket from "ws";

type ApplyRequest = {
  anchor_prompts?: unknown;
  remote_diffusion_url?: string;
  hub_ws_url?: string;
};

function normalizePrompts(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item).trim()).filter(Boolean);
}

async function applyRemote(remoteUrl: string, prompts: string[]) {
  const base = remoteUrl.replace(/\/$/, "");
  const response = await fetch(`${base}/anchors`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ anchor_prompts: prompts }),
  });
  const json = await response.json().catch(() => ({}));
  if (!response.ok || json.ok === false) {
    throw new Error(json.error || `remote diffusion returned HTTP ${response.status}`);
  }
  return json;
}

async function applyHub(hubUrl: string, prompts: string[]) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(hubUrl);
    const timeout = setTimeout(() => {
      ws.close();
      reject(new Error("hub control request timed out"));
    }, 8000);

    ws.on("open", () => {
      ws.send(JSON.stringify({ role: "control" }));
      ws.send(JSON.stringify({ command: "set_anchor_prompts", args: { anchor_prompts: prompts } }));
    });
    ws.on("message", (data) => {
      clearTimeout(timeout);
      ws.close();
      try {
        const json = JSON.parse(data.toString());
        if (json.ok === false) reject(new Error(json.error || "hub rejected anchor prompts"));
        else resolve(json);
      } catch {
        reject(new Error("hub returned invalid JSON"));
      }
    });
    ws.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
  });
}

export async function POST(request: Request) {
  let body: ApplyRequest;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON request body" }, { status: 400 });
  }

  const prompts = normalizePrompts(body.anchor_prompts);
  if (prompts.length !== 10) {
    return NextResponse.json({ error: "Exactly 10 anchor_prompts are required" }, { status: 400 });
  }

  const remoteUrl = body.remote_diffusion_url?.trim();
  const hubUrl = body.hub_ws_url?.trim();
  const result: Record<string, { ok: boolean; error?: string }> = {};

  if (remoteUrl) {
    try {
      await applyRemote(remoteUrl, prompts);
      result.remote = { ok: true };
    } catch (error) {
      result.remote = { ok: false, error: error instanceof Error ? error.message : String(error) };
    }
  }

  if (hubUrl) {
    try {
      await applyHub(hubUrl, prompts);
      result.hub = { ok: true };
    } catch (error) {
      result.hub = { ok: false, error: error instanceof Error ? error.message : String(error) };
    }
  }

  if (!remoteUrl && !hubUrl) {
    return NextResponse.json({ error: "At least one target URL is required" }, { status: 400 });
  }

  const ok = Object.values(result).every((item) => item.ok);
  return NextResponse.json({ ok, targets: result }, { status: ok ? 200 : 207 });
}
