import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const DEFAULT_API_URL = "http://127.0.0.1:8000";

function cleanUrl(value: string | undefined, fallback: string) {
  const cleaned = value?.trim().replace(/\/$/, "");
  return cleaned || fallback;
}

// Proxies the backend api_server.py /session/status so the display can tell when
// the optimizer run has finished and the finalized target frame is ready.
export async function GET() {
  const apiBase = cleanUrl(process.env.NEURIM_API_URL, DEFAULT_API_URL);
  try {
    const response = await fetch(`${apiBase}/session/status`, { cache: "no-store" });
    const json = (await response.json().catch(() => ({}))) as Record<string, unknown>;
    if (!response.ok) {
      return NextResponse.json({ error: `api_server.py returned HTTP ${response.status}` }, { status: response.status });
    }
    return NextResponse.json(json);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: `Could not reach api_server.py at ${apiBase}: ${message}` }, { status: 502 });
  }
}
