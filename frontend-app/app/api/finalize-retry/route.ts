import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const DEFAULT_API_URL = "http://127.0.0.1:8000";

export async function POST() {
  const apiBase = (process.env.NEURIM_API_URL?.trim().replace(/\/$/, "") || DEFAULT_API_URL);
  try {
    const response = await fetch(`${apiBase}/session/finalize/retry`, {
      method: "POST",
      cache: "no-store",
    });
    const json = await response.json().catch(() => ({}));
    return NextResponse.json(json, { status: response.status });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: `Could not reach api_server.py: ${message}` }, { status: 502 });
  }
}
