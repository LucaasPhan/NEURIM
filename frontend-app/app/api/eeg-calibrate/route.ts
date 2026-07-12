import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function POST() {
  const apiBase = (process.env.NEURIM_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
  try {
    const response = await fetch(`${apiBase}/eeg/calibrate`, { method: "POST", cache: "no-store" });
    return NextResponse.json(await response.json().catch(() => ({})), { status: response.status });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 502 });
  }
}
