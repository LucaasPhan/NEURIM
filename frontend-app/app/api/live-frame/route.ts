import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const path = join(process.cwd(), "..", "data", "processed", "live_frame.png");
  try {
    const file = await readFile(path);
    return new NextResponse(file, {
      headers: { "content-type": "image/png", "cache-control": "no-store" },
    });
  } catch {
    return NextResponse.json({ error: "live_frame.png not found" }, { status: 404 });
  }
}
