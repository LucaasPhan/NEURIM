import { NextResponse } from "next/server";
import OpenAI from "openai";

type GenerateRequest = {
  desired_prompt?: string;
};

function normalizePrompts(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item).trim()).filter(Boolean).slice(0, 10);
}

function normalizeAxes(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item).trim()).filter(Boolean).slice(0, 5);
}

export async function POST(request: Request) {
  let body: GenerateRequest;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON request body" }, { status: 400 });
  }

  const desired = body.desired_prompt?.trim();
  if (!desired) {
    return NextResponse.json({ error: "desired_prompt is required" }, { status: 400 });
  }

  if (!process.env.OPENAI_API_KEY) {
    return NextResponse.json({ error: "OPENAI_API_KEY is not configured for frontend-app" }, { status: 500 });
  }

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const model = process.env.OPENAI_PROMPT_MODEL ?? "gpt-5.5";

  try {
    const metaPrompt = `You are generating anchor prompts for a real-time latent morphing image system.

User prompt:
“${desired}”

Task:
Create a bank of anchor prompts based on the user prompt. These prompts will be embedded, reduced with PCA, and
used for smooth latent-space interpolation. The anchor prompts must preserve the core subject and intent of the
user prompt while spanning meaningful visual variation.

Rules:

Keep the same subject, setting, camera distance, composition, lighting, and style across all anchor prompts.
Vary only 1-2 controlled visual attributes per prompt.
Do not introduce new objects, new scenes, new characters, or different art styles unless the user prompt
explicitly asks for them.
Make the prompts different enough to create useful latent directions, but similar enough to morph smoothly.
Use a consistent sentence scaffold for every prompt.
If the user prompt is vague, infer a stable visual scaffold and vary safe visual attributes such as color,
material, texture, pose, expression, shape, density, or lighting intensity.
Return exactly 10 anchor prompts.
Each anchor prompt must be one sentence.
Also return a short list of the controlled variation axes.

Output format:
Controlled axes:

axis 1
axis 2
axis 3

Anchor prompts:

...
...
...
...
...
...
...
...
...
...`;

    const response = await client.responses.create({
      model,
      input: [
        {
          role: "system",
          content:
            "You are the NEURIM anchor-prompt generation agent. Follow the user-provided meta-prompt exactly. Return machine-readable JSON matching the supplied schema while preserving the requested controlled-axes and anchor-prompts content.",
        },
        { role: "user", content: metaPrompt },
      ],
      text: {
        format: {
          type: "json_schema",
          name: "anchor_prompt_set",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            required: ["controlled_axes", "anchor_prompts"],
            properties: {
              controlled_axes: {
                type: "array",
                minItems: 1,
                maxItems: 5,
                items: {
                  type: "string",
                  minLength: 3,
                },
              },
              anchor_prompts: {
                type: "array",
                minItems: 10,
                maxItems: 10,
                items: {
                  type: "string",
                  minLength: 12,
                },
              },
            },
          },
        },
      },
    });

    const parsed = JSON.parse(response.output_text || "{}") as { controlled_axes?: unknown; anchor_prompts?: unknown };
    const controlledAxes = normalizeAxes(parsed.controlled_axes);
    const anchorPrompts = normalizePrompts(parsed.anchor_prompts);
    if (anchorPrompts.length !== 10) {
      return NextResponse.json({ error: "OpenAI did not return exactly 10 anchor prompts" }, { status: 502 });
    }
    return NextResponse.json({ controlled_axes: controlledAxes, anchor_prompts: anchorPrompts, model });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
