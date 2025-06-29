import { promises as fs } from "fs";
import path from "path";

export async function GET(req, { params }) {
  const { id } = await params;

  try {
    const raw = await fs.readFile(
      path.join(process.cwd(), "StoredSessions", `${id}.json`),
      "utf8",
    );
    // Return the raw JSON file contents
    return new Response(raw, {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return new Response(JSON.stringify({ error: "Not found" }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  }
}
