import { promises as fs } from "fs";
import path from "path";
import { randomBytes } from "crypto";

const SESSIONS_DIR = path.join(process.cwd(), "StoredSessions");

export async function POST(req) {
  // Ensure the “StoredSessions” folder exists
  await fs.mkdir(SESSIONS_DIR, { recursive: true });

  // Generate a unique 6‐hex ID
  let id;
  do {
    id = randomBytes(3).toString("hex"); // e.g. “a1b2c3”
  } while (
    await fs
      .access(path.join(SESSIONS_DIR, `${id}.json`))
      .then(() => true)
      .catch(() => false)
  );

  // Grab the JSON body from the request
  const body = await req.json();
  // (Expecting { sources: […], activeSourceId: N } from the client.)

  // Write it to StoredSessions/<id>.json
  await fs.writeFile(
    path.join(SESSIONS_DIR, `${id}.json`),
    JSON.stringify(body, null, 2),
    "utf8",
  );

  // Return the new ID to the client
  return new Response(JSON.stringify({ id }), { status: 201 });
}
