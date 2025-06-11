const { execSync } = require("child_process");

const mode = process.env.NODE_ENV;

if (mode === "development") {
  console.log("Running in development mode...");
  execSync("npm run dev:all", { stdio: "inherit" });
} else {
  console.log("Running in production mode...");
  execSync("npm run start:all", { stdio: "inherit" });
}
