// scripts/rebuild-ove.cjs
const { build } = require("esbuild");
const cssModulesPlugin = require("esbuild-css-modules-plugin");
const fs = require("node:fs");
const path = require("node:path");

(async () => {
  const entryPath = require.resolve("@teselagen/ove"); // exported file
  const pkgRoot = path.dirname(entryPath); // package root

  const { dependencies = {}, peerDependencies = {} } = JSON.parse(
    fs.readFileSync(path.join(pkgRoot, "package.json"), "utf8")
  );

  console.log("⏳ rebuilding @teselagen/ove/index.es.js …");

  await build({
    entryPoints: [path.join(pkgRoot, "src/index.js")],
    outfile: path.join(pkgRoot, "index.es.js"), // ✅ overwrite the file that gets imported
    bundle: true,
    banner: {
      js: `import "./index.es.css";`,
    },
    format: "esm",
    platform: "browser",
    packages: "external",
    external: Object.keys({ ...dependencies, ...peerDependencies }),
    loader: { ".js": "jsx", ".scss": "css" },
    jsx: "automatic",
    target: ["es2018"],
    plugins: [
      cssModulesPlugin({
        filter: /\.module\.(css|scss)$/, // leave ordinary .css/.scss untouched
      }),
    ],
  });

  console.log("✅  bundle rebuilt");
})();
