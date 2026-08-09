import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-llama-cpp_vlm.SeedControl",
    async init() {
        console.log("[llama-cpp_vlm] Multi-image & seed control extensions initialized!");
    }
});

