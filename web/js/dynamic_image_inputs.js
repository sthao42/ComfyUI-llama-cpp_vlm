import { app } from "../../scripts/app.js";

// Native static socket registration handled directly via Python INPUT_TYPES optional dictionary.
app.registerExtension({
    name: "ComfyUI-llama-cpp_vlm.DynamicImageInputs",
    async init() {
        console.log("[llama-cpp_vlm] Multi-image & multi-video reference sockets initialized!");
    }
});
