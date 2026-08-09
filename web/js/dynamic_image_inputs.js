import { app } from "../../scripts/app.js";

/**
 * ComfyUI Extension for Llama-cpp VLM Node Suite
 * Controls multi-image socket dynamics and seed state tracking.
 */
app.registerExtension({
    name: "ComfyUI-llama-cpp_vlm.SeedControl",
    async init() {
        console.log("[llama-cpp_vlm] Multi-image & seed control extensions initialized successfully.");
    },
    async nodeCreated(node) {
        if (!node || !node.comfyClass) return;
        if (node.comfyClass === "llama_cpp_instruct_adv") {
            // Extension hook for instruct node dynamic inputs
        }
    }
});
