import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-llama-cpp_vlm.SeedControl",
    async init() {
        console.log("[llama-cpp_vlm] Multi-image & seed control extensions initialized!");
    },
    nodeCreated(node) {
        if (node.comfyClass === "llama_cpp_instruct_adv") {
            const seedWidget = node.widgets?.find(w => w.name === "seed");
            const controlWidget = node.widgets?.find(w => w.name === "control_after_generate");
            
            if (seedWidget && controlWidget) {
                const origCallback = controlWidget.callback;
                controlWidget.callback = function (value) {
                    if (origCallback) origCallback.apply(this, arguments);
                    node.setDirtyCanvas(true, true);
                };
            }
        }
    }
});
