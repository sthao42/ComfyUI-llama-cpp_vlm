import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-llama-cpp_vlm.DynamicImageInputs",
    async nodeCreated(node) {
        if (node.comfyClass === "llama_cpp_instruct_adv" || node.comfyClass === "llama_cpp_instruct") {
            const MAX_IMAGES = 8;
            
            function updateImageInputs() {
                if (!node.inputs) return;
                
                let lastConnectedIdx = 0;
                for (let i = 0; i < node.inputs.length; i++) {
                    const input = node.inputs[i];
                    if (input && input.name && input.name.startsWith("image_")) {
                        const num = parseInt(input.name.replace("image_", ""), 10);
                        if (!isNaN(num) && input.link != null) {
                            if (num > lastConnectedIdx) {
                                lastConnectedIdx = num;
                            }
                        }
                    }
                }
                
                const maxVisibleIdx = Math.min(MAX_IMAGES, Math.max(1, lastConnectedIdx + 1));
                
                // Ensure sockets up to maxVisibleIdx exist
                for (let i = 1; i <= maxVisibleIdx; i++) {
                    const inputName = `image_${i}`;
                    const existing = node.inputs.find(inp => inp && inp.name === inputName);
                    if (!existing) {
                        node.addInput(inputName, "IMAGE");
                    }
                }
                
                // Prune unlinked trailing inputs > maxVisibleIdx
                for (let i = node.inputs.length - 1; i >= 0; i--) {
                    const input = node.inputs[i];
                    if (input && input.name && input.name.startsWith("image_")) {
                        const num = parseInt(input.name.replace("image_", ""), 10);
                        if (!isNaN(num) && num > maxVisibleIdx && input.link == null) {
                            node.removeInput(i);
                        }
                    }
                }
            }

            const origOnConnectionsChange = node.onConnectionsChange;
            node.onConnectionsChange = function (type, index, connected, link_info, input_info) {
                if (origOnConnectionsChange) {
                    origOnConnectionsChange.apply(this, arguments);
                }
                updateImageInputs();
            };

            setTimeout(updateImageInputs, 50);
        }
    }
});
