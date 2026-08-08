import { app } from "../../scripts/app.js";

console.log("[llama-cpp_vlm] Dynamic image inputs JS extension registered!");

app.registerExtension({
    name: "ComfyUI-llama-cpp_vlm.DynamicImageInputs",
    async nodeCreated(node, app) {
        if (!node) return;
        const className = node.comfyClass || node.type;
        if (className === "llama_cpp_instruct_adv" || className === "Llama-cpp Instruct") {
            console.log("[llama-cpp_vlm] Initialized dynamic image inputs for Llama-cpp Instruct node:", node.id);
            const MAX_IMAGES_INDEX = 7; // Total 8 sockets: image_0 to image_7

            const updateImageInputs = (changedSlot, isConnectedEvent) => {
                if (!node.inputs || node._configuring) return;

                let maxConnectedNum = -1;
                for (let idx = 0; idx < node.inputs.length; idx++) {
                    const input = node.inputs[idx];
                    if (input && input.name && input.name.startsWith("image_")) {
                        const num = parseInt(input.name.replace("image_", ""), 10);
                        if (!isNaN(num)) {
                            let isConnected = false;
                            if (idx === changedSlot && isConnectedEvent !== undefined) {
                                isConnected = isConnectedEvent;
                            } else if (input.link != null) {
                                const linkObj = app.graph ? (app.graph.links ? app.graph.links[input.link] : null) : null;
                                isConnected = linkObj != null;
                            }

                            if (isConnected && num > maxConnectedNum) {
                                maxConnectedNum = num;
                            }
                        }
                    }
                }

                const targetMax = Math.min(MAX_IMAGES_INDEX, Math.max(0, maxConnectedNum + 1));

                // Add missing sockets up to targetMax (image_0 .. image_targetMax)
                for (let i = 0; i <= targetMax; i++) {
                    const inputName = `image_${i}`;
                    const existing = node.inputs.find(inp => inp && inp.name === inputName);
                    if (!existing) {
                        const queueIdx = node.inputs.findIndex(inp => inp && inp.name === "queue_handler");
                        if (queueIdx !== -1) {
                            node.addInput(inputName, "IMAGE");
                            const newInput = node.inputs.pop();
                            node.inputs.splice(queueIdx, 0, newInput);
                        } else {
                            node.addInput(inputName, "IMAGE");
                        }
                        console.log(`[llama-cpp_vlm] Added socket ${inputName} to node ${node.id}`);
                    }
                }

                // Remove unlinked trailing inputs greater than targetMax
                for (let i = node.inputs.length - 1; i >= 0; i--) {
                    const input = node.inputs[i];
                    if (input && input.name && input.name.startsWith("image_")) {
                        const num = parseInt(input.name.replace("image_", ""), 10);
                        if (!isNaN(num) && num > targetMax) {
                            const isConnected = input.link != null && (app.graph?.links ? app.graph.links[input.link] != null : true);
                            if (!isConnected) {
                                node.removeInput(i);
                                console.log(`[llama-cpp_vlm] Removed unlinked socket ${input.name} from node ${node.id}`);
                            }
                        }
                    }
                }

                app.graph?.setDirtyCanvas(true, true);
            };

            const origOnConnectionsChange = node.onConnectionsChange;
            node.onConnectionsChange = function (type, index, connected, link_info, input_info) {
                if (origOnConnectionsChange) {
                    origOnConnectionsChange.apply(this, arguments);
                }
                if (type === 1) { // 1 = LiteGraph.INPUT
                    updateImageInputs(index, connected);
                    setTimeout(() => updateImageInputs(), 20);
                }
            };

            const origOnConfigure = node.onConfigure;
            node.onConfigure = function () {
                node._configuring = true;
                if (origOnConfigure) {
                    origOnConfigure.apply(this, arguments);
                }
                node._configuring = false;
                setTimeout(() => updateImageInputs(), 50);
            };

            setTimeout(() => updateImageInputs(), 50);
        }
    }
});
