import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-llama-cpp_vlm.DynamicImageInputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "llama_cpp_instruct_adv" || nodeData.name === "llama_cpp_instruct") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const me = onNodeCreated?.apply(this, arguments);
                const MAX_IMAGES_INDEX = 7; // Total 8 sockets: image_0 to image_7
                
                const updateImageInputs = () => {
                    if (!this.inputs) return;
                    
                    let maxConnectedNum = -1;
                    for (const input of this.inputs) {
                        if (input.name && input.name.startsWith("image_")) {
                            const num = parseInt(input.name.replace("image_", ""), 10);
                            if (!isNaN(num) && input.link !== null) {
                                if (num > maxConnectedNum) {
                                    maxConnectedNum = num;
                                }
                            }
                        }
                    }
                    
                    const targetMax = Math.min(MAX_IMAGES_INDEX, Math.max(0, maxConnectedNum + 1));
                    
                    // Add missing sockets up to targetMax (image_0 .. image_targetMax)
                    for (let i = 0; i <= targetMax; i++) {
                        const inputName = `image_${i}`;
                        const existing = this.inputs.find(inp => inp && inp.name === inputName);
                        if (!existing) {
                            this.addInput(inputName, "IMAGE");
                        }
                    }
                    
                    // Remove unlinked trailing inputs greater than targetMax
                    for (let i = this.inputs.length - 1; i >= 0; i--) {
                        const input = this.inputs[i];
                        if (input && input.name && input.name.startsWith("image_")) {
                            const num = parseInt(input.name.replace("image_", ""), 10);
                            if (!isNaN(num) && num > targetMax && input.link === null) {
                                this.removeInput(i);
                            }
                        }
                    }
                };

                const origOnConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function (type, index, connected, link_info, input_info) {
                    if (origOnConnectionsChange) {
                        origOnConnectionsChange.apply(this, arguments);
                    }
                    updateImageInputs();
                };

                setTimeout(() => updateImageInputs(), 50);
                return me;
            };
        }
    }
});
