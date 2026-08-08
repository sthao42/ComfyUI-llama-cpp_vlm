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
                    if (!this.inputs || this._configuring) return;
                    
                    let maxConnectedNum = -1;
                    for (const input of this.inputs) {
                        if (input && input.name && input.name.startsWith("image_")) {
                            const num = parseInt(input.name.replace("image_", ""), 10);
                            if (!isNaN(num)) {
                                const isConnected = input.link != null && input.link !== undefined;
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
                        const existing = this.inputs.find(inp => inp && inp.name === inputName);
                        if (!existing) {
                            // Find queue_handler index to insert image_X above queue_handler
                            const queueIdx = this.inputs.findIndex(inp => inp && inp.name === "queue_handler");
                            if (queueIdx !== -1) {
                                this.addInput(inputName, "IMAGE");
                                // Move the newly added input above queue_handler
                                const newInput = this.inputs.pop();
                                this.inputs.splice(queueIdx, 0, newInput);
                            } else {
                                this.addInput(inputName, "IMAGE");
                            }
                        }
                    }
                    
                    // Remove unlinked trailing inputs greater than targetMax
                    for (let i = this.inputs.length - 1; i >= 0; i--) {
                        const input = this.inputs[i];
                        if (input && input.name && input.name.startsWith("image_")) {
                            const num = parseInt(input.name.replace("image_", ""), 10);
                            if (!isNaN(num) && num > targetMax && (input.link == null || input.link === undefined)) {
                                this.removeInput(i);
                            }
                        }
                    }
                    
                    app.graph?.setDirtyCanvas(true, true);
                };

                const origOnConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function (type, index, connected, link_info, input_info) {
                    if (origOnConnectionsChange) {
                        origOnConnectionsChange.apply(this, arguments);
                    }
                    setTimeout(() => updateImageInputs(), 20);
                };

                const origOnConnectInput = this.onConnectInput;
                this.onConnectInput = function () {
                    if (origOnConnectInput) {
                        origOnConnectInput.apply(this, arguments);
                    }
                    setTimeout(() => updateImageInputs(), 20);
                };

                const origOnDisconnectInput = this.onDisconnectInput;
                this.onDisconnectInput = function () {
                    if (origOnDisconnectInput) {
                        origOnDisconnectInput.apply(this, arguments);
                    }
                    setTimeout(() => updateImageInputs(), 20);
                };

                const origOnConfigure = this.onConfigure;
                this.onConfigure = function () {
                    this._configuring = true;
                    if (origOnConfigure) {
                        origOnConfigure.apply(this, arguments);
                    }
                    this._configuring = false;
                    setTimeout(() => updateImageInputs(), 100);
                };

                setTimeout(() => updateImageInputs(), 50);
                return me;
            };
        }
    }
});
