import { app } from "../../scripts/app.js";

const TARGET_CLASSES = new Set([
    "llama_cpp_instruct_adv",
    "Llama-cpp Instruct",
    "Llama-cpp Instruct (Advanced)"
]);

const MAX_SOCKETS = 9;

function isTargetNode(nodeOrData) {
    const className = nodeOrData?.comfyClass || nodeOrData?.type || nodeOrData?.name || nodeOrData?.constructor?.comfyClass || nodeOrData?.constructor?.type;
    return TARGET_CLASSES.has(className);
}

function isDynamicInput(input, prefix) {
    return Boolean(input?.name && String(input.name).startsWith(prefix));
}

function getDynamicInputs(node, prefix) {
    return (node.inputs || []).filter(inp => isDynamicInput(inp, prefix));
}

function renumberDynamicInputs(node, prefix) {
    let nextIndex = 0;
    for (const input of node.inputs || []) {
        if (!isDynamicInput(input, prefix)) {
            continue;
        }
        input.name = `${prefix}${nextIndex}`;
        input.label = input.name;
        nextIndex += 1;
    }
}

function findInsertionIndex(node, prefix) {
    const inputs = node.inputs || [];
    let lastMatchIdx = -1;
    for (let i = 0; i < inputs.length; i += 1) {
        if (isDynamicInput(inputs[i], prefix)) {
            lastMatchIdx = i;
        }
    }
    if (lastMatchIdx !== -1) {
        return lastMatchIdx + 1;
    }

    const anchorIdx = inputs.findIndex(inp => inp && inp.name === "video_0");
    if (anchorIdx !== -1) {
        return anchorIdx;
    }
    return inputs.length;
}

function ensureTrailingDynamicInput(node, prefix, type) {
    const dynamicInputs = getDynamicInputs(node, prefix);
    if (dynamicInputs.length >= MAX_SOCKETS) {
        return;
    }

    if (!dynamicInputs.length || dynamicInputs[dynamicInputs.length - 1].link != null) {
        const newIndex = dynamicInputs.length;
        const inputName = `${prefix}${newIndex}`;
        const insertIdx = findInsertionIndex(node, prefix);

        node.addInput(inputName, type);
        const newInput = node.inputs.pop();
        node.inputs.splice(insertIdx, 0, newInput);
    }
}

function removeInputSafely(node, index) {
    if (!node || !node.inputs || index < 0 || index >= node.inputs.length) {
        return;
    }
    const removedInput = node.inputs[index];
    if (removedInput.link != null && node.graph) {
        node.disconnectInput(index);
    }
    node.removeInput(index);

    if (node.graph && node.graph.links) {
        for (const linkId in node.graph.links) {
            const link = node.graph.links[linkId];
            if (link && link.target_id === node.id && link.target_slot > index) {
                link.target_slot -= 1;
            }
        }
    }
}

function syncDynamicCategoryInputs(node, prefix, type) {
    if (!node._configuring) {
        const dynamicInputs = getDynamicInputs(node, prefix);
        let lastConnectedIdx = -1;
        for (let i = 0; i < dynamicInputs.length; i += 1) {
            if (dynamicInputs[i].link != null) {
                lastConnectedIdx = i;
            }
        }
        const keepCount = Math.max(1, lastConnectedIdx + 2);

        for (let i = dynamicInputs.length - 1; i >= keepCount; i -= 1) {
            const inputToRemove = dynamicInputs[i];
            const actualIndex = (node.inputs || []).indexOf(inputToRemove);
            if (actualIndex !== -1) {
                removeInputSafely(node, actualIndex);
            }
        }
    }

    renumberDynamicInputs(node, prefix);
    ensureTrailingDynamicInput(node, prefix, type);
    renumberDynamicInputs(node, prefix);
}

const LEGACY_WIDGET_ORDER = [
    "preset_prompt",
    "custom_prompt",
    "system_prompt",
    "inference_mode",
    "max_frames",
    "max_size",
    "seed",
    "control_after_generate",
    "force_offload",
    "save_states",
    "video_path"
];

function reorderVideoPathWidget(node) {
    if (!node || !node.widgets) {
        return;
    }
    const idx = node.widgets.findIndex(w => w && w.name === "video_path");
    if (idx === -1) {
        return;
    }
    const presetIdx = node.widgets.findIndex(w => w && w.name === "preset_prompt");
    if (presetIdx !== -1 && idx > presetIdx) {
        const [videoPathWidget] = node.widgets.splice(idx, 1);
        node.widgets.splice(presetIdx, 0, videoPathWidget);
        node.arrange?.();
        node.setDirtyCanvas?.(true, true);
    } else if (presetIdx === -1 && idx > 0) {
        const [videoPathWidget] = node.widgets.splice(idx, 1);
        node.widgets.unshift(videoPathWidget);
        node.arrange?.();
        node.setDirtyCanvas?.(true, true);
    }
}

function restoreLegacyWidgetsValues(node, values) {
    if (!Array.isArray(values) || !node.widgets) {
        return;
    }

    const presetWidget = node.widgets.find(w => w && w.name === "preset_prompt");
    const isLegacyValues = presetWidget?.options?.values 
        ? presetWidget.options.values.includes(values[0])
        : (typeof values[0] === "string" && !values[0].includes("/") && !values[0].includes("\\") && values[0].length > 0 && typeof values[4] === "number");

    if (!isLegacyValues) {
        return;
    }

    let order = LEGACY_WIDGET_ORDER;
    if (values.length > 7 && typeof values[7] === "boolean") {
        order = [
            "preset_prompt",
            "custom_prompt",
            "system_prompt",
            "inference_mode",
            "max_frames",
            "max_size",
            "seed",
            "force_offload",
            "save_states",
            "video_path"
        ];
    } else if (values.length === 10) {
        order = [
            "preset_prompt",
            "custom_prompt",
            "system_prompt",
            "inference_mode",
            "max_frames",
            "max_size",
            "seed",
            "control_after_generate",
            "force_offload",
            "save_states"
        ];
    } else if (values.length === 9) {
        order = [
            "preset_prompt",
            "custom_prompt",
            "system_prompt",
            "inference_mode",
            "max_frames",
            "max_size",
            "seed",
            "force_offload",
            "save_states"
        ];
    }

    const valMap = {};
    for (let i = 0; i < values.length && i < order.length; i += 1) {
        valMap[order[i]] = values[i];
    }
    for (const w of node.widgets) {
        if (w && w.name in valMap) {
            w.value = valMap[w.name];
        }
    }
}

function syncAllDynamicInputs(node) {
    if (!node || node.__syncingDynamicInputs) {
        return;
    }

    node.__syncingDynamicInputs = true;
    try {
        syncDynamicCategoryInputs(node, "image_", "IMAGE");
        syncDynamicCategoryInputs(node, "video_", "IMAGE");
        reorderVideoPathWidget(node);

        if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
            const computed = node.computeSize();
            const current = node.size || [0, 0];
            node.setSize([
                Math.max(current[0], computed[0]),
                Math.max(current[1], computed[1])
            ]);
        }
        node.arrange?.();
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas(true, true);
    } finally {
        node.__syncingDynamicInputs = false;
    }
}

app.registerExtension({
    name: "ComfyUI-llama-cpp_vlm.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNode(nodeData)) {
            return;
        }

        // Ensure video_path is in required, directly before preset_prompt in nodeData
        if (nodeData.input?.optional?.video_path) {
            const videoPathSpec = nodeData.input.optional.video_path;
            delete nodeData.input.optional.video_path;
            if (nodeData.input.required && !nodeData.input.required.video_path) {
                const newReq = {};
                for (const [k, v] of Object.entries(nodeData.input.required)) {
                    if (k === "preset_prompt") {
                        newReq.video_path = videoPathSpec;
                    }
                    newReq[k] = v;
                }
                if (!newReq.video_path) {
                    newReq.video_path = videoPathSpec;
                }
                nodeData.input.required = newReq;
            }
        } else if (nodeData.input?.required?.video_path && nodeData.input?.required?.preset_prompt) {
            const keys = Object.keys(nodeData.input.required);
            const vIdx = keys.indexOf("video_path");
            const pIdx = keys.indexOf("preset_prompt");
            if (vIdx > pIdx) {
                const videoPathSpec = nodeData.input.required.video_path;
                delete nodeData.input.required.video_path;
                const newReq = {};
                for (const [k, v] of Object.entries(nodeData.input.required)) {
                    if (k === "preset_prompt") {
                        newReq.video_path = videoPathSpec;
                    }
                    newReq[k] = v;
                }
                nodeData.input.required = newReq;
            }
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            reorderVideoPathWidget(this);
            setTimeout(() => {
                syncAllDynamicInputs(this);
            }, 50);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, index, connected, linkInfo) {
            const result = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            if (type === 2) {
                return result;
            }
            const input = this.inputs?.[index];
            if (input && (isDynamicInput(input, "image_") || isDynamicInput(input, "video_"))) {
                setTimeout(() => syncAllDynamicInputs(this), 0);
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            this._configuring = true;
            const isLegacy = info && !info._widgets_order_v2;
            const savedValues = isLegacy && Array.isArray(info.widgets_values) ? [...info.widgets_values] : null;

            reorderVideoPathWidget(this);
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (isLegacy && savedValues) {
                restoreLegacyWidgetsValues(this, savedValues);
            }

            this._configuring = false;
            setTimeout(() => syncAllDynamicInputs(this), 50);
            return result;
        };

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(o) {
            const result = onSerialize ? onSerialize.apply(this, arguments) : undefined;
            if (o) {
                o._widgets_order_v2 = true;
            }
            return result;
        };
    },
    async nodeCreated(node) {
        if (isTargetNode(node)) {
            reorderVideoPathWidget(node);
            setTimeout(() => syncAllDynamicInputs(node), 50);
        }
    }
});

