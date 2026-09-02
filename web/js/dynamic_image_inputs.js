import { app } from "../../scripts/app.js";

const TARGET_CLASSES = new Set([
    "llama_cpp_instruct_adv",
    "Llama-cpp Instruct",
    "Llama-cpp Instruct (Advanced)"
]);

const MAX_SOCKETS = 9;

function isTargetNode(nodeOrData) {
    const className = nodeOrData?.comfyClass || nodeOrData?.type || nodeOrData?.name;
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

    const anchorIdx = inputs.findIndex(inp => inp && (inp.name === "video_0" || inp.name === "queue_handler"));
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

function syncAllDynamicInputs(node) {
    if (!node || node.__syncingDynamicInputs) {
        return;
    }

    node.__syncingDynamicInputs = true;
    try {
        syncDynamicCategoryInputs(node, "image_", "IMAGE");
        syncDynamicCategoryInputs(node, "video_", "IMAGE");

        if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
            const computed = node.computeSize();
            const current = node.size || [0, 0];
            node.setSize([
                Math.max(current[0], computed[0]),
                computed[1]
            ]);
        }
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

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
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
        nodeType.prototype.onConfigure = function() {
            this._configuring = true;
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            this._configuring = false;
            setTimeout(() => syncAllDynamicInputs(this), 50);
            return result;
        };
    },
    async nodeCreated(node) {
        if (isTargetNode(node)) {
            setTimeout(() => syncAllDynamicInputs(node), 50);
        }
    }
});

