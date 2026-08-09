import { app } from "../../scripts/app.js";

const TARGET_CLASSES = new Set([
    "llama_cpp_instruct_adv",
    "Llama-cpp Instruct"
]);

const MAX_SOCKETS = 8;

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

function ensureTrailingDynamicInput(node, prefix, type) {
    const dynamicInputs = getDynamicInputs(node, prefix);
    if (dynamicInputs.length >= MAX_SOCKETS) {
        return;
    }

    if (!dynamicInputs.length || dynamicInputs[dynamicInputs.length - 1].link != null) {
        const newIndex = dynamicInputs.length;
        const inputName = `${prefix}${newIndex}`;

        const queueIdx = (node.inputs || []).findIndex(inp => inp && inp.name === "queue_handler");
        if (queueIdx !== -1) {
            node.addInput(inputName, type);
            const newInput = node.inputs.pop();
            node.inputs.splice(queueIdx, 0, newInput);
        } else {
            node.addInput(inputName, type);
        }
    }
}

function syncDynamicCategoryInputs(node, prefix, type) {
    if (!node._configuring) {
        for (let index = (node.inputs || []).length - 1; index >= 0; index -= 1) {
            const input = node.inputs[index];
            if (isDynamicInput(input, prefix) && input.link == null) {
                const count = (node.inputs || []).filter(inp => isDynamicInput(inp, prefix)).length;
                if (count > 1) {
                    node.removeInput(index);
                }
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
            const current = node.size;
            node.setSize([
                Math.max(current[0], computed[0]),
                Math.max(current[1], computed[1])
            ]);
        }
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
            const input = this.inputs?.[index];
            if (type === 2 && !isDynamicInput(input, "image_") && !isDynamicInput(input, "video_")) {
                return result;
            }

            setTimeout(() => syncAllDynamicInputs(this), 0);
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
