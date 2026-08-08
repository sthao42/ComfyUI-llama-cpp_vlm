import { app } from "../../scripts/app.js";

const TARGET_CLASSES = new Set([
    "llama_cpp_instruct_adv",
    "Llama-cpp Instruct"
]);

const IMAGE_INPUT_PREFIX = "image_";
const IMAGE_INPUT_TYPE = "IMAGE";
const MAX_IMAGES = 8;

function isTargetNode(nodeOrData) {
    const className = nodeOrData?.comfyClass || nodeOrData?.type || nodeOrData?.name;
    return TARGET_CLASSES.has(className);
}

function isDynamicImageInput(input) {
    return Boolean(input?.name && String(input.name).startsWith(IMAGE_INPUT_PREFIX));
}

function getDynamicImageInputs(node) {
    return (node.inputs || []).filter(isDynamicImageInput);
}

function renumberDynamicImageInputs(node) {
    let nextIndex = 0;
    for (const input of node.inputs || []) {
        if (!isDynamicImageInput(input)) {
            continue;
        }
        input.name = `${IMAGE_INPUT_PREFIX}${nextIndex}`;
        input.label = input.name;
        nextIndex += 1;
    }
}

function ensureTrailingDynamicImageInput(node) {
    const imageInputs = getDynamicImageInputs(node);
    if (imageInputs.length >= MAX_IMAGES) {
        return;
    }
    
    if (!imageInputs.length || imageInputs[imageInputs.length - 1].link != null) {
        const newIndex = imageInputs.length;
        const inputName = `${IMAGE_INPUT_PREFIX}${newIndex}`;
        
        // Find queue_handler index to insert above queue_handler
        const queueIdx = (node.inputs || []).findIndex(inp => inp && inp.name === "queue_handler");
        if (queueIdx !== -1) {
            node.addInput(inputName, IMAGE_INPUT_TYPE);
            const newInput = node.inputs.pop();
            node.inputs.splice(queueIdx, 0, newInput);
        } else {
            node.addInput(inputName, IMAGE_INPUT_TYPE);
        }
    }
}

function syncDynamicImageInputs(node) {
    if (!node || node.__syncingImageInputs) {
        return;
    }

    node.__syncingImageInputs = true;
    try {
        // Remove unconnected dynamic inputs except during node configuration
        if (!node._configuring) {
            for (let index = (node.inputs || []).length - 1; index >= 0; index -= 1) {
                const input = node.inputs[index];
                if (isDynamicImageInput(input) && input.link == null) {
                    node.removeInput(index);
                }
            }
        }

        renumberDynamicImageInputs(node);
        ensureTrailingDynamicImageInput(node);
        renumberDynamicImageInputs(node);

        if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
            const computed = node.computeSize();
            const currentSize = Array.isArray(node.size) ? node.size : null;
            if (!currentSize) {
                node.setSize(computed);
            } else {
                const nextWidth = Math.max(currentSize[0] || 0, computed[0] || 0);
                const nextHeight = Math.max(currentSize[1] || 0, computed[1] || 0);
                if (nextWidth !== currentSize[0] || nextHeight !== currentSize[1]) {
                    node.setSize([nextWidth, nextHeight]);
                }
            }
        }
        app.graph?.setDirtyCanvas(true, true);
    } finally {
        node.__syncingImageInputs = false;
    }
}

app.registerExtension({
    name: "ComfyUI-llama-cpp_vlm.DynamicImageInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNode(nodeData)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setTimeout(() => {
                syncDynamicImageInputs(this);
            }, 50);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, index, connected, linkInfo) {
            const result = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            const input = this.inputs?.[index];
            if (type === 2 && !isDynamicImageInput(input)) {
                return result;
            }

            setTimeout(() => syncDynamicImageInputs(this), 0);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            this._configuring = true;
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            this._configuring = false;
            setTimeout(() => syncDynamicImageInputs(this), 50);
            return result;
        };
    },
    async nodeCreated(node) {
        if (isTargetNode(node)) {
            setTimeout(() => syncDynamicImageInputs(node), 50);
        }
    }
});
