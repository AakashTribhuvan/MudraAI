import type { AnimationContext, AnimationFrame } from "./types";

/**
 * Default neutral pose for avatar
 * Sets initial position with arms at sides and slight head tilt
 */
export const setDefaultPose = (context: AnimationContext): void => {
    context.animations.push(defaultPoseFrame());

    // Start animation loop if idle
    if (context.pending === false) {
        context.pending = true;
        context.animate();
    }
};

/** The neutral pose as one frame of [bone, "rotation", axis, value, sign] instructions. */
export const defaultPoseFrame = (): AnimationFrame => {
    const poseFrame: AnimationFrame = [];

    // Set neck angle for natural head position
    poseFrame.push(["mixamorig:Neck", "rotation", "x", Math.PI / 12, "+"]);

    // Position left arm at side
    poseFrame.push(["mixamorig:LeftArm", "rotation", "z", -Math.PI / 3, "-"]);
    poseFrame.push([
        "mixamorig:LeftForeArm",
        "rotation",
        "y",
        -Math.PI / 1.5,
        "-",
    ]);

    // Position right arm at side
    poseFrame.push(["mixamorig:RightArm", "rotation", "z", Math.PI / 3, "+"]);
    poseFrame.push([
        "mixamorig:RightForeArm",
        "rotation",
        "y",
        Math.PI / 1.5,
        "+",
    ]);

    return poseFrame;
};
