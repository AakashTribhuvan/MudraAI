# Avatar models

Put rigged avatar files here (`.fbx` or `.glb`). Each one appears in the avatar dropdown of the 3D
avatar viewer after the page reloads; the file name becomes its display name (`my_avatar.fbx` →
"My avatar").

Requirements:

- **Mixamo skeleton** (bones named `mixamorig:Hips`, `mixamorig:RightArm`, …). The sign animations move
  these bones; a model without them loads but cannot sign (the viewer shows a warning).
- **Textures embedded** in the FBX (Mixamo: Format "FBX Binary (.fbx)", Skin "With Skin"). Separate
  texture files next to the FBX are not loaded.
- Any size or unit: the viewer scales every model to the same height and frames the camera from it.

These files are not committed (`.gitignore`): third-party characters (e.g. Mixamo) may not be redistributed
as files, and they are large. Each person drops their own avatar here.
