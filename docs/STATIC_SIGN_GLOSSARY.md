# Static-sign glossary

The live Words mode uses the 32-label still-pose medoid model adapted from
`MudraPhoneWebcam/training_data/models/intro_static_pose_medoids.npz`. It has 59 label/hand-count
profiles and finite rejection thresholds for ten labels: `DO`, `FOOD`, `FROM`, `HELP`,
`I_ME_MINE_MY`, `NAME`, `NICE`, `THANK`, `WHAT`, and `YOU`.

The browser streams landmarks to the backend. Static mode uses the phone prototype's activity
gates, predicts each eligible frame, and requires the same accepted label for 450 ms before it
returns the existing one-result-per-sign event used by Words. A rest interval rearms recognition.
Spell mode continues to use the separate alphabet model.

The source model's exploratory still-image test had 73 samples and 13 correct predictions (17.8%).
This is not a live-camera or new-signer accuracy estimate. The source labels are folder glosses and
have not been independently verified by an ISL-fluent signer. The phone model's held-out metric is
lower than the earlier MudraTraining static artifact; it was selected here to reproduce the
previously working phone recognition path, not because its benchmark is stronger.
