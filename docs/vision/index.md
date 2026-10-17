# Vision

Dense-Armor sees a video the way it reads a robot's joints: one frame
at a time, in fixed memory, with no pretrained model. The pipeline is
`Frame` → `FrameFeatures` → reduce → anomaly or tracking.

The pages, in order:

1. [Frames](frames.md): every source (array, folder, camera, file, URL,
   cube) gives the same `Frame` with the image and the timestamp.
2. [Features](features.md): the oriented gradients, the intensity
   moments, the colour, from `FrameFeatures`.
3. [Flow](flow.md): Lucas–Kanade in a cell, the pyramid, dense flow,
   the motion mask.
4. [Reduce](reduce.md): the random projection, the incremental PCA, the
   whole chain against a change in the scene.
5. [Motion](motion.md): the motion energy, the global shift, the scene
   changes, the sliding window.
6. [Tracking](tracking.md): the moving object tracker, the video to
   signal adapters, the joints to video fusion.
7. [Kinematics](kinematics.md): landmarks, the cleaner, the joint
   angles, the registration and the difference image.
8. [Quality](quality.md): the frame quality, the event frame, the
   false colour, the overlay, the benchmark.
9. [Patches](patches.md): patch memory and the RAD benchmark.

The detailed numbers and the story of each module are in the
[Details](#details) section of its page.
