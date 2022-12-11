# CountingMOT
## Abstract
The recent trend in multiple object tracking (MOT)
is jointly solving detection and tracking, where object detection
and appearance feature (or motion) are learned simultaneously.
Despite competitive performance, in crowded scenes, joint detection
and tracking usually fail to find accurate object associations
due to missed or false detections. In this paper, we jointly
model counting, detection and re-identification in an end-toend
framework, named CountingMOT, tailored for crowded
scenes. By imposing mutual object-count constraints between
detection and counting, the CountingMOT tries to find a balance
between object detection and crowd density map estimation,
which can help it to find missed detections or rejecting false
detections. Our approach is an attempt to bridge the gap of object
detection, counting, and re-Identification. This is in contrast to
prior MOT methods that either ignore the crowd density and
thus are prone to failure in crowded scenes, or depend on
local correlations to build a graphical relationship for matching
targets. The proposed MOT tracker can perform online and realtime
tracking, and achieves the state-of-the-art results on public
benchmarks MOT16 (MOTA of 77.6), MOT17 (MOTA of 78.0%)
and MOT20 (MOTA of 70.2%).
