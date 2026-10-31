# Media

Every capture comes from the real Explorer, using recorded runs and tracked evidence. No game imagery is included.

| Asset | Size | What it shows |
| --- | --- | --- |
| `hero.gif` | 900x513, 10.5 s | Explorer views from the network to the generated text |
| `story.gif` | 800x304, 13.5 s | stories260K tokens, prerecorded and sped up about 1,190× |
| `accelerator.png`, `network.png` | 1400 wide | The 3D network map, with and without the accelerator filter |
| `model.png`, `life.png`, `mnist.png` | 1400 wide | Explorer application views |
| `social-preview.png` | 1280x640 | GitHub repository preview |

The social videos are 42.5 seconds, 1080x1920 and 1920x1080, silent H.264. They render to `.cache/media/` and stay out of git.

## Accuracy

- Story text comes from `experiments/tiny-llm-ndot-fused/evidence/story-validation-20261031.json` through `media/capture/story_data.py`, which checks that the tokens join to the recorded story.
- Speed-up labels are computed in `media/video/src/parts.tsx`: 14,225.90 compute seconds divided by the time on screen. That is about 1,780× over 8 seconds in the videos and about 1,190× over 12 seconds in `story.gif`.
- The network orbit moves the Explorer's camera, not the recorded data.
- The nested boxes in the videos are a diagram and are labelled as one.

## Regenerate

Needs Chrome, Node and ffmpeg.

```sh
media/capture/capture.sh
python3 media/capture/story_data.py
cd media/video && npm ci && sh render.sh
```
