# External baseline inventory

Clone the upstream repository and check out the exact commit before applying
the corresponding local patch series. Model weights and environments remain
subject to each upstream license.

| Method/repository | Upstream | Frozen commit | Local state |
|---|---|---|---|
| PromptHMR | `https://github.com/yufu-wang/PromptHMR.git` | `bd3a7c482b587591c0b1b8daacb60dde639da55c` | one local commit; patch archived |
| OnlineHMR | `https://github.com/Tsukasane/Video-OnlineHMR.git` | `157ae880fb75921c65fa039166edae09a90ed029` | three local commits; patch archived |
| TRACE/ROMP | `https://github.com/Arthur151/ROMP.git` | `75b74f75f6d0a23a0e956151876ebf51bc6c1bb7` | two local commits; patch archived |
| TRAM | `https://github.com/yufu-wang/tram.git` | upstream `4861c112f3c148201326680a50c9199650da6088` | two small local-cache edits; patch archived |
| JOSH | `https://github.com/genforce/JOSH.git` | `352efc5793e454e7d9995224f06114b68dc295c3` | one local commit; patch archived |
| GVHMR | `https://github.com/zju3dv/GVHMR.git` | `6ec3ca39336c50492c0fae65fba2fb831fc7d866` | clean upstream |
| WHAM | `https://github.com/yohanshin/WHAM.git` | `2b54f7797391c94876848b905ed875b154c4a295` | clean upstream |
| SLAHMR | `https://github.com/vye16/slahmr.git` | `58518fec991877bc4911e260776589185b828fe9` | clean upstream |
| HSfM | `https://github.com/hongsukchoi/HSfM_RELEASE.git` | `75f835e91c1b1dc97713ed7ab78b2d311f3b173a` | clean upstream |
| HumanMM | `https://github.com/zhangyuhong01/HumanMM-code.git` | `dd5049b1a776f2c395005235dd048e2119e110ce` | clean upstream |
| MultiShot | `https://github.com/geopavlakos/multishot.git` | `745287ec850f8e23912f0c863b0770fd4b634d26` | clean upstream |

Patch files are stored under `docs/recovery/baseline_patches/`. They contain
code and configuration changes only; no third-party weights or caches are
included. JOSH's dirty `third_party/deco` state was only deletion of a tracked
`__pycache__` file and is intentionally not preserved.

TRAM's patch makes official ZoeDepth and ViTDet weights load from a local cache
when present so network download time is not included in runtime. If the cache
is absent, it retains the official URL fallback.

