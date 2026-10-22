# gpui-folio

folio 的 GPU 仿真翻页所需的 UI 框架分叉。

这个仓库不是通用库，只服务 folio 的一个效果：把**当前页烘成纹理**，再用
gmpagecurl-ios 那套「圆柱 / 盒子」纸模型在 GPU 上做网格形变、法线光照和
阴影 map。上游 `gpui-pre` / `gpui-pre-wgpu` 不暴露自定义网格、离屏页面纹理
或 shadow pass，所以只能在这里补。

## 目录

| 目录 | 内容 |
| --- | --- |
| `gpui-pre/` | gpui-pre 0.3.9 源码；新增 `Mesh` primitive、`paint_mesh`、`capture_page_region` |
| `gpui-pre-wgpu/` | gpui-pre-wgpu 0.3.9 源码；新增页面捕获 pass、卷曲网格管线、shadow pass |
| `folio-curl/` | 纯 CPU 的页面网格与 uniform 定义；可单测，不依赖 gpui |

## 为什么不直接在 folio 里画

folio 现在那套 `Shades` 只能填多边形和两段线性渐变。它能画出「折线 + 假卷筒」，
但画不出：

- 页面纹理跟着圆柱弯曲（正文是逐字形绘制的，字形变换被 `paint_glyph` 写死）；
- 卷曲纸投在新页上的 shadow map；
- 纸面法线带来的连续受光。

这些都必须落到 GPU 的网格与纹理上。

## 依赖接法

已推送到 <https://github.com/panzhifu/gpui-folio>（public）。

folio 根 `Cargo.toml` 用远端提交（推荐，构建可复现）：

```toml
[patch.crates-io]
gpui-pre = { git = "https://github.com/panzhifu/gpui-folio", rev = "3ebf9172532c82c90e55e11e99604af7b33cc0dd" }
gpui-pre-wgpu = { git = "https://github.com/panzhifu/gpui-folio", rev = "3ebf9172532c82c90e55e11e99604af7b33cc0dd" }
```

本地开发也可以继续用 path：

```toml
[patch.crates-io]
gpui-pre = { path = "../gpui-folio/gpui-pre" }
gpui-pre-wgpu = { path = "../gpui-folio/gpui-pre-wgpu" }
```

包名与版本必须保持 `gpui-pre 0.3.9` / `gpui-pre-wgpu 0.3.9`：`gpui-kit` 与
`gpui-mobile` 都精确依赖这两个版本，改版本会分出第二棵 gpui。Cargo 能从同一个
git workspace 里解析这两个包，实测 `cargo metadata` 已通过。

## 当前状态

- [x] 分叉骨架、workspace、依赖改 path
- [x] `Mesh` primitive / `paint_page_curl_mesh`
- [x] 页面捕获 pass（`capture_page_region`，按页面 bounds 重画到离屏纹理）
- [x] 卷曲网格管线与 WGSL（圆柱 / 盒子模型、弱透视、正反面、页面纹理）
- [x] shadow map 与新页投影（`Rgba8Unorm` 深度图 + 3×3 PCF）
- [x] 接入 folio `turn_animation`（`PageTurn::Curl` 走 GPU 网格，其余档保留 CPU `Shades`）
- [x] 离屏冒烟示例：`cargo run -p gpui-pre-wgpu --example page_curl_offscreen --features offscreen`

## 画一帧的顺序

1. 阅读页在 `PageTurn::Curl` 的静止帧调用 `Window::capture_page_region_with_color(版心)`，
   同时把旧页整页画一份副本、夹在 `set_capture_layer(1)` / `set_capture_layer(0)` 两道
   **闸**之间；动画在飞的那一帧把「揭起来那张纸的背面」画进层 2（双页是目标跨页的
   另一半，单页是目标页整张）。
2. 渲染器在主 pass 之前，用同一棵场景、同一个 scissor 重画到 `page_capture` 纹理，
   只取 `capture_order_range(1)` 那一段；层 2 写另一张纹理。主 pass 只画
   `order < CAPTURE_ORDER_BASE`，所以层里的副本不会上屏。
3. 翻页时阅读页只摆**进场**那一张，旧页不再走字形；`PageCurlMesh` 用捕获纹理
   作为正面纹理，按 `curl::cylinder_at` 的折线与半径卷起来。
4. 网格的片元自己采样 shadow map：卷面受光、平摊页上收卷筒投下的影（3×3 PCF）。
5. 网格背面是**正面这一页的墨透过来**（浓度 `BLEED`），不是下一页的正文：读者翻起来
   的是一张纸，反面看到的本来就是同一页的字。材质坐标**不再额外镜像** —— 纸绕过柱顶
   之后几何本身已经把材质投成镜像（`u = πR − d`），再反一次就两次抵消、读出来是正的字。
   捕获层 2 / 背面纹理暂时保留着（shader 已不采样）。

## 捕获分层的两个坑（都踩过）

### 1. 闸必须记进 `paint_operations`

捕获层的开关只在一次绘制的跨度里有效。gpui 有两条路会跳过元素的 `paint` 而只重放
上一帧记录的操作（视图没 notify 的 `reuse_paint`、deferred 里 `element` 为 `None`
那一支），层号若只是场景上的一个字段，重放时读到的就是主场景（`0`）：本该进层的
副本被重新编号排进主 pass，与正文叠成两页字。所以 `Scene::set_capture_layer` 把
`PaintOperation::SetCaptureLayer` 也写进操作流，重放时自己重新开合；deferred 那一头
在每条目开始时把层号收回 0，别让上一条目残留的层号把别的屏（书架、书目页）的字
一并排进捕获段。

### 2. `batches_in(range)` 跳过区间外 primitive 时，`*_start` 必须同步推进

`BatchIterator` 按 order 区间取批次。跳过区间外的 primitive 时**只推进了迭代器**、
没推进 `shadows_start` / `monochrome_sprites_start` 那九个计数器 —— 而那几个计数器
才是批次区间（也就是 instance 数据从第几条开始画）的唯一来源。

主 pass 的区间是 `0..CAPTURE_ORDER_BASE`，区间外的都在末尾，跳过发生在所有批次之后，
所以看不出问题；捕获 pass 的区间是 `1e9..1e9+1e6`，**一上来就要跳过全部主场景
primitive**，于是区间起点停在主场景那一段：卷起来那张纸上画的其实是主场景的字
（底下的书架、书目页跟着一起被卷进去），而层里自己那三百来条副本一条都没画。

真机上读回捕获纹理才看清：里面是「1本」「《斗战狂潮》」「书架/统计/笔记/我的」
这些别的屏的字，位置还是它们自己的位置。修法是九个跳过循环各补
`self.<kind>_start += 1`（`gpui-pre/src/scene.rs` 的 `BatchIterator::next`）。

## 翻页阴影：三个必须同时成立的条件

真机上「一点阴影都没有」查下来是三处独立的坑，缺一处就退回「全受光」：

1. **光源矩阵的 near/far 要取负。** `look_at` 的视图空间 z 是负的（点在相机前方），
   而 `ortho` 把 `[near, far]` 线性映到 `[0,1]`。传 `0.1` 与正 far 时整页 z ≈ −distance
   落在近平面之外，shadow pass 的三角形全被裁掉，图里只剩清屏色；主片那边
   `ndc.z > stored + bias` 永远为假。正确区间是 `[-0.1, -(2d + 4R)]`。
2. **shadow pass 要有深度测试，并写 `@builtin(frag_depth)`。** 颜色目标是 8 位、
   光栅顺序又不保证从近到远：没有深度测试时后画的三角形会盖掉先画的深度，图里存的是
   「最后一个」而不是「离光源最近的一个」。加一张 `Depth32Float` 附件 + `LessEqual`，
   片元把同一个深度写进 `frag_depth`，颜色里留下的才是最近表面。
3. **偏置要大于 8 位的一档。** `stored` 是量化过的，平面与自己做比较时会被半个量化步长
   判成「在影里」（自阴影噪点）。`SHADOW_BIAS = 0.0045` 约一档多一点；纸抬起来那一段的
   深度差约 0.03，仍吃得下。

另外，**光的方向要跟着折痕走**：`light_dir` 的水平分量沿 `curl.axis_normal`（已经卷走
的那一侧），影子才落在还平摊着的那半边。固定 `[-0.35, -0.25, 1]` 只对正向翻页成立，
往回翻时影子会落到已经卷走的一边、屏上仍然一片干净 —— app 侧现在用
`[n.x·0.25, n.y·0.25 − 0.12, 1]`。

水平分量取 **0.25** 而不是更大：影子的偏移量 = 水平分量 × 纸抬起来的高度，给到 0.55
时柱顶那一段的影会被推到折痕外一百多像素，读起来是一条独立的黑带；0.25 只让影子从纸边
探出六十来像素，是贴在折痕上的**接触影**（真机量到影里 158、纸边 255/240 两档都在）。

量过的读数：平摊页纸色 255，影里最深约 158（`0.62` 那一档），空白行上全 255（没有自阴影
噪点），正反两个方向都有影。

## 已知边界

- 捕获纹理是**整窗尺寸**：只在版心 scissor 内画，整张清成纸色。正面、背面各一张，
  手机竖屏约 10 MB/张。
- 页面捕获按 `(章, 页)` 烘一次就留着用；换页、重排、改尺寸才重烘，翻页动画期间不捕。
- 网格顶点/索引缓冲每帧重建一次。4k 顶点级别够用；要再省就把网格按页缓存。
- WebGL2 没接：`Rgba8Unorm` shadow map、自定义 mesh pipeline 只在原生 wgpu 后端验证。
- folio 目前用 `path = "../../../gpui-folio/..."` 接本地分叉；推远端后把
  `[patch.crates-io]` 的两条换成 `git = "..."` + `rev`，并让 `folio-curl` 也走同一个仓库。

## 关键实现点

- `GpuCurlUniforms` 的 GPU 布局必须是 **240 字节**：WGSL 的 `vec3` 与 `mat4x4` 都按
  16 字节对齐，`shadow_map_size` 之后有空洞；Rust `repr(C)` 用 `_pad0: [f32; 3]`、
  `_pad1a: f32` 显式补齐，并有 `offset_of!` 的编译期断言。少一格 Vulkan 会以
  「shader 要 240、布局只给 224」拒掉整条管线。
- 平摊段必须严格还原成原坐标：`axis_foot + n * d == p`。卷曲段才把横截面换成
  `u = R·sinθ`；透视用 `D / (D − z)`（z 越大离读者越近），深度用 `1 − z / D`，
  背面（z = 2R）才会压住平摊页。
- 正反面与折痕位置都在**片元里从材质坐标现算**（`past = n·local − d0`），不用顶点
  插值：网格一格在手机上约 11 个物理像素，插值出来的等值线是一条台阶，斜折痕会一格
  一格地跳。winding 也不参与判断。

上游许可：`gpui-pre` 与 `gpui-pre-wgpu` 均为 Apache-2.0，见各自
`LICENSE-APACHE`。
