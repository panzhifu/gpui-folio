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

1. 阅读页在 `PageTurn::Curl` 的静止帧调用 `Window::capture_page_region(版心)`。
2. 渲染器在主 pass 之前，用同一棵场景、同一个 scissor 重画到 `page_capture` 纹理；
   版心之外的顶栏/底栏不进去。
3. 翻页时阅读页只摆**进场**那一张，旧页不再走字形；`PageCurlMesh` 用捕获纹理
   作为正面纹理，按 `curl::cylinder_at` 的折线与半径卷起来。
4. mesh pass 先画一张整页 shadow overlay（新页的投影），再画网格（depth 排序正反面）。
5. 网格背面用 `back_color`（纸色），不采样正面纹理 —— 这一档的背面是空白纸，
   不是下一页的镜像正文。

## 已知边界

- 捕获纹理是**整窗尺寸**，只在版心 scissor 内写；版心外是透明的。手机竖屏一帧
  约 5–10 MB，桌面大窗更大。
- 页面捕获当前在**每个分页静止帧**请求一次。只在有输入、真重绘的帧上发生；
  翻页动画期间不捕获。后续可按 `(章, 页)` 加一代脏标记，省掉重复 pass。
- 网格顶点/索引缓冲每帧重建一次。4k 顶点级别够用；要再省就把网格按页缓存。
- 背面是纸色，不是下一页。要做「背面 = 下一页」，需要第二次捕获/第二张纹理。
- WebGL2 没接：`Rgba8Unorm` shadow map、自定义 mesh pipeline 只在原生 wgpu 后端验证。
- folio 目前用 `path = "../../../gpui-folio/..."` 接本地分叉；推远端后把
  `[patch.crates-io]` 的两条换成 `git = "..."` + `rev`，并让 `folio-curl` 也走同一个仓库。

## 关键实现点

- `PageCurlUniforms` 的 GPU 布局必须是 **160 字节**：WGSL 的 `mat4x4` 按 16 字节
  对齐，`shadow_map_size` 之后有 12 字节空洞；Rust `repr(C)` 用 `_pad0: [f32; 3]`
  显式补齐，少一格就是 `min_binding_size` 对不上。
- 平摊段必须严格还原成原坐标：`axis_foot + n * d == p`。卷曲段才把横截面换成
  `u = R·sinθ`；透视用 `D / (D − z)`（z 越大离读者越近），深度用 `1 − z / D`，
  背面（z = 2R）才会压住平摊页。
- 正反面由 `normal.z` 判，不看 winding：NDC 的 y 翻转会让平摊页的三角绕序被
  当成背面，整页读成纸背。

上游许可：`gpui-pre` 与 `gpui-pre-wgpu` 均为 Apache-2.0，见各自
`LICENSE-APACHE`。
