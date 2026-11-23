# 前端验收记录

## 当前交付 · 2026-11-23

- 首页及三个工作区沿用当前审阅版本。公共按钮底色在蓝色与冰白色之间以 800ms ease 过渡，减弱动态效果下也保留颜色变化。
- 桌面 1440px 与手机 390px 浏览器验收通过。三个入口均能从按钮变形到侧边图标，关闭后回到首页；页面与工作区无横向溢出。
- 配置选择 B 算法并保存，运行页读取正确，返回修改参数仍保留已保存配置。
- 结果页 CSV 和 JSON 的实际下载与 public 中的发布文件一致，任务分页往返正常。
- 普通动画模式下，观测钟外圈与扫光的变换随时间变化，小时标签与任务弧段保持数据位置。
- 使用减弱动态效果验证了按钮悬停，时长仍为 0.8 秒，0.25 秒时采到中间色。
- 清理了工作区已被公共按钮样式覆盖的旧规则和废弃类名，发布摘要清理本机缓存路径，复用地球品牌图标补充 favicon。
- 生产构建和静态托管检查在提交前运行。浏览器检查脚本存放在临时目录；截图与设计参考保存在本地 qa/ 和 references/，由 .gitignore 排除。

## 历史审阅记录

以下记录保留各轮审阅的实现与验证过程。

### Homepage design QA — 2026-11-14

final result: passed

## Source and comparison scope

- Structural source: `references/homepage-sketch.png`, converted for viewing from the user's `IMG_9833.HEIC`. The file contains 4032 × 3024 pixels and displays as a portrait sketch after its orientation metadata is applied; it is a hand-drawn layout/interaction specification, not a typography or pixel-perfect mock.
- Visual source: `public/assets/observatory-original.png`, the exact user-selected 1672 × 941 image. Do not substitute the subsequent giant-observatory variants.
- Final desktop: `qa/desktop-final.jpg`, browser CSS viewport 1280 × 720, screenshot 1280 × 720.
- Final narrow screen: `qa/mobile-final.jpg`, CSS viewport and screenshot 390 × 844. Additional minimum-width capture: `qa/mobile-360.jpg`, CSS viewport 360 × 780, full-page screenshot for the 840 px minimum content height.
- States: default home, keyboard focus on third entry while pointer remains over first, configuration saved and read by run panel, results, mobile configuration, meteor animation.
- The selected background and desktop capture have the same approximately 16:9 aspect ratio; background landmarks were compared proportionally. Source sketch was compared structurally rather than stretched to a false pixel match. Browser screenshots were captured at CSS-pixel dimensions, without density upscaling.
- Full-view evidence: the source sketch, exact background, desktop and mobile captures were opened together in one visual-review input. A second review opened the source sketch and `qa/desktop-focus-third-fixed.jpg` together to verify the P2 fix. The final desktop/mobile captures were also inspected directly in the browser.
- Focused evidence: the focus-state capture makes the three entry backgrounds, focus ring and telescope direction clearly readable; mobile configuration was captured separately in `qa/mobile-config.jpg`. No additional crop was needed at these screenshot sizes.

## Findings and comparison history

1. **P2 — mobile controls obscured the telescope.** `qa/mobile-before.jpg` showed the telescope behind the second/third entries. The mobile composition now uses a lower, smaller view of the same source photograph. The two photograph views use a feathered mask; no drawn landscape substitutes for the image. An initial visible join was corrected. `qa/mobile-final.jpg` shows the entire visible telescope below the card stack, without a visible rectangular join. Fixed.
2. **P2 — keyboard and pointer produced two highlighted entries.** `qa/desktop-focus-third.jpg` showed the third entry's focus ring while the first remained brighter. Focused and hovered entries are now tracked separately, with keyboard focus taking precedence. In a real browser, focusing the third entry and moving the pointer over the first leaves both focus and active state on the third; the telescope angle is 52.50 degrees. `qa/desktop-focus-third-fixed.jpg` was visually reviewed against the source and confirmed to show only the third entry emphasized. Fixed.
3. **P3 — mobile supporting copy and breathing room.** Supporting copy was increased from 10 to 11 px and brightened; the observatory was moved a further 20 px down. Final 390 px capture is readable, with space between the last entry and observatory. Fixed.

No remaining actionable P0/P1/P2 findings in the checked views.

## Required fidelity surfaces

- **Fonts and typography:** the sketch does not prescribe a font. Noto Sans SC with Chinese system fallbacks provides a restrained technical appearance. Browser font readiness was confirmed. Heading, entry titles, explanation and auxiliary text have distinct weights/sizes; the desktop heading uses two intentional lines. At 360 px, entry text width does not exceed its content box.
- **Spacing and layout rhythm:** desktop preserves top tools, lower-left observatory and right-side vertical entries. Three controls use consistent padding, radii and gaps. Mobile retains the same order with a vertical layout. Document width equals viewport width at 1280, 390 and 360 px; no horizontal page overflow was found.
- **Colors and tokens:** midnight-blue source colors, off-white title, quieter blue-gray explanatory copy, restrained translucent controls. Inactive entries fade to 72% while the active entry brightens and grows. Focus is visible and corresponds to the emphasized entry. This is a targeted readability check, not a comprehensive WCAG contrast certification.
- **Image quality and fidelity:** complete background is the exact selected image. Only the original dome opening is locally covered by the clean plate, with the transparent telescope placed above it. Generated source detail, alpha and loading were checked. Dome/landscape remain stationary; telescope rotates around its pivot. Meteor is a real raster asset. No CSS illustration replaces the sky, observatory or telescope.
- **Copy/content:** final user-approved division is `运行观测规划 / 配置与调整 / 查看与导出`. The second entry saves parameters; the first reads them. Results are clearly labeled an existing example. Neither a configuration preview nor opening the old workbench is presented as a new algorithm run.

## Browser interaction checks

- Set planning duration to 3 days and selected station 03 in addition to 01/02; saved configuration; run panel showed 3 days and stations 01/02/03. Closing/reopening the panel retains current-page saved state.
- Downloaded configuration JSON through the browser and parsed the actual file; `days=3`, `stations=[1,2,3]`, `status=configuration_only`.
- Opened existing example results and verified six visible task rows, 239 tasks and the example's recorded independent validation state. Downloaded the actual CSV; file content matches the served asset and parses to 239 rows.
- Downloaded station TXT template and used the file chooser to read it back in the local upload panel; filename and text preview appeared. No server upload occurs.
- Opened UTC conversion: 2026-08-01 12:00 UTC displayed 20:00 Beijing time. Escape closed the dialog.
- Verified motion pause/enable, initial reduced-motion default (`prefers-reduced-motion: reduce` was true in the test browser), manual enable, idle angle change, focus-driven angle, and the meteor element with loaded raster and active `meteor-pass` animation. `qa/meteor.jpg` records the animation state.
- Checked desktop and mobile configuration panels. Native dialog focus containment and Escape behavior are retained.
- Final browser console error/warning log: empty.

## Build and implementation checklist

- [x] Production build passed.
- [x] Four bundled static-hosting checks passed.
- [x] `git diff --check` passed.
- [x] Local preview remains running at `http://127.0.0.1:4173/` and the in-app browser tab is retained.
- [x] Temporary viewport overrides reset after QA.
- [x] Existing backend, algorithms and Streamlit implementation were not edited in this frontend task.
- [x] No deployment, commit, push or merge.

## Practical limits

This is the first-level frontend and its supporting interactions. Saved parameters are held in the current page; a refresh restores defaults. The run panel links to the existing localhost workbench but does not transfer configuration or trigger a calculation. Upload is local text preview only. Existing sample validation is historical evidence, not a new backend test. New users' five-second comprehension, complete screen-reader behavior, every browser/device, and a fully integrated backend workflow were not independently tested.

## Scoped annotation update — remove unsolicited homepage copy

User requested removal of the selected hero eyebrow, heading and explanatory paragraph, the lower-left slogan, and the caption below the entries; the same decorative English footer was removed as well. No replacement slogan or introductory copy was added. The document title was reduced to the product name.

- Source evidence: the user's annotated screenshot plus `qa/copy-removal-before.jpg`.
- Final evidence: `qa/copy-removal-after.jpg`.
- Both browser captures are 1176 × 826 at the same viewport and were opened together for comparison.
- Required surfaces: no new typography or colors; original assets and navigation retained; visible copy matches the deletion request. Entry geometry is preserved (top 401.3125 px before, 401.328125 px after; identical width and height), within subpixel rounding.
- Browser DOM check confirms all requested phrases and the decorative footer are absent. The three entry controls and data/tool menus remain present.
- Production build and `git diff --check` passed. No backend changes or new tests were needed for this text-only change. This scoped pass does not rerun the previous full interaction checks.
- No actionable P0/P1/P2 findings in the annotated view. final result: passed

## 2026-11-20 首页动效与入口复核

- 在内置浏览器刷新首页，桌面和手机均会自动播放。分别对比相隔约一秒的截图，星带和前景草尖发生位移，选取的建筑墙面区域像素完全相同。桌面另在模拟系统减少动态偏好下重复了冷启动检查。
- 在 390 × 844 和 360 × 640 两种视口核对了入口、穹顶与底部草地的首屏位置。浏览器已恢复默认视口。
- 三个入口都能打开对应面板。键盘聚焦与鼠标悬停均可操作。三个按钮只显示名称，导航和工具菜单中的动态开关已移除。
- 逐一悬停时，镜筒相对原始姿态的角度分别为 −9°、0°、+10°。固定支架与轴盖均无变换，底部托架与舱口相连。
- 实际浏览器连续画面保存在 `qa/homepage-interaction.webp`。生产构建及四项静态托管测试通过。

## 2026-11-20 黑白圆角入口与固定观测站

- 在 828 px 高的桌面视口，三个入口整体上移约 75 px。390 × 844 视口的首个入口上移约 45 px，360 × 640 视口也保留了导航与入口之间的流星空间。
- 浏览器报告首个按钮的六个汉字全部使用本地 Observation Rounded 字体。三个入口仍可打开对应面板，悬停描边沿 16 px 圆角衔接。
- 桌面实录包含两组流星和三次按钮悬停。对比首帧与随后五帧，镜筒和建筑墙面的取样区完全相同，银河和草地的取样区持续变化。另检查了手机的流星、草地和站体构图。
- 实录为 `qa/monochrome-night.webp`。生产构建与四项静态托管测试通过。

## 2026-11-20 纯黑入口与描边收尾

- 三个入口均为纯黑底白字。普通桌面、大屏桌面与手机的字号分别为 20、21、19 px。
- 使用完整静止的圆角描边和四个揭示区域。桌面连续截图对比最终轮廓，边角没有突出轮廓的高亮像素。分别在进入后 160、430、735 ms 移出再移入，均能清除高亮并正常闭合。手机键盘焦点亦能显示完整描边。
- 三个入口的实际悬停画面见 `qa/black-buttons-outline.webp`。生产构建及四项静态托管测试通过。

## 2026-11-21 仰角刻度入口

- 视觉目标为用户选定的第三张概念图，保存在 `references/elevation-menu-option-3.png`。原图与静止状态浏览器实拍 `qa/elevation-menu-reference-size.jpg` 均为 1586 × 992。浏览器采用同尺寸 CSS 视口，每个 CSS 像素对应一个截图像素。
- 首轮在 1440 × 900 核对了文字、三个节点及地图。随后将菜单水平锚点按视口宽度定位，避免到原图尺寸时被外层列宽上限向左拉偏。最终原尺寸下，菜单绘制起点为 (989.7, 300.0)，参考起点为 (990, 300)。
- `qa/elevation-menu-comparison.jpg` 将原图与最终实拍等比缩小并排对照，参考在左。`qa/elevation-menu-detail.jpg` 在相同位置截取原像素细节。弧形、短刻线、金银节点与文字间距均已核对。字形由浏览器实时渲染，与参考位图有轻微笔画和基线差异。动态银河按实拍时刻呈现。
- 在 1440 × 900 逐个点击弧上的三个节点，均打开相应工作区。逐次用 Esc 关闭均把焦点还给原入口。用 Shift+Tab 选中配置并按空格打开时，文字清晰，焦点轮廓可见。浏览器控制台检查未发现警告或错误。
- 在 390 × 844 核对了三行文字、节点及完整天线，三个按钮实际点击区域均高 48 像素。在 360 × 640 核对结果入口、关闭后的焦点与页面横向宽度。弧底与天线顶端留有约 37 像素。手机实拍见 `qa/elevation-menu-mobile.jpg`。
- 最终生产构建和四项静态托管测试通过，桌面与窄屏检查没有遗留 P0、P1 或 P2 问题。

final result: passed

## 2026-11-21 弧形菜单的悬停反馈

桌面实测覆盖金色播放、齿轮和文件下载图标。来回跨过三个按钮时，每次只有当前操作项呈现金色；移入空白区域后恢复默认播放节点。鼠标停在配置项时，用 Tab 切到结果项，金色节点切到结果；轻移鼠标后，金色回到配置项。点击配置再按 Esc，焦点回到该入口。

在 1440 × 900 录下了实际浏览器的连续画面，动图保存在 `qa/elevation-hover.webp`。系统偏好减少动态时，金色、图标和大小即时切换；录制正常缓动时临时使用标准动态偏好，录制后恢复。在 360 × 640 核对三项文字放大后的右缘都处于各自固定按钮范围内，并通过点击打开和关闭结果页。本次窄屏检查使用鼠标输入。生产构建与四项静态托管测试通过。

final result: passed

## 2026-11-21 银环待机与圆角按钮动效

首次打开首页以及鼠标离开菜单后，三个金色图标的不透明度均为 0。依次移过运行、配置和导出时，每次只有对应入口进入高亮。实际浏览器录屏见 `qa/elevation-pill-hover.webp`。

在 1440 × 900 下，运行入口悬停约 110、250、550 ms 时的字距分别为 1.98、2.69、3 px，按钮逐步放大到 1.07 倍，点击区域全程固定。减少动态模式也保留渐变，最终放大到 1.025 倍。两种模式的按下位移分别为 10 px 和 2 px。360 × 640 窄屏下，三项放大后均处于各自的点击区域内；实际打开结果页并按 Esc 后，焦点返回结果入口。控制台检查无警告或错误。

## 2026-11-21 圆体菜单与地图信号

在 1440 × 900 实测，刻度弧的顶端下移约 48 px，高度从 426 px 增至 491 px；首个按钮高 67 px，字号约 20 px。浏览器实际使用本地 Observation Rounded 字体绘制按钮，三个入口均可打开工作区，按 Esc 后焦点逐一返回。360 × 640、390 × 844、801 × 740 和 1024 × 768 下三个按钮均处于屏幕内。最窄尺寸下，天线与最下方入口之间仍有可见留白。

常规模式有四颗信号点。桌面录制约 6 秒，抽样时四条已分配线路均互异，录制期间七条连线均有光点经过。在 360 × 640 切换至减少动态模式时，三颗可见光点均持续缓慢移动。实拍见 `qa/observatory-rounded-home.jpg` 与 `qa/observatory-map-signals.webp`。生产构建和四项静态托管测试均通过。

## 2026-11-21 天线寻星与接收信号

在 1440 × 900 的首页预览中，碟面从原姿态缓慢抬至 −3.5°，停驻后转至 +2.5°。对照两端姿态，接收器和支撑杆跟随碟面，固定轴承在画面中的位置一致。三张透明图层解码完成前显示完整原照，随后一次性合成。浏览器局部实录为 `qa/observatory-tracking-reception.webp`。

桌面录制中，第三次地图光点到站后接收器首次亮起，随后熄灭。在 390 × 844 和 360 × 640 检查了手机裁切，最后一个按钮与碟面之间有留白。减少动态模式下，转动幅度降低，接收灯随到站事件淡入淡出。生产构建和四项静态托管测试通过。

## 2026-11-21 加快寻星与扩大摆幅

桌面常规模式已实录完整 14 秒周期。约 1.8 秒完成首次 8° 调向，随后在 −8° 和 +6° 两端停留约 2.8 秒及 3.5 秒。1440 × 900 时，接收器中心在两端之间移动约 57 个 CSS 像素，固定轴承位置全程相同。动图 `qa/dish-tracking-14s.webp` 以实测时间播放。360 × 640 的减少动态模式采用 18 秒周期，实测到达 −4.75° 和 +3.75°，碟面与按钮之间保持留白。生产构建与四项静态托管测试通过。

## 2026-11-22 页头数据入口与薄荷绿高亮

在 1160 × 740 核对了品牌旁的短分隔线和常显标签。悬停时下载箭头下移 3px、上传箭头上移 3px，两项文字和点击区域的位置均固定。用 Enter 展开模板列表、在列表内按 Esc 后焦点返回下载按钮；上传按钮打开对应对话框。360 × 640 下两个入口并排在第二行，点击区域高 44px，展开的模板列表位于视口内。逐个悬停运行、配置、导出后，只有当前项亮起薄荷绿圆标（#69e5ab）与深绿图形（#103a29）。实拍见 `qa/header-mint-menu.jpg`。生产构建与四项静态托管测试通过。

## 2026-11-22 运行观测规划定稿

以用户确认的柔和过渡效果图为基准，与默认配置下的 [桌面实拍](qa/run-window-desktop.jpg) 在 1275 × 1233 的同一画布并排核对，浏览器设备像素比为 1。第一次对照中，站点行与入口偏高、时间刻度偏小。调整字号和垂直间距后再次截图，入口顶部为 1040px，参考图约为 1042px，标题、两列参数和各段分隔线对齐。同一张夜景柔和融入墨蓝面板，使用原有动态碟面，朝向随时间略有变化。站点旁直接使用定稿里的碟形图标。标题、日期、UTC、站点和同步提示均为可读取的文本。

1440 × 900 下整条默认流程和主操作可在一屏内读完。[390 × 844 手机实拍](qa/run-window-mobile.jpg) 中运行页占满宽度，两列参数、五个刻度、站点和入口均完整可见。桌面分隔控件按 Home 可缩至 480px，按左方向键增至 504px。Esc 关闭后焦点回到首页「运行观测规划」。

在「修改参数」设置 2026-12-30、3 天、站点 01/02/03，保存并进入运行页，页面显示已保存配置，时间尺终点为 2027-01-02 00:00 UTC。刷新后恢复默认的 2026-08-01 和 24 小时时间尺。已核对前端入口的目标地址；验收范围为前端，未执行外部工作台的规划任务。生产构建及四项静态托管测试通过。

final result: passed

## 2026-11-22 品牌与二级工作区动效

在桌面核对了首页和运行页的品牌文字、地球和标签的 bounding box。1440px 宽时两态的品牌文字都是 24px，地球都是 32px，x/y 一致。在本地浏览器逐个打开并关闭了三个工作区。用常规模式采样，运行页入场的横向偏移依次为 59.5、17.9、1.8、0px，退场透明度依次为 0.99、0.62、0.14，关闭后焦点返回入口。快速 Esc 也能退出。390px 下的减少动态模式为 200ms 原地淡变。[工作区进出实录](qa/workspace-motion.webp)。生产构建及 4 项静态托管测试通过。

final result: passed

## 2026-11-22 运行入口变形与横向进出

在 1200 × 820 的常规动态模式逐帧检查，启动胶囊从实际点击位置出发，移动中收窄后落在运行页标题左侧，面板整体透明度始终为 1。调整了图标的位移曲线，使最后一段与面板一起落位。关闭时返回原入口并恢复焦点，入场途中按 Esc 也能连续折返。减弱动态模式下用 390 × 844 视窗验证右向左展开、直线压缩及落点，标题与关闭按钮之间有留白。通过配置页保存后返回运行页不会重播首页发射。网页内的轨道只有椭圆、播放三角和巡行光点，已去掉左上方的短弧与刻度。[实际进出录屏](qa/run-pill-orbit.webp)。生产构建与四项静态托管测试通过。

final result: passed

## 2026-11-22 运行图标落点与文案修订

在与标注图相同的 1394 × 954 视口打开运行页，轨道图标中心为 (183, 235)，位于品牌与天线之间的星空。标题与下方参数区域左对齐。加宽右侧面板后，图标随可见星空移动，正常接收点击；关闭后焦点回到首页运行按钮。390 × 844 下，图标位于独立的星空窄带里，标题在其下方。已检查副标题和同步提示均已删除。生产构建和四项静态托管测试通过。[桌面实拍](qa/run-sky-dock.jpg)

final result: passed

## 2026-11-22 启动图标大小与轨道动画

1394 × 954 视口实测轨道外形约 173 × 145 px，中心位于 (183, 235)。390 × 844 视口实测约 75 × 63 px，图标与标题之间留有间距。普通模式录下完整一圈，逐帧确认光点经过轨道的不同位置。减少动态效果模式下也确认光点持续移动。生产构建及四项静态站点测试通过。[轨道动图](qa/run-sky-orbit-large.webp) · [桌面原图](qa/run-sky-orbit-large.jpg)

## 2026-11-22 运行页时间窗与站址核对

按[设计稿](references/run-instrument.png)组合起止日期、天数、总小时数和弧形时间尺。站点表显示当前配置选中的真实站址。桌面 1097 × 828 与手机 390 × 844 均实测默认两站。另在 320px 视口验证 10 站与 7 天配置，从 2026-12-30 到 2027-01-06，时间尺跨年、共 168 小时，四列站址没有水平溢出。生产构建及四项静态站点测试通过。[桌面实拍](qa/run-instrument-desktop.jpg) · [手机实拍](qa/run-instrument-mobile.jpg)

## 2026-11-22 运行页改为浅灰观测记录单

按[新设计稿](references/run-register.png)移除蓝绿色卡片与网格填充。桌面 1097 × 828 实测启动图标随按钮入场，落位后光点继续移动。默认配置的时间尺与两行站址表、主操作均可在首屏读到。390 × 844 下时长另起一行，320px 下日期与表格数值无横向溢出。生产构建及 4 项静态站点测试通过。[桌面实拍](qa/run-register-desktop.jpg) · [手机实拍](qa/run-register-mobile.jpg)

## 2026-11-23 配置页与三种规划算法

在 1380 × 920 桌面验证了默认 H、改选 E 和 B、保存后运行页回显，以及关闭配置页丢弃未保存的更改。以 2026-12-30 起的 3 天和站点 01、03、10 验证跨年时间范围与 JSON 下载。清空日期或取消全部站点时，保存按钮禁用。390 × 844 手机视口的算法选项纵向排列，320px 视口无横向溢出。生产构建与 4 项静态托管测试通过。[桌面实拍](qa/config-algorithms-desktop.jpg) · [手机实拍](qa/config-algorithms-mobile.jpg)

## 2026-11-23 查看与导出

用示例 20261113-083604-HR 的完整 CSV 校对站点 01 的 154 项、站点 02 的 85 项及全部 239 条任务。用原始轨道目录核对了每项的目标名称与编号。浏览器实测从首页第三个入口和「下载模板 → 查看示例输出」进入，Esc 返回相应入口。已逐页翻到第 40 页，下载的 CSV 和 JSON 分别与原文件逐字节相同；模拟读取错误后重新进入可恢复。桌面 1440 × 960、手机 390 × 844 和 320px 均无横向溢出。普通模式录制了 13 秒整圈扫光，减弱动态效果模式显示静态时序。生产构建及 4 项静态托管测试通过。[实际动效](qa/results-clock-motion.webp) · [桌面](qa/results-clock-desktop.jpg) · [手机](qa/results-clock-mobile.jpg)

## 2026-11-23 结果入口与观测钟外圈动效

从首页第三个胶囊实测打开与关闭。桌面图标落在品牌与天线之间，390 × 844 手机版的导出图标位于标题上方。入场中按 Esc 可以折返，关闭后焦点回到原按钮。「下载模板 → 查看示例输出」只展开面板。复核了运行与配置两个入口的原有变形。观测钟外圈 3.25 秒转过 90°，刻度、数据色段和停机斜纹固定，减弱动态效果会停下表圈与扫光。已录制完整按钮动作与整圈运行。[打开和关闭实录](qa/results-entry-and-rim.webp) · [完整表圈](qa/results-clock-motion.webp)

## 二级页面按钮动效

在桌面 Chrome 验证了修改参数、保存配置、前往运行和下载的鼠标与键盘操作。仅提交配置后绘出确认勾，再次打开已保存配置时直接显示结果。键盘提交后焦点移到「前往运行」。结果页两份下载与原文件逐字节一致。连续向前、向后及到达首尾页时，表头和焦点保持稳定。320px 触屏视口验证保存、下载、翻页与关闭。开启减弱动态效果后，图标和表格静态切换。生产构建及四项静态托管测试通过。[配置按钮实录](qa/workspace-button-feedback.webp)

## 二级页虚线按钮与横向导航

桌面 Chrome 和 320px 触屏视口检查了七个文字操作的常态、悬停、按下与布局，核对 400ms 反色、2px 虚线和 44px 最小操作高度。在工作区中分别从配置页和运行页进入并返回，观测到标题与内容同向滑动，切换约 185ms 后反向可从中间位置折返。Esc 在切换途中和完成后都能返回最初入口。手机减弱动态效果下，内部页面立即切换；上传与帮助的入场中途也可关闭。[按钮与页面往返实录](qa/workspace-buttons-and-navigation.webp)

## 按钮反色过渡

在 Chrome 中验证了七个文字操作和翻页按钮。深蓝表面、浅色标签及虚线共享同一条柔边遮罩，两层图标和文字位置重合。鼠标快速扫过下载入口时，各按钮从当前进度退回。键盘聚焦可触发相同过渡，系统减弱动态效果时直接切换。320px 触屏检查通过，两个结果文件与原文件一致。[下载按钮过渡实录](qa/workspace-button-color.webp)

## 二级页按钮悬停变色

Chrome 实测「下载计划 CSV」和「摘要 JSON」的初始底色分别为 #356983 和 #415f71。悬停后立即开始过渡，2 秒后到达原来的冰白色 #f2f7fa。主按钮在约 0.5、1.0、1.5 秒时依次呈 #7b9eaf、#c9d9e0、#e9f1f5。移出后回到原蓝色。[查看两秒悬停实录](qa/workspace-button-hover-2s.webp)

排查用户预览的瞬间变色后，发现 macOS reduceMotion 为 1，工作区的减弱动态效果样式曾把按钮过渡时间覆盖为 0 秒。已移除按钮颜色与旋钮填充的该项覆盖。Chrome 模拟 reduce 模式实测「修改参数」仍为 2 秒过渡，约 0.6 秒时背景为 rgb(150, 168, 179)，结束为 rgb(242, 247, 250)，移出可平滑恢复。旋钮填充与背景同步。当前聊天无法直接读取 Codex 内置预览的媒体状态。
