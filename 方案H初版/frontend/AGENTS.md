# Prototype Instructions

## Current integration scope (2026-11-27)

- The user explicitly authorized backend integration in this repository. The historical "Frontend only" boundary below is superseded for this work.
- Preserve accepted appearance and motion. Run submission now uses the same-origin `/api` service, H/E/B, UTC date and stations. Results and immutable downloads belong to the selected persisted run; do not restore the fixed 239-task example as the live result.
- Submission dates are limited to 2026-07-31 through 2026-08-06. A displayed next-day 00:00 is exclusive; exported observation endpoints may not exceed 2026-08-06T23:59:59Z.
- Keep local Skyfield validation visibly distinct from unavailable official forecast validation. API and operational details live in `../docs/v3-integration.md`.

Run the local server yourself and open the preview in the browser available to this environment. Do not give the user server-start instructions when you can run it.

Before making substantial visual changes, use the Product Design plugin's `get-context` skill when the visual source is unclear or no longer matches the current goal. When the user gives durable prototype-specific design feedback, preferences, or decisions, record them in `AGENTS.md`.

When implementing from a selected generated mock, treat that image as the source of truth for layout, component anatomy, density, spacing, color, typography, visible content, and hierarchy.

Build app UI in `src/`. Keep `.openai/hosting.json`, `worker/index.js`, `scripts/prepare-sites-build.mjs`, and `tests/sites-worker.test.mjs` intact so the same local prototype can be handed to Sites. Before a Sites handoff, run `npm run build` and `npm run test:sites`; the build must leave `dist/client/index.html`, `dist/server/index.js`, and `dist/.openai/hosting.json`.

## Confirmed homepage direction (2026-11-14)

- Frontend only. Preserve the existing Streamlit and algorithm files.
- The homepage follows the composition in `references/homepage-sketch.png`. On 2026-11-21 the user requested a flared, dish-shaped ground station. Use `public/assets/radio-observatory.webp` for its night landscape.
- Layout: Download / Upload navigation, observatory at lower left, and three stacked entries on the right. Keep the atmosphere restrained and professional.
- Place **下载模板 / 上传数据** together to the right of the brand, after a short vertical divider. Each transparent button keeps its text and ice-blue tray icon visible. On hover or keyboard focus only the arrow moves toward or away from the tray. The download entry includes a small chevron for the template list. On phones keep the two actions on the second row with 44px hit areas and omit the divider.
- Final confirmed entry responsibilities: **运行观测规划 / 配置与调整 / 查看与导出**. The second entry sets and saves parameters; the first reads the saved parameters for a run; the third displays results and downloads. Do not put initial parameter editing exclusively inside the first entry.
- The three homepage entries open a full-height workspace from the right edge. On desktop its left divider adjusts the width; narrow screens use the full viewport. Switching between configuration and run keeps the workspace open.
- Give the three workspace pages a clear reading order with large titles, legible field and table text, and prominent parameter values. Keep guidance short and group each page around its main task.
- The frontend names the existing planning workbench “规划安排”. Use “进入规划安排” for links to it.
- Current implementation keeps configuration in the current React session; refresh restores defaults. The old workbench link does not transfer configuration or automatically run an algorithm. Keep this boundary explicit until a future integration is requested.
- On 2026-11-21 the user selected the elevation-arc homepage menu (third visual concept). Beneath the existing station map, three horizontal Chinese action labels follow a single fine, graduated silver arc. At rest, all three nodes are open silver rings. Keep labels visible on hover and keyboard focus. Both the node and its label open the corresponding workspace. On phones shorten the arc and keep the dish fully visible beneath it.
- Position the three pills and their graduated silver arc lower on the homepage, leaving clear space below the station map. Give their larger labels the local Resource Han Rounded typeface. On short phones leave clearance between the last entry and the telescope.
- Hovering an entry enlarges its label and marker together. The active marker uses a white face with a blue-black play, settings or export icon and a subtle white glow. Move the highlight to the new entry on each hover or keyboard focus change. Use barisdogansutcu’s fuzzy-bullfrog-72 pill for each label, with its 500 ms ease in both directions, tracking expansion, hover glow and 100 ms press. Under reduced motion, limit travel while retaining the gradual response. Keep the arc and the hit areas stationary.
- Give the brand and both navigation labels a bold weight, with the brand larger than the two actions. Use Novaxlo’s animated blue-and-green Earth Loader as the small brand mark, without a loading caption.
- Play the Milky Way and windblown grass automatically whenever the homepage is visible. Omit scene playback controls. The dish, struts and receiver start tracking together as soon as their layers load. Use a visibly broad sweep (−8° to +6°), eased starts and stops, and short observing pauses. Keep the pedestal and front bearing stationary. Occasionally, when a signal on the map reaches a station, show one clearly visible ice-blue glow on the moving receiver. Both effects are independent of cursor movement and keyboard focus. Bring out the dust and stars of the Milky Way while retaining the dark sky around it. Let meteors appear regularly, with occasional pairs and quiet gaps.
- On narrow screens, put the observatory below the entry list so the telescope stays visible.
- Above the homepage entries, use a transparent dotted world map with clearly traceable, fine blue-gray connecting arcs and small station nodes. Several ice-blue signal points should be visibly in transit at the same time on different arcs, with varied departure times and speeds. Keep the continents prominent against the night sky, especially Asia and Australia.
- User annotation explicitly removed the hero eyebrow, heading, introductory paragraph, entry caption, and decorative footer copy. Do not add slogans, taglines, decorative English, or other unsolicited explanatory small text without the user's approval. Preserve scoped edits rather than adding replacement copy.

## Confirmed run workspace (2026-11-22)

- Keep the homepage brand, Earth mark and their spacing at the same size and position when opening and closing the run workspace. Slide the three workspaces in from the right and settle with a slight spring. In reduced motion, reveal the panel in the same direction with shorter travel.
- Keep the original midnight background (#061422) and its feathered left edge blending into the live observatory. On phones retain the sky-to-midnight gradient under the header. The user allows updates to copy, typography, foreground colors and layout. Keep the telescope visible on desktop and the resize affordance on interaction.
- Give the run page more substance using its actual inputs. Display start and exclusive end as a large month-day interval, with the UTC year and time directly below each date. Set duration alongside it on desktop. Use a single silver arch and two subdued light endpoints on the time ruler. Show 00:00–24:00 for a single day and calendar ticks for longer windows. Below it, show the selected stations with dish glyphs, longitude, latitude and altitude from the competition site catalogue. Set the station table directly on the midnight background with fine row rules. Align “进入规划安排” at the lower right. The run page omits the former heading subtitle and synchronization note.

- When the user opens the run workspace from its homepage pill, carry the visible pill into the exposed left sky, below the brand and above the radio dish (center near 13.2% of viewport width and 24.6% of viewport height at the default sheet width), compressing it into the tilted orbital play mark during travel. Remove the upper-left ticks and short scale arc. Return the mark to its source when closing and keep the brand in place. On phones, dock it in a shallow star field above the title.
- At the annotated 1394 × 954 desktop size, allow the visible orbit to span about 173 × 145 px. Keep the central triangle steady as one bright point with a short tail circles the orbit. Slow the point in reduced motion.

## Confirmed configuration workspace (2026-11-23)

- Follow `references/configure-night-algorithms.png` for 配置与调整. Use the run workspace's midnight surface and feathered sky edge. The homepage settings pill settles into a sliders glyph in orbit above the dish.
- Arrange observation time, participating stations and planning algorithm in reading order. Use a single-choice control for H 价格协调 (default), E 进化权重贪心 + 局部搜索 and B 固定优先级贪心. Save the algorithm with the date, duration and stations. Show the saved algorithm on the run page and include its code in the downloadable configuration.

## Confirmed results workspace (2026-11-23)

- Follow `references/results-observation-clock.png` for 查看与导出. Use a 24-hour UTC double-track clock on the midnight surface, with an animated clockwise sweep over the fixed task arcs. The outer track is station 01 (ice blue), the inner track is station 02 (muted copper). Hatch the recorded outage on the affected track.
- Identify the saved example and its date and algorithm. Show station totals and the independent validation next to the dial. The task detail table uses catalogue names and IDs with exact observation intervals, and offers the matching complete CSV and summary JSON.
- The results pill follows the same flight as the run and configuration pills. Compress the clicked pill into a file-and-down-arrow orbit in the left sky and return it to the pill on close. On mobile give the dock clearance above the title. The outer metal ring on the observation clock must visibly rotate as a whole; long arcs, breaks and a bright edge show its motion while UTC labels and scheduled observation intervals stay anchored.
- Secondary-page text actions use Shakil-Babu’s moody-mule-8. Preserve the original blue fill (#356983 for primary actions and #415f71 for secondary actions) and ice-white hover color (#f2f7fa). Start interpolating the surface, text and icon colors immediately on hover or keyboard focus, with an 800 ms ease transition in both directions. Retain this color interpolation under reduced motion, including the settings icon's knob fill. Keep the two-pixel dashed border in the resting blue and the pagination and close controls compact.
- Opening a secondary workspace pushes it in from the right. The first internal step pushes a new title and its content in together; returning to the original page reverses the direction. Keep the sky, resize handle and close control in place during internal navigation. Exiting returns to the button that opened the workspace.
