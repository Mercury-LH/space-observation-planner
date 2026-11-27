import { useEffect, useLayoutEffect, useRef, useState } from "react";
import "./ConfigurationPanel.css";
import { WorkspaceAction } from "./WorkspaceAction";
import { WorkspaceDownloadIcon, animateWorkspaceDownload } from "./WorkspaceDownloadIcon";
import { LandscapeMotion } from "./LandscapeMotion";
import { TelescopeMount } from "./TelescopeMount";
import { Meteors } from "./Meteors";
import { ObservatoryNetwork } from "./ObservatoryNetwork";
import { ElevationMenu } from "./ElevationMenu";
import { EarthMark } from "./EarthMark";
import Results from "./ResultsPanel";
import { usePlanner } from "./plannerApi";
import { WorkspaceDrawer } from "./WorkspaceDrawer";
import { AuxiliaryDialog } from "./AuxiliaryDialog";
import { captureWorkspacePill } from "./RunLaunchBadge";
import PlanningWindow from "./PlanningWindow";
import RunStationTable from "./RunStationTable";
import "./RunPanel.css";
import {
  ArrowUpRightIcon,
  ArrowRightIcon,
  UploadSimpleIcon,
  QuestionIcon,
  FileTextIcon,
  CheckIcon,
  PlayIcon,
  GearSixIcon,
  FileArrowDownIcon,
} from "@phosphor-icons/react";

const planningWorkbenchUrl = "#run-status";

const algorithms = {
  H: "价格协调",
  E: "进化权重贪心 + 局部搜索",
  B: "固定优先级贪心",
};
const defaultAlgorithm = "H";

const entries = [
  { id: "run", title: "运行观测规划", icon: PlayIcon },
  { id: "configure", title: "配置与调整", icon: GearSixIcon },
  { id: "results", title: "查看与导出", icon: FileArrowDownIcon },
];
function downloadText(name, text, type = "text/plain;charset=utf-8") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function ConfigurePlan({ configuration, saved, onSave, onRun }) {
  const [date, setDate] = useState(configuration.date);
  const [days, setDays] = useState(configuration.days);
  const [stations, setStations] = useState(configuration.stations);
  const savedAlgorithm = configuration.algo ?? defaultAlgorithm;
  const [algo, setAlgo] = useState(savedAlgorithm);
  const validRange = date >= "2026-07-31" && date <= "2026-08-06"
    && Date.parse(`${date}T00:00:00Z`) + Number(days) * 86400000 <= Date.parse("2026-08-07T00:00:00Z");
  const ready = saved && date === configuration.date && Number(days) === Number(configuration.days)
    && algo === savedAlgorithm && stations.length === configuration.stations.length
    && stations.every((station) => configuration.stations.includes(station));
  const saveInteraction = useRef(null);
  const savedNotice = useRef(null);
  const nextAction = useRef(null);
  useLayoutEffect(() => {
    if (!ready || !saveInteraction.current) return;
    savedNotice.current.dataset.justSaved = "";
    if (saveInteraction.current.moveFocus) nextAction.current.focus({ preventScroll: true });
    saveInteraction.current = null;
  }, [ready]);
  const config = {
    start_utc: `${date}T00:00:00Z`,
    days: Number(days),
    stations: stations.map(Number),
    algo,
    status: "configuration_only",
  };
  return (
    <form
      className="plan-config"
      onSubmit={(e) => {
        e.preventDefault();
        if (!validRange || !stations.length || ready) return;
        saveInteraction.current = { moveFocus: document.activeElement === e.nativeEvent.submitter };
        onSave({ date, days, stations, algo });
      }}
    >
      <fieldset>
        <legend><span className="plan-config__section-title">观测时间</span></legend>
        <PlanningWindow days={days} startDate={date} onDateChange={setDate} onDaysChange={setDays} />
      </fieldset>
      <fieldset>
        <legend>
          <span className="plan-config__section-title">
            参与站点<span className="plan-config__meta">已选 {stations.length} / 10</span>
          </span>
        </legend>
        <div className="plan-config__stations">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((n) => (
            <label key={n}>
              <input
                type="checkbox"
                checked={stations.includes(String(n))}
                onChange={(e) => {
                  const checked = e.target.checked;
                  setStations((s) =>
                    checked
                      ? [...s, String(n)]
                      : s.filter((x) => x !== String(n)),
                  );
                }}
              />
              站点 {String(n).padStart(2, "0")}
            </label>
          ))}
        </div>
      </fieldset>
      <fieldset>
        <legend><span className="plan-config__section-title">规划算法</span></legend>
        <div className="plan-config__algorithms">
          {Object.entries(algorithms).map(([id, name]) => (
            <label key={id}>
              <input type="radio" name="planning-algorithm" value={id}
                checked={algo === id} onChange={() => setAlgo(id)} />
              <span>{id} {name}</span>
            </label>
          ))}
        </div>
      </fieldset>
      <div className="plan-config__footer">
        {ready ? (
          <div className="notice" role="status" ref={savedNotice}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path className="plan-config__saved-check" d="m3 12 6 6L21 6" pathLength="1" stroke="currentColor" strokeWidth="2"
                strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <div>
              <strong>配置已保存</strong>
              <p>{days} 天 · {stations.length} 个站点 · {algo} {algorithms[algo]}</p>
            </div>
          </div>
        ) : (
          <p className="form-hint">
            {!stations.length ? "请至少选择一个参与站点。" : !validRange
              ? "请选择 2026-07-31 至 2026-08-06 范围内的日期与天数。" : "配置仅在本页保留，保存后可下载。"}
          </p>
        )}
        <div className="form-actions">
          {ready ? (
            <>
              <WorkspaceAction className="primary" type="button" onClick={onRun} ref={nextAction}>
                前往运行 <ArrowRightIcon size={20} aria-hidden="true" />
              </WorkspaceAction>
              <WorkspaceAction
                className="secondary"
                type="button"
                onClick={(event) => {
                  animateWorkspaceDownload(event);
                  downloadText("observation-config.json", JSON.stringify(config, null, 2), "application/json");
                }}
              >
                下载配置 <WorkspaceDownloadIcon />
              </WorkspaceAction>
            </>
          ) : (
            <WorkspaceAction className="primary" disabled={!stations.length || !validRange} type="submit">
              保存规划配置 <CheckIcon size={20} aria-hidden="true" />
            </WorkspaceAction>
          )}
        </div>
      </div>
    </form>
  );
}
function RunPlan({ configuration, saved, onConfigure, planner, onResults }) {
  const algo = configuration.algo ?? defaultAlgorithm;
  return (
    <div className="plan-run">
      <section className="run-sheet" aria-labelledby="run-config-heading">
        <header className="run-sheet__head">
          <h3 id="run-config-heading">{saved ? "已保存的规划配置" : "当前默认配置"}</h3>
          <WorkspaceAction type="button" className="run-edit" onClick={onConfigure}>
            <svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              strokeWidth="1.6" strokeLinecap="round">
              <path d="M2.5 7h19M2.5 17h19" />
              <circle cx="8" cy="7" r="2.8" />
              <circle cx="16" cy="17" r="2.8" />
            </svg>
            修改参数
          </WorkspaceAction>
        </header>
        <PlanningWindow days={configuration.days} startDate={configuration.date} />
        <RunStationTable stations={configuration.stations} />
      </section>
      <div className="form-actions">
        <dl className="run-algorithm">
          <dt>规划算法</dt>
          <dd>{algo} {algorithms[algo]}</dd>
        </dl>
        <WorkspaceAction className="primary" type="button"
          disabled={planner.submitting || planner.running} onClick={() => planner.submit(configuration)}>
          {planner.submitting ? "正在提交" : planner.running ? "计算进行中" : planner.run ? "开始新规划" : "开始规划"}
          <PlayIcon size={20} aria-hidden="true" />
        </WorkspaceAction>
      </div>
      <div id="run-status" className="run-status" aria-live="polite">
        {planner.run && <>
          <p>{planner.run.stage} · {planner.run.request.algo} · {planner.run.request.date} 起 {planner.run.request.days} 天</p>
          {["queued", "running"].includes(planner.run.state) && <progress max="1" value={planner.run.progress} aria-label="本次规划阶段" />}
          {planner.run.worker_online === false && planner.running && <p>计算服务暂时离线，任务记录已保留。</p>}
          {planner.run.state === "failed" && <p role="alert">{planner.run.error}。可点击开始新规划重试。</p>}
          {planner.run.state === "ready" && <WorkspaceAction type="button" className="secondary" onClick={onResults}>
            查看本次结果 <ArrowRightIcon size={20} aria-hidden="true" />
          </WorkspaceAction>}
        </>}
        {planner.error && <p role="alert">{planner.error}</p>}
      </div>
    </div>
  );
}
function UploadPanel() {
  const [files, setFiles] = useState([]),
    [preview, setPreview] = useState(""),
    [error, setError] = useState("");
  const readId = useRef(0);
  async function readFiles(event) {
    const id = ++readId.current,
      selected = [...event.target.files];
    setError("");
    setPreview("");
    setFiles(selected);
    if (!selected.length) return;
    if (selected.some((f) => f.size > 10 * 1024 * 1024)) {
      setError("请选择单个不超过 10 MB 的文本文件。");
      return;
    }
    try {
      const text = await selected[0].text();
      if (readId.current === id) setPreview(text.slice(0, 3500));
    } catch {
      if (readId.current === id) setError("文件读取失败，请重新选择。");
    }
  }
  return (
    <>
      <label className="upload-zone">
        <UploadSimpleIcon size={30} weight="light" />
        <strong>选择本地数据文件</strong>
        <span>TXT、TLE、CSV 或 JSON · 每个文件最多 10 MB</span>
        <input
          aria-label="选择本地数据文件"
          type="file"
          accept=".txt,.tle,.csv,.json"
          multiple
          onChange={readFiles}
        />
      </label>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {files.length > 0 && (
        <ul className="file-list">
          {files.map((file, i) => (
            <li key={`${file.name}-${i}`}>
              <FileTextIcon size={19} />
              <span>{file.name}</span>
              <small>{Math.max(1, Math.round(file.size / 1024))} KB</small>
            </li>
          ))}
        </ul>
      )}
      {preview && (
        <>
          <p className="preview-label">{files[0].name} · 内容预览</p>
          <pre className="file-preview">{preview}</pre>
        </>
      )}
      <p className="form-hint">
        文件仅在当前浏览器中读取和预览，不会上传到服务器，也不会自动进入规划计算。
      </p>
    </>
  );
}
function DataNavButton({ label, direction, ...buttonProps }) {
  return (
    <button type="button" className="nav-button" data-direction={direction} {...buttonProps}>
      <svg className="nav-button__icon" width="18" height="18" viewBox="0 0 18 18"
        fill="none" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round"
        strokeLinejoin="round" aria-hidden="true" focusable="false">
        <path d="M2.25 11.5v4h13.5v-4" />
        <path className="nav-button__arrow" d={direction === "down"
          ? "M9 2v8.5 M5.5 7 9 10.5 12.5 7"
          : "M9 12V4 M5.5 7.5 9 4 12.5 7.5"} />
      </svg>
      <span>{label}</span>
    </button>
  );
}

export function App() {
  const planner = usePlanner();
  const [downloadOpen, setDownloadOpen] = useState(false);
  const [panel, setPanel] = useState(null);
  const [configuration, setConfiguration] = useState({
    date: "2026-07-31",
    days: "1",
    stations: ["1", "2"],
    algo: defaultAlgorithm,
  });
  const [saved, setSaved] = useState(false);
  const landscapeRef = useRef(null);
  const mobileSkyRef = useRef(null);
  const telescope = useRef(null);
  const downloadRef = useRef(null);
  const downloadTrigger = useRef(null);
  const workspaceTrigger = useRef(null);
  const runOrigin = useRef(null);
  const PanelContainer = entries.some((entry) => entry.id === panel) ? WorkspaceDrawer : AuxiliaryDialog;
  function openPanel(id, trigger) {
    if (trigger) {
      workspaceTrigger.current = trigger;
      runOrigin.current = captureWorkspacePill(trigger);
    }
    setDownloadOpen(false);
    setPanel(id);
  }
  useEffect(() => {
    if (!downloadOpen) return;
    const closeOutside = (event) => {
      if (!downloadRef.current?.contains(event.target)) setDownloadOpen(false);
    };
    const onEscape = (event) => {
      if (event.key === "Escape") {
        setDownloadOpen(false);
        downloadTrigger.current?.focus();
      }
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("focusin", closeOutside);
    document.addEventListener("keydown", onEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("focusin", closeOutside);
      document.removeEventListener("keydown", onEscape);
    };
  }, [downloadOpen]);
  const titles = {
    run: "运行观测规划",
    configure: "配置与调整",
    results: "查看与导出",
    upload: "上传数据",
    help: "从这里开始",
  };
  function renderWorkspacePage(page) {
    if (page === "run") return (
      <RunPlan
        configuration={configuration}
        saved={saved}
        planner={planner}
        onResults={() => openPanel("results")}
        onConfigure={() => openPanel("configure")}
      />
    );
    if (page === "configure") return (
      <ConfigurePlan
        configuration={configuration}
        saved={saved}
        onSave={(value) => {
          setConfiguration(value);
          setSaved(true);
        }}
        onRun={() => openPanel("run")}
      />
    );
    return <Results planner={planner} onRun={() => openPanel("run")} />;
  }
  const subtitles = {
    upload: "选择文件，先检查数据内容。",
    help: "安排哪个站、在什么时候、观测哪个目标。",
  };
  return (
    <main className="homepage">
      <a className="skip-link" href="#main-entries">
        跳到功能入口
      </a>
      <div className="scene" aria-hidden="true">
        <canvas ref={mobileSkyRef} className="mobile-sky-motion" hidden />
        <div className="scene-plane">
          <img
            className="landscape"
            ref={landscapeRef}
            src="/assets/radio-observatory.webp"
            alt=""
            fetchPriority="high"
          />
          <LandscapeMotion imageRef={landscapeRef} mobileSkyRef={mobileSkyRef} />
          <TelescopeMount ref={telescope} />
        </div>
        <Meteors />
      </div>
      <header className="topbar">
        <a className="brand" href="/" aria-label="观测规划首页">
          <EarthMark />
          <span>观测规划</span>
        </a>
        <nav className="navigation" aria-label="数据操作">
          <div ref={downloadRef} className="nav-item">
            <DataNavButton
              ref={downloadTrigger}
              label="下载模板"
              direction="down"
              aria-expanded={downloadOpen}
              aria-controls={downloadOpen ? "download-menu" : undefined}
              onClick={() => setDownloadOpen((open) => !open)}
            />
            {downloadOpen && (
              <div className="dropdown" id="download-menu">
                <p>输入格式模板</p>
                <button
                  onClick={() => {
                    downloadText(
                      "站址模板.txt",
                      "1: 116.4,39.9,50\n2: 100.0,30.0,3000\n",
                    );
                    setDownloadOpen(false);
                  }}
                >
                  <FileTextIcon size={18} />
                  <span>
                    站址模板<small>TXT · 编号、经纬度、海拔</small>
                  </span>
                </button>
                <button
                  onClick={() => {
                    downloadText("高优先级目标模板.txt", "26382\n16650\n");
                    setDownloadOpen(false);
                  }}
                >
                  <FileTextIcon size={18} />
                  <span>
                    目标优先级模板<small>TXT · 每行一个目标编号</small>
                  </span>
                </button>
                <div className="menu-rule" />
                <button onClick={() => openPanel("results", downloadTrigger.current)}>
                  <ArrowUpRightIcon size={18} />
                  <span>
                    查看运行结果<small>提交计划 JSON、CSV 与校验报告</small>
                  </span>
                </button>
              </div>
            )}
          </div>
          <DataNavButton
            label="上传数据"
            direction="up"
            aria-haspopup="dialog"
            onClick={(event) => openPanel("upload", event.currentTarget)}
          />
        </nav>
        <div className="header-actions">
          <button
            className="icon-button help-button"
            aria-label="使用帮助"
            onClick={(event) => openPanel("help", event.currentTarget)}
          >
            <QuestionIcon size={21} weight="light" />
          </button>
        </div>
      </header>
      <section className="hero-content" aria-label="功能入口">
        <ObservatoryNetwork onStationArrival={() => telescope.current?.receive()} />
        <ElevationMenu entries={entries} onOpen={openPanel} />
      </section>
      {panel && (
        <PanelContainer
          runOrigin={runOrigin.current}
          workbenchUrl={planningWorkbenchUrl}
          workspacePage={panel}
          pageTitles={titles}
          renderPage={renderWorkspacePage}
          title={titles[panel]}
          subtitle={subtitles[panel]}
          onClose={() => setPanel(null)}
          returnFocusRef={workspaceTrigger}
        >
          {panel === "upload" && <UploadPanel />}
          {panel === "help" && (
            <div className="help-content">
              <p>
                先进入<strong>配置与调整</strong>
                ，设置观测时间、参与站点和规划算法，并保存参数。
              </p>
              <p>
                再进入<strong>运行观测规划</strong>
                ，核对刚保存的配置，准备启动计算。
              </p>
              <p>
                通过<strong>查看与导出</strong>
                了解任务表和校验状态。生成计划与通过独立校验是两个步骤。
              </p>
              <p className="form-hint">
                配置在刷新后恢复默认值；已提交的运行可在「查看与导出」中继续查看。
              </p>
              <button type="button"
                className="primary"
                onClick={() => openPanel("run")}
              >
                运行观测规划
                <ArrowUpRightIcon size={18} />
              </button>
            </div>
          )}
        </PanelContainer>
      )}
    </main>
  );
}
