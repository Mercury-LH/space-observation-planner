import { useEffect, useState } from "react";
import { CaretLeftIcon, CaretRightIcon, CheckIcon, XIcon } from "@phosphor-icons/react";
import ObservationClock from "./ObservationClock";
import { WorkspaceAction } from "./WorkspaceAction";
import { WorkspaceDownloadIcon, animateWorkspaceDownload } from "./WorkspaceDownloadIcon";
import { api } from "./plannerApi";
import "./ResultsPanel.css";

const pageSize = 6;
const timeOfDay = (utc) => utc.slice(11, 19);
const stationNumber = (id) => String(id).padStart(2, "0");

function durationLabel(value) {
  const seconds = Number(value);
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  if (!minutes) return `${seconds} 秒`;
  return remainder ? `${minutes} 分 ${remainder} 秒` : `${minutes} 分钟`;
}

function TaskDetails({ tasks, files }) {
  const [{ page, direction }, setPagination] = useState({ page: 0, direction: null });
  const offset = page * pageSize;
  const rows = tasks.slice(offset, offset + pageSize);
  const lastPage = Math.max(0, Math.ceil(tasks.length / pageSize) - 1);

  function turnPage(step) {
    setPagination((current) => {
      const next = current.page + step;
      if (next < 0 || next > lastPage) return current;
      return { page: next, direction: step > 0 ? "next" : "previous" };
    });
  }

  return (
    <section className="result-details" aria-labelledby="result-details-title">
      <div className="result-details__head">
        <h3 id="result-details-title">观测任务明细</h3>
        <div className="result-downloads">
          <WorkspaceAction as="a" className="result-downloads__plan" href={files.submission}
            download onClick={animateWorkspaceDownload}>
            <WorkspaceDownloadIcon />提交计划 JSON
          </WorkspaceAction>
          <WorkspaceAction as="a" className="result-downloads__summary" href={files.csv}
            download onClick={animateWorkspaceDownload}>
            <WorkspaceDownloadIcon />计划 CSV
          </WorkspaceAction>
          <WorkspaceAction as="a" className="result-downloads__summary" href={files.summary}
            download onClick={animateWorkspaceDownload}>
            <WorkspaceDownloadIcon />摘要 JSON
          </WorkspaceAction>
        </div>
      </div>
      <table className="result-table" aria-labelledby="result-details-title">
        <thead><tr>
          <th scope="col">目标名称</th><th scope="col">站点</th>
          <th scope="col">观测时段 <span>UTC</span></th><th scope="col">时长</th>
        </tr></thead>
        <tbody key={page} data-turn={direction}>
          {rows.map((task) => (
            <tr key={task.task_id}>
              <td>
                <span className="result-table__name">{task.target_name || task.target_id}</span>
                {task.target_name && <small className="result-table__id" aria-label={`目标编号 ${task.target_id}`}>{task.target_id}</small>}
              </td>
              <td>{stationNumber(task.station_id)}</td>
              <td>
                <small>{task.observation_start_utc.slice(5, 10)} → {task.observation_end_utc.slice(5, 10)}</small>
                <span className="result-table__times">
                  <time dateTime={task.observation_start_utc}>{timeOfDay(task.observation_start_utc)}</time>
                  <span className="result-table__dash">–</span>
                  <time dateTime={task.observation_end_utc}>{timeOfDay(task.observation_end_utc)}</time>
                </span>
              </td>
              <td>{durationLabel(task.duration_seconds)}
                <small>{task.validation_status ? ({ normal: "正常", soft: "软错误", hard: "硬错误", unresolved: "待确认" })[task.validation_status] : "未校验"}</small>
                {task.validation_reasons?.length > 0 && <small>{task.validation_reasons.join("；")}</small>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {tasks.length > 0 && (
        <div className="result-pages">
          <span aria-live="polite">{offset + 1}–{offset + rows.length} / {tasks.length} 项</span>
          {tasks.length > pageSize && <div>
            <WorkspaceAction type="button" aria-label="上一页任务"
              aria-disabled={page === 0} tabIndex={page === 0 ? -1 : 0}
              data-turn="previous" onClick={() => turnPage(-1)}>
              <CaretLeftIcon aria-hidden="true" />
            </WorkspaceAction>
            <WorkspaceAction type="button" aria-label="下一页任务"
              aria-disabled={page === lastPage} tabIndex={page === lastPage ? -1 : 0}
              data-turn="next" onClick={() => turnPage(1)}>
              <CaretRightIcon aria-hidden="true" />
            </WorkspaceAction>
          </div>}
        </div>
      )}
    </section>
  );
}

export default function Results({ planner, onRun }) {
  const [result, setResult] = useState(null);
  const [failed, setFailed] = useState("");
  const [retry, setRetry] = useState(0);
  const [day, setDay] = useState("");
  const [pair, setPair] = useState(0);
  const { run, history } = planner;
  useEffect(() => {
    setResult(null);
    setFailed("");
    if (!run?.revision) return;
    const controller = new AbortController();
    api(`/runs/${run.id}/result`, { signal: controller.signal }).then(
      (data) => { if (!controller.signal.aborted) { setResult(data); setDay(data.startUtc.slice(0, 10)); setPair(0); } },
      error => { if (!controller.signal.aborted) setFailed(error.message); },
    );
    return () => controller.abort();
  }, [run?.id, run?.revision, retry]);

  const historyControl = <div className="result-filters"><label>运行记录
    <select aria-label="选择运行记录" value={run?.id ?? ""} onChange={event => planner.select(history.find(item => item.id === event.target.value))}>
      {!run && <option value="">尚无运行记录</option>}
      {history.map(item => <option key={item.id} value={item.id}>
        {item.request.algo} · {item.request.date} · {new Date(item.created * 1000).toLocaleString()} · {item.state === "ready" ? "已完成" : item.stage}
      </option>)}
    </select>
  </label><WorkspaceAction type="button" onClick={onRun}>前往运行</WorkspaceAction></div>;
  if (failed) return <>{historyControl}<p role="alert">{failed}</p><WorkspaceAction type="button" onClick={() => setRetry(value => value + 1)}>重新读取</WorkspaceAction></>;
  if (!run?.revision) return <>{historyControl}<p role="status">{run ? run.error || run.stage : "尚无观测计划。保存配置后即可开始规划。"}</p>{planner.error && <p role="alert">{planner.error}</p>}</>;
  if (!result) return <>{historyControl}<p role="status">正在读取观测计划…</p></>;
  const { summary, tasks, stationIds, outages, startUtc } = result;
  const visibleStations = stationIds.slice(pair * 2, pair * 2 + 2);
  const dayStart = Date.parse(`${day}T00:00:00Z`);
  const visibleTasks = tasks.filter(task => Date.parse(task.observation_start_utc) < dayStart + 86400000
    && Date.parse(task.observation_end_utc) > dayStart);
  const counts = new Map();
  for (const task of visibleTasks) counts.set(task.station_id, (counts.get(task.station_id) || 0) + 1);
  const valid = run.validation_state === "passed";
  const validating = ["queued", "running"].includes(run.validation_state);
  const ValidationIcon = valid ? CheckIcon : XIcon;
  return (
    <div className="plan-results">
      {historyControl}
      <div className="results-edition">
        <strong>本次结果 · {run.id.slice(0, 8)}</strong>
        <time dateTime={startUtc}>{startUtc.slice(0, 10).replaceAll("-", ".")} 起 {summary.request.days} 天 UTC</time>
        <span>{summary.algo.replace(/（推荐）$/, "")}</span>
      </div>
      <div className="result-filters">
        <label>时序日期 <select aria-label="时序日期" value={day} onChange={event => setDay(event.target.value)}>
          {Array.from({ length: summary.request.days }, (_, index) => new Date(Date.parse(startUtc) + index * 86400000).toISOString().slice(0, 10))
            .map(date => <option key={date}>{date}</option>)}
        </select></label>
        <label>观测站 <select aria-label="时序站点" value={pair} onChange={event => setPair(Number(event.target.value))}>
          {Array.from({ length: Math.ceil(stationIds.length / 2) }, (_, index) => <option key={index} value={index}>
            {stationIds.slice(index * 2, index * 2 + 2).map(id => `站点 ${stationNumber(id)}`).join(" / ")}
          </option>)}
        </select></label>
      </div>
      <section className="result-figure" aria-labelledby="result-figure-title">
        <div className="result-figure__clock">
          <ObservationClock tasks={tasks} startUtc={`${day}T00:00:00Z`} stationIds={visibleStations} outages={outages} />
        </div>
        <div className="result-caption">
          <h3 id="result-figure-title">当日观测时序</h3>
          <dl className="result-legend">
            {visibleStations.map((id) => (
              <div key={id}>
                <dt><i className="result-legend__swatch" aria-hidden="true" />站点 {stationNumber(id)}</dt>
                <dd>{counts.get(id) ?? 0}<small> 项</small></dd>
              </div>
            ))}
          </dl>
          {outages.map((outage) => (
            <p className="result-outage" key={outage.stationId + outage.startUtc}>
              <i aria-hidden="true" /><span>站点 {stationNumber(outage.stationId)} 停机</span>
              <span>{timeOfDay(outage.startUtc).slice(0, 5)}–{timeOfDay(outage.endUtc).slice(0, 5)} UTC</span>
            </p>
          ))}
          <div className="result-verdict">
            <p>覆盖 <strong>{summary.metrics.covered}</strong> 个目标</p>
            <span data-valid={valid}><ValidationIcon aria-hidden="true" />{valid ? "本地独立校验通过" : validating ? "独立校验进行中" : run.validation_state === "issues" ? "校验发现问题" : "尚未通过独立校验"}</span>
          </div>
        </div>
      </section>
      <div className="result-validation">
        <WorkspaceAction type="button" disabled={validating || valid || run.validation_state === "issues"} onClick={planner.validate}>
          {validating ? "正在独立校验" : valid || run.validation_state === "issues" ? "校验已完成" : "运行独立校验"}
        </WorkspaceAction>
        {result.files.validation && <WorkspaceAction as="a" href={result.files.validation} download onClick={animateWorkspaceDownload}>
          <WorkspaceDownloadIcon />逐任务校验报告
        </WorkspaceAction>}
        {summary.v3 && <p>本地 v3 得分 {summary.v3.final_score === null ? "待确认" : summary.v3.final_score.toFixed(3)} · 硬错误 {summary.v3.hard_errors} · 软错误 {summary.v3.soft_errors} · 待确认 {summary.v3.unresolved}</p>}
        <p className="form-hint">{summary.forecast_source ? "预报窗口由本地 Skyfield 独立计算；尚未与官方预报文件核对。" : "计划已生成，独立校验尚未完成。"}</p>
        <p className="form-hint">可提交目标 {summary.notes.eligible_targets} 个；另有 {summary.notes.excluded_id_below_10000 + summary.notes.excluded_id_above_99999} 个原始编号不满足五位整数要求。</p>
        {run.validation_error && <p role="alert">{run.validation_error}</p>}
        {planner.error && <p role="alert">{planner.error}</p>}
      </div>
      <TaskDetails key={run.id} tasks={tasks} files={result.files} />
    </div>
  );
}
