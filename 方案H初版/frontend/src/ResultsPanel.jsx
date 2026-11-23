import { useEffect, useState } from "react";
import { CaretLeftIcon, CaretRightIcon, CheckIcon, DownloadSimpleIcon, XIcon } from "@phosphor-icons/react";
import ObservationClock from "./ObservationClock";
import { loadObservationResults, resultFiles } from "./resultsData";
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

function TaskDetails({ tasks }) {
  const [page, setPage] = useState(0);
  const offset = page * pageSize;
  const rows = tasks.slice(offset, offset + pageSize);
  return (
    <section className="result-details" aria-labelledby="result-details-title">
      <div className="result-details__head">
        <h3 id="result-details-title">观测任务明细</h3>
        <div className="result-downloads">
          <a className="result-downloads__plan" href={resultFiles.plan} download>
            <DownloadSimpleIcon aria-hidden="true" size={20} />下载计划 CSV
          </a>
          <a className="result-downloads__summary" href={resultFiles.summary} download>摘要 JSON</a>
        </div>
      </div>
      <table className="result-table" aria-labelledby="result-details-title">
        <thead><tr>
          <th scope="col">目标名称</th><th scope="col">站点</th>
          <th scope="col">观测时段 <span>UTC</span></th><th scope="col">时长</th>
        </tr></thead>
        <tbody>
          {rows.map((task) => (
            <tr key={task.task_id}>
              <td>
                <span className="result-table__name">{task.target_name || task.target_id}</span>
                {task.target_name && <small className="result-table__id" aria-label={`目标编号 ${task.target_id}`}>{task.target_id}</small>}
              </td>
              <td>{stationNumber(task.station_id)}</td>
              <td>
                <span className="result-table__times">
                  <time dateTime={task.observation_start_utc}>{timeOfDay(task.observation_start_utc)}</time>
                  <span className="result-table__dash">–</span>
                  <time dateTime={task.observation_end_utc}>{timeOfDay(task.observation_end_utc)}</time>
                </span>
              </td>
              <td>{durationLabel(task.duration_seconds)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {tasks.length > 0 && (
        <div className="result-pages">
          <span aria-live="polite">{offset + 1}–{offset + rows.length} / {tasks.length} 项</span>
          {tasks.length > pageSize && <div>
            <button type="button" aria-label="上一页任务" disabled={!page} onClick={() => setPage(page - 1)}><CaretLeftIcon aria-hidden="true" /></button>
            <button type="button" aria-label="下一页任务" disabled={offset + rows.length >= tasks.length} onClick={() => setPage(page + 1)}><CaretRightIcon aria-hidden="true" /></button>
          </div>}
        </div>
      )}
    </section>
  );
}

export default function Results() {
  const [result, setResult] = useState(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    loadObservationResults(controller.signal).then(
      (data) => { if (!controller.signal.aborted) setResult(data); },
      () => { if (!controller.signal.aborted) setFailed(true); },
    );
    return () => controller.abort();
  }, []);

  if (failed) return <p role="alert">示例文件暂时无法读取，请关闭面板后重试。</p>;
  if (!result) return <p role="status">正在读取观测计划…</p>;
  const { summary, tasks, stationIds, outages, startUtc } = result;
  const counts = new Map();
  for (const task of tasks) counts.set(task.station_id, (counts.get(task.station_id) || 0) + 1);
  const day = startUtc.slice(0, 10);
  const valid = summary.validation.legal;
  const ValidationIcon = valid ? CheckIcon : XIcon;
  return (
    <div className="plan-results">
      <div className="results-edition">
        <strong>示例结果</strong>
        <time dateTime={startUtc}>{day.replaceAll("-", ".")} UTC</time>
        <span>{summary.algo.replace(/（推荐）$/, "")}</span>
      </div>
      <section className="result-figure" aria-labelledby="result-figure-title">
        <div className="result-figure__clock">
          <ObservationClock tasks={tasks} startUtc={startUtc} stationIds={stationIds} outages={outages} />
        </div>
        <div className="result-caption">
          <h3 id="result-figure-title">双站观测时序</h3>
          <dl className="result-legend">
            {stationIds.map((id) => (
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
            <span data-valid={valid}><ValidationIcon aria-hidden="true" />独立校验{valid ? "通过" : "未通过"}</span>
          </div>
        </div>
      </section>
      <TaskDetails tasks={tasks} />
    </div>
  );
}
