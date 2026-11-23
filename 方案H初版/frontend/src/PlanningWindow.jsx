import { useId, useLayoutEffect, useRef } from "react";
import { CalendarBlankIcon, CaretDownIcon } from "@phosphor-icons/react";
import "./PlanningWindow.css";

const HOUR = 3_600_000;
const EDGE = 8;
const pad = (value) => String(value).padStart(2, "0");

function formatUTCDate(date) {
  if (!Number.isFinite(date.getTime())) return null;
  const year = String(date.getUTCFullYear()).padStart(4, "0");
  return `${year}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}`;
}

function drawRuler(canvas, width, height, days) {
  if (width <= 2 * EDGE || height <= 16) return;
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const pen = canvas.getContext("2d");
  if (!pen) return;
  pen.scale(canvas.width / width, canvas.height / height);

  const left = EDGE;
  const right = width - EDGE;
  const baseline = height - 8;
  const apex = 3;
  const divisions = days === 1 ? 24 : days * 4;
  const majorInterval = days === 1 ? 6 : 4;
  const ink = getComputedStyle(canvas);
  pen.strokeStyle = ink.color;
  pen.lineWidth = width > 600 ? 1.5 : 1.2;
  pen.beginPath();
  pen.moveTo(left, baseline);
  pen.quadraticCurveTo(width / 2, 2 * apex - baseline, right, baseline);
  pen.stroke();

  pen.globalAlpha = .7;
  pen.lineWidth = .85;
  pen.beginPath();
  pen.moveTo(left, baseline);
  pen.lineTo(right, baseline);
  for (let tick = 1; tick < divisions; tick++) {
    const x = left + (right - left) * tick / divisions;
    const major = tick % majorInterval === 0;
    pen.moveTo(x, baseline - (major ? 8 : 4));
    pen.lineTo(x, baseline + (major ? 5 : 0));
  }
  pen.stroke();
  pen.globalAlpha = 1;

  pen.fillStyle = ink.getPropertyValue("--ruler-endpoint").trim();
  for (const x of [left, right]) {
    pen.beginPath();
    pen.arc(x, baseline, Math.min(4.7, Math.max(3, width * .006)), 0, Math.PI * 2);
    pen.fill();
  }
}

function DateValue({ date, showArrow }) {
  if (!date) return "—";
  const [year, month, day] = date.split("-");
  return (
    <time dateTime={`${date}T00:00:00Z`} aria-label={`${date} 00:00 UTC`}>
      <span className="planning-window__numeral">
        {month}.{day}
        {showArrow && <span className="planning-window__arrow" aria-hidden="true">→</span>}
      </span>
      <small>{year} 年 · 00:00 UTC</small>
    </time>
  );
}

export default function PlanningWindow({ days, startDate, onDateChange, onDaysChange }) {
  const dateId = useId();
  const daysId = useId();
  const editable = Boolean(onDateChange && onDaysChange);
  const canvasRef = useRef(null);
  const dayCount = Number(days);
  const totalHours = dayCount * 24;
  const [year, month, day] = startDate.split("-").map(Number);
  const start = new Date(0);
  // Setting the full year also preserves user-entered years from 0001 to 0099.
  start.setUTCFullYear(year, month - 1, day);
  const dateAtHour = (hour) => new Date(start.getTime() + hour * HOUR);
  const endDate = formatUTCDate(dateAtHour(totalHours));
  const labelEvery = dayCount === 1 ? 6 : Math.ceil(dayCount / 4) * 24;
  const labelHours = Array.from({ length: Math.ceil(totalHours / labelEvery) }, (_, i) => i * labelEvery);
  labelHours.push(totalHours);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      drawRuler(canvas, width, height, dayCount);
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [dayCount]);

  const range = endDate
    ? `${startDate} 00:00 至 ${endDate} 00:00 UTC`
    : startDate ? `从 ${startDate} 00:00 UTC 起` : "待选择开始日期";
  return (
    <div className={editable ? "planning-window planning-window--editable" : "planning-window"}>
      {editable ? (
        <dl className="planning-window__summary planning-window__edit-fields">
          <div>
            <dt><label htmlFor={dateId}>开始日期</label></dt>
            <dd>
              <span className="planning-window__control planning-window__control--date">
                <span className="planning-window__date-face" aria-hidden="true">
                  {startDate ? startDate.split("-").slice(1).join(".") : "月.日"}
                </span>
                <input id={dateId} type="date" value={startDate} required
                  onInput={(event) => onDateChange(event.currentTarget.value)} />
                <CalendarBlankIcon aria-hidden="true" />
              </span>
              <small>{startDate ? `${year} 年 · ` : ""}00:00 UTC</small>
            </dd>
          </div>
          <div>
            <dt><label htmlFor={daysId}>规划时长</label></dt>
            <dd>
              <span className="planning-window__control">
                <select id={daysId} value={days} onChange={(event) => onDaysChange(event.currentTarget.value)}>
                  {[1, 2, 3, 4, 5, 6, 7].map((n) => <option key={n} value={n}>{n} 天</option>)}
                </select>
                <CaretDownIcon aria-hidden="true" />
              </span>
            </dd>
          </div>
          <div className="planning-window__end">
            <dt>结束时间</dt>
            <dd><DateValue date={endDate} /></dd>
          </div>
        </dl>
      ) : (
        <dl className="planning-window__summary">
          {[[startDate, "开始时间"], [endDate, "结束时间"]].map(([date, label], i) => (
            <div className="planning-window__date" key={label}>
              <dt>{label}</dt>
              <dd><DateValue date={date} showArrow={i === 0} /></dd>
            </div>
          ))}
          <div className="planning-window__duration">
            <dt>规划时长</dt>
            <dd>{dayCount}<span> 天</span><small>{totalHours} 小时</small></dd>
          </div>
        </dl>
      )}
      <figure className="planning-window__chart" role="img" aria-label={`规划时间范围：${range}，共 ${dayCount} 天`}>
        <canvas className="planning-window__plot" ref={canvasRef} aria-hidden="true" />
        <div className="planning-window__labels" style={{ marginInline: EDGE }} aria-hidden="true">
          {labelHours.map((hour) => {
            const date = dateAtHour(hour);
            const dateLabel = Number.isFinite(date.getTime())
              ? `${pad(date.getUTCMonth() + 1)}/${pad(date.getUTCDate())}`
              : `+${hour / 24} 天`;
            return (
              <span key={hour} style={{ left: `${hour / totalHours * 100}%` }}>
                {dayCount === 1 ? `${pad(hour)}:00` : dateLabel}
              </span>
            );
          })}
        </div>
      </figure>
    </div>
  );
}
