(() => {
  const $ = id => document.getElementById(id);
  const view = document.body.dataset.page;
  const captions = {
    scheme_g: '方案 G · AI 筛选与 CP-SAT',
    ai_greedy: 'AI 权重贪心',
    priority_greedy: '优先级贪心',
    random_greedy: '随机基线'
  };
  const storageKey = 'observation-workbench-config';
  const form = $('config-form');
  let defaults;
  let settings;
  let page = 0;
  let tasks = [];
  let poll;
  let lastLog = '';
  let eventRows = new Map();
  let previewSource = null;
  let sceneryStarted = false;
  let sceneryRequested = false;
  const limit = 20;
  const localHelp = '当前工作台需要连接规划服务，才能读取配置、运行任务和查看结果。';
  document.querySelectorAll('.spotlight').forEach(card => {
    card.addEventListener('pointermove', event => {
      const {left, top} = card.getBoundingClientRect();
      card.style.setProperty('--x', `${event.clientX-left}px`);
      card.style.setProperty('--y', `${event.clientY-top}px`);
    }, {passive:true});
  });

  function alignSceneryWindow() {
    const windowFrame = document.querySelector('.day-cycle');
    const backdrop = document.querySelector('.run-sky');
    const sourceWidth = 1672;
    const sourceHeight = 941;
    const [horizontal,vertical] = getComputedStyle(backdrop).backgroundPosition
      .split(/\s+/).map(value => parseFloat(value) / 100);
    function align() {
      const sky = backdrop.getBoundingClientRect();
      const frame = windowFrame.getBoundingClientRect();
      const scale = Math.max(sky.width/sourceWidth,sky.height/sourceHeight);
      const width = sourceWidth*scale;
      const height = sourceHeight*scale;
      const x = sky.left+(sky.width-width)*horizontal-frame.left;
      const y = sky.top+(sky.height-height)*vertical-frame.top;
      windowFrame.style.setProperty('--sky-size', `${width}px ${height}px`);
      windowFrame.style.setProperty('--sky-offset', `${x}px ${y}px`);
    }
    const observer = new ResizeObserver(align);
    observer.observe(backdrop);
    observer.observe(windowFrame);
    windowFrame.closest('.run-overview').addEventListener('scroll', align, {passive:true});
    align();
  }

  function playScenery(config) {
    const first = Date.parse(config.start);
    const last = Date.parse(config.end);
    if (!Number.isFinite(first) || !Number.isFinite(last) || last < first) return;
    const oneDay = 86400000;
    const startDay = Math.floor(first/oneDay);
    const days = Math.floor(last/oneDay)-startDay+1;
    const frame = document.querySelector('.day-cycle');
    const layer = frame.querySelector('.day-cycle-day');
    const date = $('scenery-date');
    const counter = $('scenery-count');
    const duration = Math.max(.8, Math.min(3,12/days));
    function showDay(index) {
      const iso = new Date((startDay+index)*oneDay).toISOString().slice(0,10);
      date.dateTime = iso;
      date.textContent = `${iso} UTC`;
      counter.textContent = `${index+1} / ${days}`;
      date.parentElement.hidden = false;
    }
    frame.removeAttribute('data-playing');
    layer.onanimationiteration = null;
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      date.dateTime = new Date(first).toISOString().slice(0,10);
      date.textContent = `${date.dateTime} 至 ${new Date(last).toISOString().slice(0,10)} UTC`;
      counter.textContent = `${days} 天`;
      date.parentElement.hidden = false;
      return;
    }
    frame.style.setProperty('--day-count', days);
    frame.style.setProperty('--day-duration', `${duration}s`);
    // Commit the removal before starting the animation again for a new run.
    void layer.offsetWidth;
    showDay(0);
    layer.onanimationiteration = event => {
      showDay(Math.min(days-1, Math.round(event.elapsedTime/duration)));
    };
    frame.setAttribute('data-playing','');
  }

  async function api(path, options) {
    if (location.protocol === 'file:') throw new Error(localHelp);
    let response;
    try {
      response = await fetch(path, {cache:'no-store', ...options});
    } catch {
      throw new Error(localHelp);
    }
    const data = await response.json().catch(() => { throw new Error(localHelp); });
    if (!response.ok) throw new Error(data.error || '请求失败。');
    return data;
  }

  function savedSettings() {
    try {
      return JSON.parse(localStorage.getItem(storageKey)) || {};
    } catch {
      return {};
    }
  }

  function definition(term, value) {
    const row = document.createElement('div');
    const label = document.createElement('dt');
    const detail = document.createElement('dd');
    label.textContent = term;
    detail.textContent = value;
    row.append(label, detail);
    return row;
  }

  function describe(config) {
    const time = value => value ? value.replace('T', ' ').replace(/Z|\+00:00$/, '') : '—';
    $('run-settings').replaceChildren(
      definition('规划器', captions[config.planner] || config.planner),
      definition('规划开始（UTC）', time(config.start)),
      definition('规划结束（UTC）', time(config.end)),
      definition('最低仰角', `${config.elevation}°`),
      definition('目标 / 站点上限', `${config.targets} / ${config.stations}`)
    );
  }

  function fillForm(config) {
    for (const [name, value] of Object.entries(config)) {
      const field = form.elements.namedItem(name);
      if (field) field.value = field.type === 'datetime-local' ? new Date(value).toISOString().slice(0,16) : value;
    }
  }

  function renderEvents(state, samples) {
    const list = $('run-events');
    $('events-idle').hidden = state !== 'idle';
    const events = state === 'idle' ? [] : [{id:'start', label:'规划器已启动'}];
    for (const [,processed,total] of samples.slice(-4)) {
      events.push({id:`window-${processed}-${total}`, label:'已计算可见窗口的目标与站点组合', detail:`${processed} / ${total}`});
    }
    const latest = samples.at(-1);
    if (state === 'running' && latest && latest[1] === latest[2]) {
      events.push({id:'solve', label:'可见窗口已计算完毕，等待任务分配与校验'});
    }
    if (state === 'complete') events.push({id:'end', label:'任务表与校验结果已生成'});
    if (state === 'failed') events.push({id:'error', label:'规划运行失败，展开原始日志查看原因'});

    const nextRows = new Map();
    events.forEach(({id,label,detail}, index) => {
      if (nextRows.has(id)) return;
      const existing = eventRows.get(id);
      const row = existing || document.createElement('li');
      if (!existing) {
        row.className = 'run-event';
        const text = document.createElement('span');
        text.textContent = label;
        row.append(text);
        if (detail) {
          const counter = document.createElement('small');
          counter.textContent = detail;
          row.append(counter);
        }
        list.append(row);
      }
      row.toggleAttribute('data-live', state === 'running' && index === events.length - 1);
      row.toggleAttribute('data-terminal', id === 'end');
      row.toggleAttribute('data-failed', id === 'error');
      nextRows.set(id, row);
    });
    for (const [id,row] of eventRows) if (!nextRows.has(id)) row.remove();
    eventRows = nextRows;
  }

  function renderProgress(job) {
    const {state} = job;
    document.body.dataset.runState = state;
    if (!sceneryStarted && (state === 'running' || (sceneryRequested && state === 'complete'))) {
      sceneryStarted = true;
      sceneryRequested = false;
      if (job.settings) playScenery(job.settings);
    }
    const labels = {idle:'待运行',running:'运行中',complete:'已完成',failed:'运行失败'};
    $('run-state').textContent = labels[state];
    $('start-run').disabled = state === 'running';
    $('view-output').hidden = state !== 'complete';
    describe(state === 'running' && job.settings ? job.settings : settings);
    const log = job.logs.join('\n') || (state === 'running' ? '正在启动规划器…' : '等待启动。');
    if (log !== lastLog) {
      const panel = $('run-log');
      const pinned = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 20;
      panel.textContent = log;
      if (pinned) panel.scrollTop = panel.scrollHeight;
      lastLog = log;
    }
    const matches = [...new Map(
      [...log.matchAll(/可见窗口计算:\s*(\d+)\/(\d+)/g)].map(match => [match[0],match])
    ).values()];
    renderEvents(state, matches);
    const counts = matches.at(-1);
    const percent = counts ? Math.round(Number(counts[1])/Number(counts[2])*100) : 0;
    const stage = state === 'complete' ? 3 : percent === 100 ? 1 : 0;
    $('run-progress').value = state === 'complete' ? 100 : percent;
    $('progress-message').textContent = state === 'complete' ? '任务表和校验结果已生成。'
      : state === 'failed' ? '请查看日志中的失败原因。'
      : state === 'idle' ? '启动后显示规划器报告的进度。'
      : percent === 100 ? '窗口计算完成，等待规划器生成并校验任务表。'
      : counts ? `已处理 ${counts[1]} / ${counts[2]} 个目标与站点组合。` : '正在读取输入并准备计算可见窗口。';
    document.querySelectorAll('.stages li').forEach((item,index) => {
      item.toggleAttribute('data-active', state === 'running' && index === stage);
      item.toggleAttribute('data-done', state !== 'idle' && index < stage);
    });
  }

  function setRunDetail(showPreview) {
    const preview = $('run-preview');
    if (preview.hidden === !showPreview) return;
    preview.hidden = !showPreview;
    $('monitor-log').hidden = showPreview;
    $('show-log').setAttribute('aria-pressed', String(!showPreview));
    $('show-preview').setAttribute('aria-pressed', String(showPreview));
    document.querySelector('.monitor-content').scrollTop = 0;
  }

  async function renderPreview(job) {
    if (job.state !== 'complete' || !job.source) {
      setRunDetail(false);
      $('show-preview').disabled = true;
      previewSource = null;
      return;
    }
    if (previewSource === job.source) return;
    previewSource = job.source;
    const source = encodeURIComponent(job.source);
    const href = `results.html?source=${source}`;
    $('view-output').href = href;
    try {
      const result = await api(`/api/results?source=${source}`);
      if (previewSource !== job.source) return;
      const {metrics,tasks} = result;
      const errors = metrics.validation_errors;
      const validation = Array.isArray(errors) ? (errors.length ? `有 ${errors.length} 项校验错误` : '约束校验通过') : '无校验数据';
      $('preview-summary').textContent = `${tasks.length} 条任务 · 覆盖 ${metrics.covered_targets ?? '—'} 个目标 · ${validation}`;
      $('preview-rows').replaceChildren(...tasks.slice(0,5).map(taskRow));
      $('preview-table').hidden = !tasks.length;
      $('show-preview').disabled = false;
      if (!$('monitor-log').contains(document.activeElement)) setRunDetail(true);
    } catch (error) {
      if (previewSource === job.source) $('run-feedback').textContent = error.message;
    }
  }

  async function status() {
    try {
      const job = await api('/api/run');
      renderProgress(job);
      if (job.state === 'running') poll = setTimeout(status, 1000);
      await renderPreview(job);
    } catch (error) {
      $('run-feedback').textContent = error.message;
      $('start-run').disabled = false;
    }
  }

  async function start() {
    clearTimeout(poll);
    $('start-run').disabled = true;
    $('run-feedback').textContent = '';
    try {
      await api('/api/run', {
        method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(settings)
      });
      sceneryStarted = false;
      sceneryRequested = true;
      $('run-events').replaceChildren();
      eventRows.clear();
    } catch (error) {
      $('run-feedback').textContent = error.message;
    }
    await status();
  }

  function taskRow(task) {
    const row = document.createElement('tr');
    const labels = {HIGH:'高', MEDIUM:'中', LOW:'低'};
    const cleanTime = value => value?.replace('T',' ').replace(/Z$/, '') || '—';
    for (const value of [task.target_id,task.station_id,labels[task.priority] || task.priority,
      cleanTime(task.start),cleanTime(task.end),`${task.duration_seconds} 秒`]) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  }

  function renderTasks() {
    const search = $('task-search').value.trim().toLowerCase();
    const priority = $('priority-filter').value;
    const visible = tasks.filter(task =>
      (!priority || priority === task.priority) &&
      [task.target_id, task.station_id].some(value => String(value).toLowerCase().includes(search))
    );
    const pages = Math.max(1,Math.ceil(visible.length/limit));
    page = Math.min(page,pages-1);
    const rows = visible.slice(page*limit,(page+1)*limit).map(taskRow);
    if (!rows.length) {
      const row = document.createElement('tr');
      const cell = row.insertCell();
      cell.colSpan = 6;
      cell.textContent = '没有匹配的任务。';
      rows.push(row);
    }
    $('task-rows').replaceChildren(...rows);
    $('task-count').textContent = `${visible.length} / ${tasks.length} 项`;
    $('page-label').textContent = `${page+1} / ${pages}`;
    $('prev').disabled = page === 0;
    $('next').disabled = page >= pages-1;
  }

  function renderResults(result) {
    tasks = result.tasks;
    const metrics = result.metrics;
    const errors = metrics.validation_errors;
    const validation = !Array.isArray(errors) ? '无校验数据' : errors.length ? '有错误' : '通过';
    $('result-source').hidden = true;
    $('result-provenance').textContent = `结果来源 ${result.source}`;
    $('metrics').replaceChildren(...[
      ['观测任务', metrics.task_count ?? tasks.length],
      ['覆盖目标', metrics.covered_targets ?? '—'],
      ['综合得分', Number.isFinite(metrics.composite_score) ? metrics.composite_score.toFixed(4) : '—'],
      ['约束校验', validation]
    ].map(([term, value],index) => {
      const card = document.createElement('dl');
      card.className = 'panel metric';
      card.style.setProperty('--delay', `${index*55}ms`);
      card.append(definition(term,value));
      return card;
    }));
    const messages = Array.isArray(errors) ? (errors.length ? errors : ['规划器未报告约束错误。']) : ['该结果未包含约束校验详情。'];
    $('validation-rows').replaceChildren(...messages.map(message => {
      const item = document.createElement('li');
      item.textContent = typeof message === 'string' ? message : JSON.stringify(message);
      return item;
    }));
    document.querySelectorAll('#exports a').forEach(link => {
      link.href += `?source=${encodeURIComponent(result.source)}`;
    });
    $('exports').hidden = false;
    $('result-content').hidden = false;
    renderTasks();
  }

  async function main() {
    try {
      if (view === 'results') {
        const source = new URLSearchParams(location.search).get('source');
        renderResults(await api(source ? `/api/results?source=${encodeURIComponent(source)}` : '/api/results'));
        for (const key of ['task-search','priority-filter']) $(key).addEventListener('input', () => { page=0; renderTasks(); });
        $('prev').addEventListener('click', () => { page--; renderTasks(); });
        $('next').addEventListener('click', () => { page++; renderTasks(); });
        return;
      }
      defaults = await api('/api/defaults');
      settings = {...defaults,...savedSettings()};
      if (view === 'run') {
        alignSceneryWindow();
        $('start-run').addEventListener('click', start);
        $('show-log').addEventListener('click', () => setRunDetail(false));
        $('show-preview').addEventListener('click', () => setRunDetail(true));
        await status();
        return;
      }
      fillForm(settings);
      $('reset-config').addEventListener('click', () => {
        fillForm(defaults);
        $('config-feedback').textContent = '已填入默认值，保存后生效。';
      });
      form.addEventListener('submit', event => {
        event.preventDefault();
        const fields = Object.fromEntries(new FormData(form));
        if (fields.end <= fields.start) {
          $('config-feedback').textContent = '结束时间必须晚于开始时间。';
          return;
        }
        const config = {...fields, start:`${fields.start}:00Z`, end:`${fields.end}:00Z`,
          elevation:Number(fields.elevation), targets:Number(fields.targets), stations:Number(fields.stations)};
        try {
          localStorage.setItem(storageKey, JSON.stringify(config));
          $('config-feedback').textContent = '配置已保存。';
        } catch {
          $('config-feedback').textContent = '浏览器无法保存配置。';
        }
      });
    } catch (error) {
      if (view === 'results') $('result-source').textContent = error.message;
      else $('connection').textContent = error.message;
    }
  }

  main();
})();
