(() => {
  const METHOD_ORDER = [
    'APSR', 'AUSteer', 'DiffMean', 'FLAS',
    'GemmaScopeSAE', 'GemmaScopeSAEMaxAUC', 'HiDRA', 'HyperSteer',
    'LAT', 'LinearProbe', 'LoRA', 'LoReFT', 'LsReFT',
    'ODESteer', 'PCA', 'PreferenceVector', 'PromptSteering',
    'Random', 'SFT', 'SPSR', 'SimplePromptSteering',
    'SphericalSteering', 'SteeringVector', 'StepODESteer',
  ];

  const COLORS = [
    '#4477AA', '#EE6677', '#228833', '#CCBB44',
    '#66CCEE', '#AA3377', '#BBBBBB', '#332288',
    '#44AA99', '#117733', '#999933', '#CC6677',
    '#882255', '#6699CC', '#DDCC77', '#AA4499',
    '#0072B2', '#D55E00', '#009E73', '#CC79A7',
    '#E69F00', '#56B4E9', '#6F4E7C', '#5F6B6D',
  ];

  const SYMBOLS = [
    'circle', 'square', 'triangle-up', 'diamond',
    'triangle-down', 'cross', 'x', 'star',
    'hexagon', 'triangle-left', 'triangle-right', 'pentagon',
  ];

  const DASHES = ['solid', 'dash', 'dashdot', 'dot'];

  const CONFIG = {
    responsive: true,
    scrollZoom: false,
    displayModeBar: false,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: { format: 'svg', scale: 1 },
  };

  const CHART_SIZES = {
    'composite-chart': { ratio: 2.18, min: 360, max: 470, narrow: 700 },
    'capability-chart': { ratio: 1.68, min: 500, max: 610, narrow: 1680 },
    'sample-chart': { ratio: 2.08, min: 360, max: 460, narrow: 730 },
    'generalization-chart': { ratio: 1.42, min: 580, max: 710, narrow: 1380 },
    'sensitivity-chart': { ratio: 1.68, min: 560, max: 660, narrow: 1100 },
  };

  const methodIndex = (method) => {
    const index = METHOD_ORDER.indexOf(String(method));
    return index >= 0 ? index : METHOD_ORDER.length;
  };

  const methodStyle = (method) => {
    const index = methodIndex(method);
    return {
      color: COLORS[index % COLORS.length],
      symbol: SYMBOLS[index % SYMBOLS.length],
      dash: DASHES[Math.floor(index / 6) % DASHES.length],
    };
  };

  const sortMethods = (methods) => (
    [...methods].sort((a, b) => methodIndex(a) - methodIndex(b))
  );

  const groupByMethod = (points) => {
    const groups = new Map();
    points.forEach((point) => {
      if (!groups.has(point.method)) groups.set(point.method, []);
      groups.get(point.method).push(point);
    });
    return groups;
  };

  const axisRef = (axis, index) => (index === 0 ? axis : `${axis}${index + 1}`);
  const axisKey = (axis, index) => (index === 0 ? `${axis}axis` : `${axis}axis${index + 1}`);

  const baseAxis = (panel, axis, index, domain) => ({
    domain,
    range: axis === 'x' ? panel.xlim : panel.ylim,
    anchor: axisRef(axis === 'x' ? 'y' : 'x', index),
    showline: true,
    linewidth: 1,
    linecolor: '#777',
    mirror: false,
    ticks: 'outside',
    ticklen: 4,
    tickcolor: '#777',
    tickfont: { size: 11, color: '#444' },
    gridcolor: 'rgba(80, 80, 80, 0.16)',
    griddash: 'dot',
    zeroline: false,
    fixedrange: false,
  });

  const transparentLayout = (height) => ({
    autosize: true,
    height,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Arial, Helvetica, sans-serif', size: 12, color: '#222' },
    hoverlabel: {
      bgcolor: '#fff',
      bordercolor: '#bbb',
      font: { family: 'Arial, Helvetica, sans-serif', size: 12, color: '#222' },
    },
    hovermode: 'closest',
    dragmode: 'pan',
    uirevision: 'steerscope',
  });

  const chartHeight = (chartId, width) => {
    const size = CHART_SIZES[chartId];
    if (!width) {
      const chart = document.getElementById(chartId);
      width = chart?.parentElement?.clientWidth || chart?.clientWidth || 1040;
    }
    if (width < 760) return size.narrow;
    return Math.round(Math.min(size.max, Math.max(size.min, width / size.ratio)));
  };

  const activateChart = async (chartId, shellId, responsiveLayout) => {
    const chart = document.getElementById(chartId);
    const shell = document.getElementById(shellId);

    let lastWidth = 0;
    const resize = async (force = false) => {
      const width = Math.round(shell.clientWidth || chart.clientWidth || 1040);
      if (!width || (!force && Math.abs(width - lastWidth) < 2)) return;
      lastWidth = width;
      const update = {
        width,
        height: chartHeight(chartId, width),
        ...(responsiveLayout ? responsiveLayout(width) : {}),
      };
      if (Plotly.relayout) {
        await Plotly.relayout(chart, update);
        if (Plotly.Plots?.resize) Plotly.Plots.resize(chart);
      } else if (Plotly.Plots?.resize) {
        Plotly.Plots.resize(chart);
      }
    };

    await resize(true);
    shell.classList.add('is-ready');
    let resizeFrame = 0;
    const scheduleResize = () => {
      if (resizeFrame) window.cancelAnimationFrame?.(resizeFrame);
      resizeFrame = window.requestAnimationFrame
        ? window.requestAnimationFrame(() => {
          resizeFrame = 0;
          resize();
        })
        : window.setTimeout(() => resize(), 0);
    };
    if (window.ResizeObserver) {
      const observer = new window.ResizeObserver(scheduleResize);
      observer.observe(shell);
      chart._steerscopeResizeObserver = observer;
    }
    window.addEventListener?.('resize', scheduleResize, { passive: true });
    chart._steerscopeWindowResize = scheduleResize;
  };

  const addReferenceLines = (layout, panel, index) => {
    const xref = axisRef('x', index);
    const yref = axisRef('y', index);
    layout.shapes ??= [];
    (panel.reference_lines || []).forEach((line) => {
      if (line.axis === 'x') {
        layout.shapes.push({
          type: 'line', xref, yref, x0: line.value, x1: line.value,
          y0: panel.ylim[0], y1: panel.ylim[1],
          line: { color: 'rgba(55, 65, 81, 0.35)', width: 1 },
        });
      } else {
        layout.shapes.push({
          type: 'line', xref, yref, x0: panel.xlim[0], x1: panel.xlim[1],
          y0: line.value, y1: line.value,
          line: { color: 'rgba(55, 65, 81, 0.35)', width: 1 },
        });
      }
    });
  };

  const panelTitle = (text, x, y) => ({
    x, y, xref: 'paper', yref: 'paper',
    text: `<b>${text}</b>`,
    showarrow: false,
    xanchor: 'center',
    yanchor: 'bottom',
    font: { size: 13, color: '#222' },
  });

  const stackedDomains = (count, gap = 0.07) => {
    const height = (1 - gap * (count - 1)) / count;
    return Array.from({ length: count }, (_, index) => {
      const top = 1 - index * (height + gap);
      return [top - height, top];
    });
  };

  const betterAnnotation = (index) => ({
    x: 0.98,
    y: 0.04,
    xref: `${axisRef('x', index)} domain`,
    yref: `${axisRef('y', index)} domain`,
    text: 'Better ↘',
    showarrow: false,
    xanchor: 'right',
    font: { size: 10, color: '#2f855a' },
  });

  const legendMarker = (method) => {
    const style = methodStyle(method);
    const dash = {
      solid: '', dash: '6 3', dashdot: '6 2 1 2', dot: '1 3',
    }[style.dash];
    const shapes = {
      circle: '<circle cx="14" cy="7" r="3.5" />',
      square: '<rect x="10.5" y="3.5" width="7" height="7" />',
      'triangle-up': '<path d="M14 2.5 19 10.5H9Z" />',
      diamond: '<path d="m14 2 5 5-5 5-5-5Z" />',
      'triangle-down': '<path d="m9 3.5 10 0-5 8Z" />',
      cross: '<path d="M12 2h4v3h3v4h-3v3h-4V9H9V5h3Z" />',
      x: '<path d="m10 3 4 4 4-4 2 2-4 4 4 4-2 2-4-4-4 4-2-2 4-4-4-4Z" />',
      star: '<path d="m14 1.5 1.7 3.6 4 .5-3 2.8.8 4-3.5-2-3.5 2 .8-4-3-2.8 4-.5Z" />',
      hexagon: '<path d="m10 2.5 8 0 4 4.5-4 4.5h-8L6 7Z" />',
      'triangle-left': '<path d="m9 7 9-5v10Z" />',
      'triangle-right': '<path d="m19 7-9-5v10Z" />',
      pentagon: '<path d="m14 1.8 5 3.6-1.9 5.8h-6.2L9 5.4Z" />',
    };
    return `<svg viewBox="0 0 28 14" aria-hidden="true">
      <line x1="1" y1="7" x2="27" y2="7" stroke="${style.color}" stroke-width="2" stroke-dasharray="${dash}" />
      <g fill="${style.color}" stroke="#fff" stroke-width="0.7">${shapes[style.symbol]}</g>
    </svg>`;
  };

  const renderLegend = (chartId, traces) => {
    const chart = document.getElementById(chartId);
    const shell = chart.parentElement;
    shell.querySelector('.chart-legend')?.remove();

    const legend = document.createElement('div');
    legend.className = 'chart-legend';
    legend.setAttribute('aria-label', 'Chart methods');

    traces.filter((trace) => trace.showlegend).forEach((trace) => {
      const method = trace.meta.method;
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.method = method;
      button.setAttribute('aria-pressed', 'true');
      button.innerHTML = legendMarker(method);

      const label = document.createElement('span');
      label.textContent = trace.name;
      button.append(label);
      button.addEventListener('click', () => {
        const indices = chart.data.reduce((matches, item, index) => {
          if (item.meta?.method === method) matches.push(index);
          return matches;
        }, []);
        const hidden = button.classList.toggle('is-hidden');
        button.setAttribute('aria-pressed', String(!hidden));
        Plotly.restyle(chart, { visible: hidden ? 'legendonly' : true }, indices);
      });
      legend.append(button);
    });

    shell.insertBefore(legend, chart);
  };

  async function loadSpec(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Could not load ${path}: ${response.status}`);
    return response.json();
  }

  async function renderComposite() {
    const spec = await loadSpec('data/composite-tradeoff.json');
    const traces = [];
    const layout = transparentLayout(chartHeight('composite-chart'));
    const domains = [[0, 0.48], [0.52, 1]];

    layout.margin = { l: 72, r: 24, t: 48, b: 52 };
    layout.showlegend = false;
    layout.annotations = [];

    spec.panels.forEach((panel, panelIndex) => {
      const xref = axisRef('x', panelIndex);
      const yref = axisRef('y', panelIndex);
      layout[axisKey('x', panelIndex)] = {
        ...baseAxis(panel, 'x', panelIndex, domains[panelIndex]),
        title: { text: panel.xlabel, standoff: 10, font: { size: 12 } },
      };
      layout[axisKey('y', panelIndex)] = {
        ...baseAxis(panel, 'y', panelIndex, [0, 1]),
        title: panelIndex === 0
          ? { text: panel.ylabel, standoff: 10, font: { size: 12 } }
          : undefined,
        showticklabels: panelIndex === 0,
      };
      layout.annotations.push(
        panelTitle(panel.title, (domains[panelIndex][0] + domains[panelIndex][1]) / 2, 1.04),
        betterAnnotation(panelIndex),
      );
      addReferenceLines(layout, panel, panelIndex);

      panel.data.forEach((series) => {
        const points = series.points || [];
        if (!points.length) return;
        const method = points[0].method;
        const label = points[0].method_display_name || method;
        const style = methodStyle(method);
        traces.push({
          type: 'scatter',
          mode: 'lines+markers',
          name: label,
          meta: { method, panelIndex },
          showlegend: panelIndex === 0,
          xaxis: xref,
          yaxis: yref,
          x: points.map((point) => point[series.x]),
          y: points.map((point) => point[series.y]),
          customdata: points.map((point) => [point.factor, point.n_concepts]),
          line: { color: style.color, width: 2, dash: style.dash },
          marker: {
            color: style.color,
            symbol: style.symbol,
            size: 7,
            line: { color: '#fff', width: 0.7 },
          },
          hovertemplate:
            `<b>${label}</b><br>` +
            'Factor: %{customdata[0]:.3g}<br>' +
            'Concept Expression: %{x:.3f}<br>' +
            'Mean Side Effect: %{y:.3f}<br>' +
            'Concepts: %{customdata[1]}<extra></extra>',
        });
      });
    });

    await Plotly.newPlot('composite-chart', traces, layout, {
      ...CONFIG,
      toImageButtonOptions: { ...CONFIG.toImageButtonOptions, filename: 'steerscope-composite-tradeoff' },
    });
    renderLegend('composite-chart', traces);
    await activateChart('composite-chart', 'composite-shell', (width) => {
      const narrow = width < 760;
      const xDomains = narrow ? [[0, 1], [0, 1]] : domains;
      const yDomains = narrow ? stackedDomains(2, 0.12) : [[0, 1], [0, 1]];
      return {
        'xaxis.domain': xDomains[0],
        'xaxis2.domain': xDomains[1],
        'yaxis.domain': yDomains[0],
        'yaxis2.domain': yDomains[1],
        'yaxis2.showticklabels': narrow,
        margin: narrow
          ? { l: 66, r: 66, t: 48, b: 52 }
          : { l: 72, r: 24, t: 48, b: 52 },
        annotations: spec.panels.flatMap((panel, panelIndex) => [
          panelTitle(
            panel.title,
            (xDomains[panelIndex][0] + xDomains[panelIndex][1]) / 2,
            narrow ? yDomains[panelIndex][1] + 0.018 : 1.04,
          ),
          betterAnnotation(panelIndex),
        ]),
      };
    });
  }

  async function renderCapabilities() {
    const spec = await loadSpec('data/capability-tradeoffs.json');
    const traces = [];
    const layout = transparentLayout(chartHeight('capability-chart'));
    const xDomains = [[0, 0.30], [0.35, 0.65], [0.70, 1]];
    const yDomains = [[0.55, 1], [0, 0.45]];

    layout.margin = { l: 72, r: 24, t: 56, b: 52 };
    layout.showlegend = false;
    layout.annotations = [
      {
        x: -0.075, y: 0.5, xref: 'paper', yref: 'paper',
        text: spec.ylabel, showarrow: false, textangle: -90, font: { size: 12 },
      },
    ];

    spec.panels.forEach((panel, panelIndex) => {
      const column = panelIndex % 3;
      const row = Math.floor(panelIndex / 3);
      const xref = axisRef('x', panelIndex);
      const yref = axisRef('y', panelIndex);
      const xDomain = xDomains[column];
      const yDomain = yDomains[row];

      layout[axisKey('x', panelIndex)] = {
        ...baseAxis(panel, 'x', panelIndex, xDomain),
        showticklabels: row === 1,
        title: panelIndex === 4
          ? { text: spec.xlabel, standoff: 10, font: { size: 12 } }
          : undefined,
      };
      layout[axisKey('y', panelIndex)] = {
        ...baseAxis(panel, 'y', panelIndex, yDomain),
        showticklabels: column === 0,
      };
      layout.annotations.push(
        panelTitle(panel.title, (xDomain[0] + xDomain[1]) / 2, yDomain[1] + 0.025),
        betterAnnotation(panelIndex),
      );
      addReferenceLines(layout, panel, panelIndex);

      const points = panel.data.flatMap((series) => series.points || []);
      const groups = groupByMethod(points);
      sortMethods(groups.keys()).forEach((method) => {
        const methodPoints = groups.get(method);
        const point = methodPoints[methodPoints.length - 1];
        const label = point.method_display_name || method;
        const style = methodStyle(method);
        traces.push({
          type: 'scatter',
          mode: 'markers',
          name: label,
          meta: { method, panelIndex },
          showlegend: panelIndex === 0,
          xaxis: xref,
          yaxis: yref,
          x: [point.effect_percent],
          y: [point.side_effect_percent],
          customdata: [[point.normalized_factor, point.n_concepts]],
          marker: {
            color: style.color,
            symbol: style.symbol,
            size: 10,
            line: { color: '#fff', width: 0.8 },
          },
          hovertemplate:
            `<b>${label}</b><br>` +
            'Normalized factor: %{customdata[0]:.3g}<br>' +
            'Concept Expression gain: %{x:.2f} pp<br>' +
            'Metric degradation: %{y:.2f} pp<br>' +
            'Concepts: %{customdata[1]}<extra></extra>',
        });
      });
    });

    await Plotly.newPlot('capability-chart', traces, layout, {
      ...CONFIG,
      toImageButtonOptions: { ...CONFIG.toImageButtonOptions, filename: 'steerscope-capability-tradeoffs' },
    });
    renderLegend('capability-chart', traces);
    await activateChart('capability-chart', 'capability-shell', (width) => {
      const narrow = width < 760;
      const narrowYDomains = stackedDomains(spec.panels.length, 0.045);
      const update = {
        margin: narrow
          ? { l: 68, r: 68, t: 52, b: 52 }
          : { l: 72, r: 24, t: 56, b: 52 },
      };
      const annotations = [
        {
          x: -0.065, y: 0.5, xref: 'paper', yref: 'paper',
          text: spec.ylabel, showarrow: false, textangle: -90, font: { size: 12 },
        },
      ];

      spec.panels.forEach((panel, panelIndex) => {
        const column = panelIndex % 3;
        const row = Math.floor(panelIndex / 3);
        const xDomain = narrow ? [0, 1] : xDomains[column];
        const yDomain = narrow ? narrowYDomains[panelIndex] : yDomains[row];
        update[`${axisKey('x', panelIndex)}.domain`] = xDomain;
        update[`${axisKey('y', panelIndex)}.domain`] = yDomain;
        update[`${axisKey('x', panelIndex)}.showticklabels`] = narrow || row === 1;
        update[`${axisKey('x', panelIndex)}.title.text`] = (
          narrow ? panelIndex === spec.panels.length - 1 : panelIndex === 4
        ) ? spec.xlabel : '';
        update[`${axisKey('x', panelIndex)}.title.standoff`] = 10;
        update[`${axisKey('y', panelIndex)}.showticklabels`] = narrow || column === 0;
        annotations.push(
          panelTitle(
            panel.title,
            (xDomain[0] + xDomain[1]) / 2,
            yDomain[1] + (narrow ? 0.006 : 0.025),
          ),
          betterAnnotation(panelIndex),
        );
      });
      update.annotations = annotations;
      return update;
    });
  }

  async function renderSampleEfficiency() {
    const spec = await loadSpec('data/sample-efficiency.json');
    const traces = [];
    const layout = transparentLayout(chartHeight('sample-chart'));
    const domains = [[0, 0.48], [0.52, 1]];
    const titles = ['Overall improvement', 'Relative recovery'];

    layout.margin = { l: 72, r: 24, t: 48, b: 52 };
    layout.showlegend = false;
    layout.annotations = [];

    spec.panels.forEach((panel, panelIndex) => {
      const xref = axisRef('x', panelIndex);
      const yref = axisRef('y', panelIndex);
      layout[axisKey('x', panelIndex)] = {
        ...baseAxis(panel, 'x', panelIndex, domains[panelIndex]),
        title: { text: panel.xlabel, standoff: 10, font: { size: 12 } },
      };
      layout[axisKey('y', panelIndex)] = {
        ...baseAxis(panel, 'y', panelIndex, [0, 1]),
        title: { text: panel.ylabel, standoff: 10, font: { size: 12 } },
      };
      layout.annotations.push(
        panelTitle(titles[panelIndex], (domains[panelIndex][0] + domains[panelIndex][1]) / 2, 1.04),
      );

      const series = panel.data[0];
      const groups = groupByMethod(series.points || []);
      sortMethods(groups.keys()).forEach((method) => {
        const points = groups.get(method).sort((a, b) => a.train_examples - b.train_examples);
        const label = points[0].method_display_name || method;
        const style = methodStyle(method);
        traces.push({
          type: 'scatter',
          mode: 'lines+markers',
          name: label,
          meta: { method, panelIndex },
          showlegend: panelIndex === 0,
          xaxis: xref,
          yaxis: yref,
          x: points.map((point) => point.train_examples),
          y: points.map((point) => point[series.y]),
          line: { color: style.color, width: 2, dash: style.dash },
          marker: {
            color: style.color,
            symbol: style.symbol,
            size: 7,
            line: { color: '#fff', width: 0.7 },
          },
          hovertemplate:
            `<b>${label}</b><br>` +
            'Examples per concept: %{x}<br>' +
            (panelIndex === 0
              ? 'Overall improvement: %{y:.3f}'
              : 'Full-data improvement recovered: %{y:.1f}%') +
            '<extra></extra>',
        });
      });
    });

    await Plotly.newPlot('sample-chart', traces, layout, {
      ...CONFIG,
      toImageButtonOptions: { ...CONFIG.toImageButtonOptions, filename: 'steerscope-sample-efficiency' },
    });
    renderLegend('sample-chart', traces);
    await activateChart('sample-chart', 'sample-shell', (width) => {
      const narrow = width < 760;
      const xDomains = narrow ? [[0, 1], [0, 1]] : domains;
      const yDomains = narrow ? stackedDomains(2, 0.12) : [[0, 1], [0, 1]];
      return {
        'xaxis.domain': xDomains[0],
        'xaxis2.domain': xDomains[1],
        'yaxis.domain': yDomains[0],
        'yaxis2.domain': yDomains[1],
        margin: narrow
          ? { l: 72, r: 72, t: 48, b: 52 }
          : { l: 72, r: 24, t: 48, b: 52 },
        annotations: titles.map((title, panelIndex) => panelTitle(
          title,
          (xDomains[panelIndex][0] + xDomains[panelIndex][1]) / 2,
          narrow ? yDomains[panelIndex][1] + 0.018 : 1.04,
        )),
      };
    });
  }

  async function renderGeneralization() {
    const specs = await Promise.all([
      loadSpec('data/generalization-concept-2b.json'),
      loadSpec('data/generalization-concept-9b.json'),
      loadSpec('data/generalization-overall-2b.json'),
      loadSpec('data/generalization-overall-9b.json'),
    ]);
    const titles = [
      'Concept retention · Gemma-2-2B · L20',
      'Concept retention · Gemma-2-9B · L20',
      'Overall retention · Gemma-2-2B · L20',
      'Overall retention · Gemma-2-9B · L20',
    ];
    const xDomains = [[0, 0.47], [0.53, 1]];
    const yDomains = [[0.57, 1], [0, 0.43]];
    const traces = [];
    const layout = transparentLayout(chartHeight('generalization-chart'));
    const legendMethods = new Set();

    layout.margin = { l: 82, r: 24, t: 56, b: 58 };
    layout.showlegend = false;
    layout.annotations = [];
    layout.shapes = [];

    specs.forEach((spec, panelIndex) => {
      const panel = spec.panels[0];
      const column = panelIndex % 2;
      const row = Math.floor(panelIndex / 2);
      const xref = axisRef('x', panelIndex);
      const yref = axisRef('y', panelIndex);
      const xDomain = xDomains[column];
      const yDomain = yDomains[row];

      layout[axisKey('x', panelIndex)] = {
        ...baseAxis(panel, 'x', panelIndex, xDomain),
        title: { text: panel.xlabel, standoff: 10, font: { size: 11 } },
      };
      layout[axisKey('y', panelIndex)] = {
        ...baseAxis(panel, 'y', panelIndex, yDomain),
        title: column === 0
          ? { text: panel.ylabel, standoff: 9, font: { size: 11 } }
          : undefined,
      };
      layout.annotations.push(
        panelTitle(titles[panelIndex], (xDomain[0] + xDomain[1]) / 2, yDomain[1] + 0.022),
      );
      layout.shapes.push({
        type: 'line', xref, yref,
        x0: panel.xlim[0], x1: panel.xlim[1], y0: 1, y1: 1,
        line: { color: 'rgba(55, 65, 81, 0.42)', width: 1, dash: 'dash' },
      });

      const series = panel.data[0];
      const points = series.points || [];
      sortMethods(points.map((point) => point.method)).forEach((method) => {
        const point = points.find((candidate) => candidate.method === method);
        if (!point) return;
        const label = point.method_display_name || method;
        const style = methodStyle(method);
        const showlegend = !legendMethods.has(method);
        legendMethods.add(method);
        traces.push({
          type: 'scatter',
          mode: 'markers',
          name: label,
          meta: { method, panelIndex },
          showlegend,
          xaxis: xref,
          yaxis: yref,
          x: [point[series.x]],
          y: [point[series.y]],
          customdata: [[point.factor, point.n_concepts]],
          marker: {
            color: style.color,
            symbol: style.symbol,
            size: 10,
            line: { color: '#fff', width: 0.8 },
          },
          hovertemplate:
            `<b>${label}</b><br>` +
            'Factor: %{customdata[0]:.3g}<br>' +
            `${panel.xlabel.replace(' ↑', '')}: %{x:.3f}<br>` +
            `${panel.ylabel.replace(' ↑', '')}: %{y:.3f}<br>` +
            'Concepts: %{customdata[1]}<extra></extra>',
        });
      });
    });

    await Plotly.newPlot('generalization-chart', traces, layout, {
      ...CONFIG,
      toImageButtonOptions: { ...CONFIG.toImageButtonOptions, filename: 'steerscope-generalization' },
    });
    renderLegend('generalization-chart', traces);
    await activateChart('generalization-chart', 'generalization-shell', (width) => {
      const narrow = width < 760;
      const narrowYDomains = stackedDomains(specs.length, 0.065);
      const update = {
        margin: narrow
          ? { l: 76, r: 76, t: 52, b: 58 }
          : { l: 82, r: 24, t: 56, b: 58 },
      };
      const annotations = [];

      specs.forEach((spec, panelIndex) => {
        const panel = spec.panels[0];
        const column = panelIndex % 2;
        const row = Math.floor(panelIndex / 2);
        const xDomain = narrow ? [0, 1] : xDomains[column];
        const yDomain = narrow ? narrowYDomains[panelIndex] : yDomains[row];
        update[`${axisKey('x', panelIndex)}.domain`] = xDomain;
        update[`${axisKey('y', panelIndex)}.domain`] = yDomain;
        update[`${axisKey('y', panelIndex)}.title.text`] = narrow || column === 0
          ? panel.ylabel
          : '';
        annotations.push(panelTitle(
          titles[panelIndex],
          (xDomain[0] + xDomain[1]) / 2,
          yDomain[1] + (narrow ? 0.008 : 0.022),
        ));
      });
      update.annotations = annotations;
      return update;
    });
  }

  async function renderSampleSensitivity() {
    const specs = await Promise.all([
      loadSpec('data/sample-sensitivity-2b.json'),
      loadSpec('data/sample-sensitivity-9b.json'),
    ]);
    const titles = ['Gemma-2-2B · L20', 'Gemma-2-9B · L20'];
    const domains = [[0, 0.42], [0.58, 1]];
    const traces = [];
    const layout = transparentLayout(chartHeight('sensitivity-chart'));

    layout.margin = { l: 150, r: 24, t: 72, b: 54 };
    layout.showlegend = false;
    layout.annotations = [];

    specs.forEach((spec, panelIndex) => {
      const panel = spec.panels[0];
      const series = panel.data[0];
      const points = (series.points || []).slice().sort((a, b) => a.score_mean - b.score_mean);
      const xref = axisRef('x', panelIndex);
      const yref = axisRef('y', panelIndex);

      layout[axisKey('x', panelIndex)] = {
        ...baseAxis(panel, 'x', panelIndex, domains[panelIndex]),
        title: {
          text: 'Overall Improvement (mean ± mean concept seed SD)',
          standoff: 10,
          font: { size: 11 },
        },
      };
      layout[axisKey('y', panelIndex)] = {
        domain: [0, 1],
        anchor: xref,
        type: 'category',
        categoryorder: 'array',
        categoryarray: points.map((point) => point.method_display_name || point.method),
        tickfont: { size: 11, color: '#333' },
        showgrid: false,
        showline: true,
        linewidth: 1,
        linecolor: '#777',
        fixedrange: true,
      };
      layout.annotations.push(
        panelTitle(titles[panelIndex], (domains[panelIndex][0] + domains[panelIndex][1]) / 2, 1.035),
        {
          x: 0.02, y: 0.98,
          xref: `${xref} domain`, yref: `${yref} domain`,
          text: '24 examples/concept', showarrow: false,
          xanchor: 'left', yanchor: 'top',
          font: { size: 10, color: '#555' },
        },
      );

      points.forEach((point) => {
        const method = point.method;
        const label = point.method_display_name || method;
        const style = methodStyle(method);
        traces.push({
          type: 'scatter',
          mode: 'markers',
          name: label,
          meta: { method, panelIndex },
          showlegend: false,
          xaxis: xref,
          yaxis: yref,
          x: [point.score_mean],
          y: [label],
          customdata: [[point.score_std, point.train_examples, point.n_concepts]],
          error_x: {
            type: 'data', array: [point.score_std], visible: true,
            color: style.color, thickness: 1.5, width: 4,
          },
          marker: {
            color: style.color,
            symbol: style.symbol,
            size: 10,
            line: { color: '#fff', width: 0.8 },
          },
          hovertemplate:
            `<b>${label}</b><br>` +
            'Mean Overall improvement: %{x:.4f}<br>' +
            'Sample sensitivity: %{customdata[0]:.4f}<br>' +
            'Training examples: %{customdata[1]}<br>' +
            'Concepts: %{customdata[2]}<extra></extra>',
        });
      });
    });

    await Plotly.newPlot('sensitivity-chart', traces, layout, {
      ...CONFIG,
      toImageButtonOptions: { ...CONFIG.toImageButtonOptions, filename: 'steerscope-sample-sensitivity' },
    });
    await activateChart('sensitivity-chart', 'sensitivity-shell', (width) => {
      const narrow = width < 760;
      const xDomains = narrow ? [[0, 1], [0, 1]] : domains;
      const yDomains = narrow ? stackedDomains(2, 0.12) : [[0, 1], [0, 1]];
      const annotations = [];
      const update = {
        margin: narrow
          ? { l: 108, r: 108, t: 54, b: 54 }
          : { l: 150, r: 24, t: 72, b: 54 },
      };

      specs.forEach((spec, panelIndex) => {
        const xref = axisRef('x', panelIndex);
        const yref = axisRef('y', panelIndex);
        update[`${axisKey('x', panelIndex)}.domain`] = xDomains[panelIndex];
        update[`${axisKey('y', panelIndex)}.domain`] = yDomains[panelIndex];
        if (narrow) update[`${axisKey('y', panelIndex)}.tickfont.size`] = 9;
        annotations.push(
          panelTitle(
            titles[panelIndex],
            (xDomains[panelIndex][0] + xDomains[panelIndex][1]) / 2,
            narrow ? yDomains[panelIndex][1] + 0.018 : 1.035,
          ),
          {
            x: 0.02, y: 0.98,
            xref: `${xref} domain`, yref: `${yref} domain`,
            text: '24 examples/concept', showarrow: false,
            xanchor: 'left', yanchor: 'top',
            font: { size: 10, color: '#555' },
          },
        );
      });
      update.annotations = annotations;
      return update;
    });
  }

  async function renderResults() {
    if (!window.Plotly) return;
    const results = await Promise.allSettled([
      renderComposite(),
      renderCapabilities(),
      renderSampleEfficiency(),
      renderGeneralization(),
      renderSampleSensitivity(),
    ]);
    results.forEach((result) => {
      if (result.status === 'rejected') console.warn(result.reason);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderResults);
  } else {
    renderResults();
  }
})();
