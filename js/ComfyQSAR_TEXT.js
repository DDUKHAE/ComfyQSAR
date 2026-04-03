/**
 * ComfyQSAR_TEXT.js  (전면 개선판)
 *
 * 개선 사항:
 *   1. 로그 창 토글 — [▼ 로그] / [▶ 로그] 버튼으로 숨기기/보이기
 *   2. 줄별 색상 — 라이트 테마 기반 색상 (밝은 배경 위 명도 조정)
 *   3. 요약 배지 — 통과/차단 수를 토글 버튼 옆에 항상 표시 (숨겨도 보임)
 *   4. 노드 크기 자동 조절 — 위젯 추가 전 baseH를 1회 스냅샷하여
 *                             열기/닫기 시 오차 없이 node.setSize() 강제 적용
 *
 * 색상/테마: ComfyQSAR_TEXT.js 라이트 테마 계승
 *   - 배경: #f9f9f9 (로그 패널), #f0f0f0 (토글 바)
 *   - 글자: 밝은 배경 위 가독성을 위해 어두운 색상 계열 사용
 *
 * 기능: protein_log_display.js 계승
 *   - 토글 바 (열기/닫기)
 *   - 줄별 색상 규칙 (이모지/키워드 기반)
 *   - 요약 배지 (통과/차단)
 *   - 노드 크기 자동 조절 (setSize 강제 적용)
 */

import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// ══════════════════════════════════════════════════════════════════
// 상수
// ══════════════════════════════════════════════════════════════════

const TARGET_NODES = Object.freeze([
  // Regression
  "Data_Loader_Regression",
  "Standardization_Regression",
  "Load_and_Standardize_Regression",
  "Descriptor_Calculations_Regression",
  "Replace_inf_with_nan_Regression",
  "Remove_high_nan_compounds_Regression",
  "Remove_high_nan_descriptors_Regression",
  "Impute_missing_values_Regression",
  "Descriptor_preprocessing_Regression",
  "Remove_Low_Variance_Descriptors_Regression",
  "Remove_High_Correlation_Features_Regression",
  "Descriptor_Optimization_Regression",
  "LassoFeatureSelection",
  "DecisionTreeFeatureSelection",
  "RandomForestFeatureSelection",
  "XGBoostFeatureSelection",
  "LightGBMFeatureSelection",
  "SelectFromModelFeatureSelection",
  "RFEFeatureSelection",
  "Regression_Feature_Combination_Search",
  "Hyperparameter_Grid_Search_Regression",
  "Model_Validation_Regression",
  "QSARDataSplit_Regression",

  // Classification
  "Data_Loader_Classification",
  "Standardization_Classification",
  "Load_and_Standardize_Classification",
  "Descriptor_Calculations_Classification",
  "Replace_inf_with_nan_Classification",
  "Remove_high_nan_compounds_Classification",
  "Remove_high_nan_descriptors_Classification",
  "Impute_missing_values_Classification",
  "Descriptor_preprocessing_Classification",
  "Remove_Low_Variance_Descriptors_Classification",
  "Remove_High_Correlation_Features_Classification",
  "Descriptor_Optimization_Classification",
  "lasso_CL",
  "decision_tree_CL",
  "random_forest_CL",
  "xgb_CL",
  "lgb_CL",
  "rfe_CL",
  "select_from_model_CL",
  "Feature_Combination_Search_Classification",
  "Hyperparameter_Grid_Search_Classification",
  "Model_Validation_Classification",
  "QSARDataSplit_Classification",

  // Screener
  "QSARDBScreener",
  "QSARCustomUserScreener",
]);

const LOG_H_OPEN = 120; // 펼쳤을 때 로그 패널 높이 (px) — 기존 TEXT_WIDGET_HEIGHT
const TOGGLE_BAR_H = 28; // 토글 바 높이 (px)
const WIDGET_PAD = 14; // 위젯 상하 여백 합산 (px) — computeSize 오차 보정용

// ══════════════════════════════════════════════════════════════════
// 노드 크기 재계산 헬퍼
// ══════════════════════════════════════════════════════════════════

/**
 * 토글 직후 노드의 실제 높이를 다시 계산하고 캔버스에 반영한다.
 *
 * baseH를 고정 스냅샷으로 사용하지 않고, 매번 node.computeSize()로
 * 전체 높이를 구한 뒤 로그 위젯 자체 기여분을 빼서 순수 다른 위젯
 * 높이를 동적으로 계산한다. 이로써 advanced 위젯 토글 후에도
 * 노드 크기가 올바르게 반영된다.
 *
 * @param {object}  node       - LiteGraph 노드
 * @param {object}  logWidget  - 로그 DOM 위젯 (기여분 제거에 사용)
 * @param {boolean} open       - true = 열림, false = 닫힘
 */
function applyNodeHeight(node, logWidget, open) {
  if (!node) return;

  // 로그 위젯의 현재 기여분 (isOpen 상태 기준으로 이미 반영돼 있음)
  // → computeSize() 합산에서 이 값을 빼고 목표 로그 높이로 교체
  const curLogH =
    logWidget?.computeSize?.()?.[1] ?? TOGGLE_BAR_H + LOG_H_OPEN + WIDGET_PAD;

  // advanced 토글 등 다른 위젯 변화가 반영된 최신 전체 높이
  const totalH = node.computeSize()[1];

  // 순수 다른 위젯 높이 = 전체 - 현재 로그 기여분
  const baseH = totalH - curLogH;

  const logH = TOGGLE_BAR_H + (open ? LOG_H_OPEN : 0) + WIDGET_PAD;
  const newH = Math.max(baseH + logH, 60);
  const w = node.size[0];

  if (node.setSize) {
    node.setSize([w, newH]);
  } else {
    node.size = [w, newH];
  }

  if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
  if (node.graph) node.graph.setDirtyCanvas(true, true);
}

// ══════════════════════════════════════════════════════════════════
// 줄별 색상 규칙 — 라이트 테마 (밝은 배경 기준)
// ══════════════════════════════════════════════════════════════════

/**
 * 리포트 한 줄의 텍스트를 받아 색상 코드를 반환.
 * 라이트 테마: 밝은 배경(#f9f9f9) 위에서 가독성 있는 어두운 색상 계열.
 */
function lineColor(line) {
  const t = line.trimStart();
  if (
    t.startsWith("❌") ||
    t.includes("Error") ||
    t.includes("Traceback") ||
    t.includes("Failed") ||
    t.includes("Blocked") ||
    t.includes("failed") ||
    t.includes("blocked")
  )
    return "#b91c1c"; // 진한 빨강 — 오류/차단
  if (t.startsWith("✅")) return "#15803d"; // 진한 초록 — 통과
  if (t.startsWith("🔶") || t.startsWith("⚠️")) return "#b45309"; // 진한 노랑/갈색 — 경고
  if (t.startsWith("💡") || t.startsWith("⏭️")) return "#64748b"; // 중간 회색 — 힌트/스킵
  if (t.startsWith("💰")) return "#0369a1"; // 진한 파랑 — 절감 정보
  if (t.startsWith("🔎") || t.startsWith("  📋") || t.startsWith("  🗂️"))
    return "#6d28d9"; // 진한 보라 — 설정/경로 정보
  if (
    t.startsWith("📊") ||
    t.startsWith("🧬") ||
    t.startsWith("═") ||
    t.startsWith("─")
  )
    return "#1e293b"; // 연한 회색 — 구분선/헤더
  return "#1e293b"; // 기본 — 거의 검정에 가까운 어두운 색
}

/**
 * 텍스트를 줄 단위로 파싱해 색상이 적용된 HTML을 반환.
 */
function textToColoredHTML(text) {
  if (!text) return `<span style="color:#94a3b8">— No output —</span>`;
  return text
    .split("\n")
    .map((line) => {
      const color = lineColor(line);
      const safe = line
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
      return `<span style="color:${color}">${safe}</span>`;
    })
    .join("\n");
}

/**
 * 텍스트에서 통과/차단 수 추출 → 요약 객체 반환.
 */
function parseSummary(text) {
  if (!text) return null;
  const pm = text.match(/Passed[:\s]+(\d+)/i);
  const bm = text.match(/Blocked[:\s]+(\d+)/i);
  if (!pm && !bm) return null;
  return {
    passed: pm ? parseInt(pm[1]) : 0,
    blocked: bm ? parseInt(bm[1]) : 0,
  };
}

// ══════════════════════════════════════════════════════════════════
// DOM 위젯 — 토글 바 + 로그 패널  (라이트 테마)
// ══════════════════════════════════════════════════════════════════

/**
 * 노드에 토글 바(버튼 + 요약 배지) + 로그 패널(pre)을 붙인다.
 * 색상 테마: ComfyQSAR_TEXT.js 의 라이트 팔레트 (#f9f9f9 배경, black 계열 글자)
 */
function attachLogPanel(node) {
  // ── 최상위 컨테이너 ───────────────────────────────────────────
  const root = document.createElement("div");
  root.style.cssText = `
    width:100%; margin-top:10px; margin-bottom:4px;
    display:flex; flex-direction:column; gap:0;
  `;

  // ── 토글 바 ───────────────────────────────────────────────────
  const bar = document.createElement("div");
  bar.style.cssText = `
    display:flex; align-items:center; gap:6px;
    padding:0 8px; height:${TOGGLE_BAR_H}px;
    background:#f0f0f0; border-radius:6px 6px 0 0;
    border:1px solid #e0e0e0; cursor:pointer;
    user-select:none; flex-shrink:0;
  `;

  const btnIcon = document.createElement("span");
  btnIcon.style.cssText = "font-size:11px; color:#4b5563; flex-shrink:0;";
  btnIcon.textContent = "▼";

  const btnLabel = document.createElement("span");
  btnLabel.style.cssText = "font-size:10px; color:#6b7280; flex-shrink:0;";
  btnLabel.textContent = "Execution Log";

  const summaryBadges = document.createElement("span");
  summaryBadges.style.cssText =
    "display:flex; gap:4px; margin-left:4px; flex:1;";

  const spacer = document.createElement("span");
  spacer.style.flex = "1";

  const hintSpan = document.createElement("span");
  hintSpan.style.cssText = "font-size:9px; color:#9ca3af; flex-shrink:0;";
  hintSpan.textContent = "Click to toggle";

  bar.append(btnIcon, btnLabel, summaryBadges, spacer, hintSpan);

  // ── 로그 패널 ─────────────────────────────────────────────────
  const panel = document.createElement("pre");
  panel.style.cssText = `
    width:100%; height:${LOG_H_OPEN}px;
    margin:0; padding:8px 10px;
    box-sizing:border-box;
    font-family:'Consolas','Menlo',monospace;
    font-size:10px; line-height:1.55;
    color:#1e293b; background:#f9f9f9;
    border:1px solid #e0e0e0; border-top:none;
    border-radius:0 0 6px 6px;
    overflow-y:auto; overflow-x:hidden;
    white-space:pre-wrap; word-break:break-word;
    transition:height 0.18s ease;
  `;
  panel.innerHTML = `<span style="color:#9ca3af">⏳ Waiting for execution…</span>`;

  root.append(bar, panel);

  // ── 토글 상태 + 노드 크기 자동 조절 ──────────────────────────
  let isOpen = true;

  // widget 참조는 DOM 위젯 등록 후 아래에서 할당됨
  let _widget = null;

  function setOpen(open) {
    isOpen = open;
    if (open) {
      panel.style.height = `${LOG_H_OPEN}px`;
      panel.style.display = "block";
      panel.style.borderTopWidth = "0";
      btnIcon.textContent = "▼";
      hintSpan.textContent = "Click to hide";
    } else {
      panel.style.height = "0";
      panel.style.display = "none";
      btnIcon.textContent = "▶";
      hintSpan.textContent = "Click to expand";
    }
    // advanced 토글 등으로 인한 위젯 변화가 반영된 뒤 크기 계산
    applyNodeHeight(node, _widget, open);
    setTimeout(() => applyNodeHeight(node, _widget, open), 220);
  }

  bar.addEventListener("click", () => setOpen(!isOpen));

  // ── DOM 위젯 등록 ─────────────────────────────────────────────
  const widget = node.addDOMWidget("qsar_log_panel", "div", root, {
    serialize: false,
    getValue: () => panel.textContent,
    setValue: () => {},
  });

  widget.computeSize = () => [
    root.offsetWidth || 300,
    TOGGLE_BAR_H + (isOpen ? LOG_H_OPEN : 0) + WIDGET_PAD,
  ];

  // widget 참조를 setOpen에서 사용할 수 있도록 할당
  _widget = widget;

  // ── 퍼블릭 API ────────────────────────────────────────────────
  node._logPanel = {
    update(text) {
      panel.innerHTML = textToColoredHTML(text);
      panel.scrollTop = 0;

      // 요약 배지 업데이트
      summaryBadges.innerHTML = "";
      const s = parseSummary(text);
      if (s) {
        if (s.passed > 0)
          summaryBadges.appendChild(mkBadge(`✅ ${s.passed}`, "#15803d"));
        if (s.blocked > 0)
          summaryBadges.appendChild(mkBadge(`❌ ${s.blocked}`, "#b91c1c"));
      }

      // 에러가 있으면 자동으로 열기 (닫혀 있던 경우)
      const hasError =
        text &&
        (text.includes("Error") ||
          text.includes("Traceback") ||
          text.includes("Failed") ||
          text.includes("failed") ||
          text.includes("blocked") ||
          text.includes("Blocked"));
      if (hasError && !isOpen) setOpen(true);
    },
    setOpen,
  };

  return widget;
}

function mkBadge(label, color) {
  const b = document.createElement("span");
  b.style.cssText = `
    background:${color}18; border:1px solid ${color}66;
    border-radius:4px; padding:1px 6px;
    font-size:9px; font-weight:bold; color:${color};
    white-space:nowrap;
  `;
  b.textContent = label;
  return b;
}

// ══════════════════════════════════════════════════════════════════
// 노드 업데이트 헬퍼
// ══════════════════════════════════════════════════════════════════

function updateLog(node, message) {
  if (!node._logPanel) return;

  let text = null;
  if (message && message.text) {
    text = Array.isArray(message.text) ? message.text.join("") : message.text;
  }
  if (text && text.trim() !== "") {
    node._logPanel.update(text);
  } else if (text !== null) {
    node._logPanel.update("No output available");
  }
}

// ══════════════════════════════════════════════════════════════════
// ComfyUI 확장 등록
// ══════════════════════════════════════════════════════════════════

app.registerExtension({
  name: "ComfyQSAR_TEXT_STABLE_FIX",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGET_NODES.includes(nodeData.name)) return;

    // ── 노드 생성 ─────────────────────────────────────────────
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      attachLogPanel(this);
      requestAnimationFrame(() => this._logPanel?.setOpen(true));
    };

    // ── 노드 실행 완료 ────────────────────────────────────────
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      updateLog(this, message);
    };

    // ── 노드 복원 시 재생성 ───────────────────────────────────
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      onConfigure?.apply(this, arguments);
      if (!this._logPanel) attachLogPanel(this);
      requestAnimationFrame(() => this._logPanel?.setOpen(true));
    };
  },
});
