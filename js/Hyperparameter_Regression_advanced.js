import { app } from "../../../scripts/app.js";

// ComfyUI 표준 위젯 숨김 타입
const HIDDEN_TYPE = "hiddenWidget";

// 원본 위젯 속성 저장
const origProps = {};

// 알고리즘별 관련 파라미터 매핑
function getAlgorithmParameters(algorithm) {
  const parameterMappings = {
    random_forest: [
      "rf_n_estimators",
      "rf_max_depth",
      "rf_min_samples_split",
      "rf_min_samples_leaf",
      "rf_bootstrap",
    ],
    decision_tree: [
      "dt_max_depth",
      "dt_min_samples_split",
      "dt_min_samples_leaf",
      "dt_criterion",
    ],
    lasso: ["lasso_alpha"],
    ridge: ["ridge_alpha"],
    elasticnet: ["elastic_alpha", "elastic_l1_ratio"],
    svr: ["svm_C", "svm_kernel", "svm_gamma", "svm_epsilon"],
    xgboost: [
      "xgb_n_estimators",
      "xgb_learning_rate",
      "xgb_max_depth",
      "xgb_subsample",
      "xgb_reg_alpha",
      "xgb_reg_lambda",
    ],
    lightgbm: [
      "lgb_n_estimators",
      "lgb_learning_rate",
      "lgb_num_leaves",
      "lgb_max_depth",
      "lgb_reg_alpha",
      "lgb_reg_lambda",
    ],
  };
  return parameterMappings[algorithm] || [];
}

// advanced 모드에서 표시할 공통 파라미터 (알고리즘 무관)
function getAdvancedCommonParams() {
  return ["num_cores", "cv_splits", "verbose", "random_state"];
}

// 모든 조건부 파라미터 목록 (알고리즘별 + 공통 advanced)
function getAllConditionalParams() {
  return [
    // 공통 advanced 파라미터
    "num_cores",
    "cv_splits",
    "verbose",
    "random_state",
    // 알고리즘별 하이퍼파라미터
    "rf_n_estimators",
    "rf_max_depth",
    "rf_min_samples_split",
    "rf_min_samples_leaf",
    "rf_bootstrap",
    "dt_max_depth",
    "dt_min_samples_split",
    "dt_min_samples_leaf",
    "dt_criterion",
    "lasso_alpha",
    "ridge_alpha",
    "elastic_alpha",
    "elastic_l1_ratio",
    "svm_C",
    "svm_kernel",
    "svm_gamma",
    "svm_epsilon",
    "xgb_n_estimators",
    "xgb_learning_rate",
    "xgb_max_depth",
    "xgb_subsample",
    "xgb_reg_alpha",
    "xgb_reg_lambda",
    "lgb_n_estimators",
    "lgb_learning_rate",
    "lgb_num_leaves",
    "lgb_max_depth",
    "lgb_reg_alpha",
    "lgb_reg_lambda",
  ];
}

// 위젯 토글 함수 (ComfyUI 표준 방식)
function toggleWidget(node, widget, show = false, nodeKey = "") {
  if (!widget) {
    console.warn(`[HyperparamTuning] toggleWidget called with null widget`);
    return;
  }

  const propKey = `${nodeKey}_${widget.name}`;

  // 원본 속성 저장 (한 번만)
  if (!origProps[propKey]) {
    origProps[propKey] = {
      origType: widget.type,
      origComputeSize: widget.computeSize,
      origComputedHeight: widget.computedHeight || 0,
      origDisabled: widget.disabled || false,
      origDisplay: widget.options?.display || "number",
    };
    console.log(
      `[HyperparamTuning] 💾 Saved props for "${widget.name}": type=${widget.type}`,
    );
  }

  const props = origProps[propKey];

  if (show) {
    widget.type = props.origType;
    widget.computeSize = props.origComputeSize;
    widget.computedHeight = props.origComputedHeight;
    widget.disabled = props.origDisabled;
    widget.visible = true;
    widget.hidden = false;
    if (widget.options && props.origDisplay) {
      widget.options.display = props.origDisplay;
    }
    if (widget.element) widget.element.style.display = "";
    console.log(`[HyperparamTuning] ✅ Shown: ${widget.name}`);
  } else {
    widget.type = HIDDEN_TYPE;
    widget.computeSize = () => [0, -4];
    widget.computedHeight = 0;
    widget.disabled = true;
    widget.visible = false;
    widget.hidden = true;
    if (widget.options && widget.options.display === "slider") {
      widget.options.display = "number";
    }
    if (widget.element) widget.element.style.display = "none";
    console.log(`[HyperparamTuning] ❌ Hidden: ${widget.name}`);
  }
}

// 노드 크기 업데이트
function updateNodeSize(node) {
  setTimeout(() => {
    const newSize = node.computeSize();
    if (newSize[0] > 0 && newSize[1] > 0) {
      node.setSize(newSize);
      app.canvas.setDirty(true, true);
      if (app.graph?.setDirtyCanvas) {
        app.graph.setDirtyCanvas(true, true);
      }
    }
    console.log(`[HyperparamTuning] Node resized to: ${newSize}`);
  }, 20);
}

// ComfyUI 확장 등록
app.registerExtension({
  name: "ComfyQSAR_HyperparameterTuning_Regression_Advanced",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "Hyperparameter_Grid_Search_Regression") return;

    console.log(
      `[HyperparamTuning] Registering dynamic control for: ${nodeData.name}`,
    );

    const originalNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      originalNodeCreated?.apply(this, arguments);

      console.log(
        `[HyperparamTuning] Node created: ${this.title || nodeData.name}`,
      );

      // 노드별 고유 키
      const nodeKey = `${nodeData.name}_${this.id || Date.now()}`;
      this.nodeKey = nodeKey;

      const allConditionalParams = getAllConditionalParams();

      // 제어 위젯 찾기
      const advancedWidget = this.widgets.find((w) => w.name === "advanced");
      const algorithmWidget = this.widgets.find((w) => w.name === "algorithm");

      if (!advancedWidget) {
        console.warn(`[HyperparamTuning] "advanced" widget not found`);
        return;
      }
      if (!algorithmWidget) {
        console.warn(`[HyperparamTuning] "algorithm" widget not found`);
        return;
      }

      console.log(
        `[HyperparamTuning] Control widgets — advanced: ${advancedWidget.value}, algorithm: ${algorithmWidget.value}`,
      );
      console.log(
        `[HyperparamTuning] Available widgets: ${this.widgets.map((w) => w.name).join(", ")}`,
      );

      // 모든 조건부 위젯의 원본 속성 미리 저장
      allConditionalParams.forEach((paramName) => {
        const widget = this.widgets.find((w) => w.name === paramName);
        if (widget) {
          const propKey = `${nodeKey}_${paramName}`;
          origProps[propKey] = {
            origType: widget.type,
            origComputeSize: widget.computeSize,
            origComputedHeight: widget.computedHeight || 0,
            origDisabled: widget.disabled || false,
            origDisplay: widget.options?.display || "number",
          };
          console.log(
            `[HyperparamTuning] Registered widget: ${paramName} (type: ${widget.type})`,
          );
        } else {
          console.warn(
            `[HyperparamTuning] Widget not found at init: ${paramName}`,
          );
        }
      });

      // ───────────────────────────────────────────
      // 핵심 가시성 업데이트 함수
      // ───────────────────────────────────────────
      const updateParameterVisibility = (isAdvanced, algorithm) => {
        console.log(
          `[HyperparamTuning] 🔄 Updating — advanced: ${isAdvanced}, algorithm: ${algorithm}`,
        );

        const advancedCommon = getAdvancedCommonParams(); // advanced 공통 파라미터
        const algoParams = getAlgorithmParameters(algorithm); // 현재 알고리즘 파라미터

        let changedCount = 0;

        allConditionalParams.forEach((paramName) => {
          const widget = this.widgets.find((w) => w.name === paramName);
          if (!widget) return;

          let shouldShow = false;

          if (advancedCommon.includes(paramName)) {
            // 공통 advanced 파라미터: advanced 모드일 때만 표시
            shouldShow = isAdvanced;
          } else {
            // 알고리즘 하이퍼파라미터: advanced 모드 + 현재 알고리즘 소속일 때 표시
            shouldShow = isAdvanced && algoParams.includes(paramName);
          }

          console.log(
            `[HyperparamTuning] 🔧 ${paramName} → shouldShow: ${shouldShow}`,
          );
          toggleWidget(this, widget, shouldShow, nodeKey);
          changedCount++;
        });

        if (changedCount > 0) updateNodeSize(this);

        console.log(`[HyperparamTuning] ✅ Updated ${changedCount} widgets`);
      };

      // ───────────────────────────────────────────
      // "advanced" 값 변경 감지
      // ───────────────────────────────────────────
      if (!advancedWidget._valueRedefined) {
        const desc =
          Object.getOwnPropertyDescriptor(advancedWidget, "value") || {};
        let advancedValue = advancedWidget.value;

        try {
          Object.defineProperty(advancedWidget, "value", {
            get() {
              return desc.get ? desc.get.call(advancedWidget) : advancedValue;
            },
            set(newVal) {
              console.log(
                `[HyperparamTuning] 🔀 advanced: ${advancedValue} → ${newVal}`,
              );
              if (desc.set) desc.set.call(advancedWidget, newVal);
              else advancedValue = newVal;
              updateParameterVisibility(newVal, algorithmWidget.value);
            },
          });
          advancedWidget._valueRedefined = true;
        } catch (err) {
          console.warn(
            `[HyperparamTuning] Could not redefine "advanced" widget: ${err.message}`,
          );
        }
      }

      // ───────────────────────────────────────────
      // "algorithm" 값 변경 감지
      // ───────────────────────────────────────────
      if (!algorithmWidget._valueRedefined) {
        const desc =
          Object.getOwnPropertyDescriptor(algorithmWidget, "value") || {};
        let algorithmValue = algorithmWidget.value;

        try {
          Object.defineProperty(algorithmWidget, "value", {
            get() {
              return desc.get ? desc.get.call(algorithmWidget) : algorithmValue;
            },
            set(newVal) {
              console.log(
                `[HyperparamTuning] 🔀 algorithm: ${algorithmValue} → ${newVal}`,
              );
              if (desc.set) desc.set.call(algorithmWidget, newVal);
              else algorithmValue = newVal;
              updateParameterVisibility(advancedWidget.value, newVal);
            },
          });
          algorithmWidget._valueRedefined = true;
        } catch (err) {
          console.warn(
            `[HyperparamTuning] Could not redefine "algorithm" widget: ${err.message}`,
          );
        }
      }

      // ───────────────────────────────────────────
      // 초기 상태 적용
      // ───────────────────────────────────────────
      console.log(`[HyperparamTuning] 🚀 Applying initial state...`);
      updateParameterVisibility(
        advancedWidget.value || false,
        algorithmWidget.value || "random_forest",
      );
    };

    // 노드 제거 시 origProps 정리
    const originalOnRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      console.log(`[HyperparamTuning] 🗑️ Cleaning up node: ${this.nodeKey}`);
      if (this.nodeKey) {
        Object.keys(origProps).forEach((key) => {
          if (key.startsWith(this.nodeKey)) delete origProps[key];
        });
      }
      originalOnRemoved?.apply(this, arguments);
    };
  },
});

console.log("🎯 ComfyQSAR HyperparameterTuning Advanced Extension Loaded");
